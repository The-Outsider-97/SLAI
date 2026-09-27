"""Canonical LANTRA corpus discovery, extraction, normalization, and segmentation.

This module is deliberately independent of the LANTRA model/runtime.  It is the
single source of truth for raw corpus I/O used by the library analyzer,
curriculum builder, and trainer.  Importing it must not import PyTorch, agents,
or the language model.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import posixpath
import re
import unicodedata
import zipfile
import xml.etree.ElementTree as ET

from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from logs.logger import get_logger

LOGGER = get_logger("LANTRA Corpus")

DEFAULT_RAW_TEXT_CANDIDATES: Tuple[str, ...] = (
    "data/library",
    "data/raw/lantra_corpus",
    "data/raw/language_corpus",
    "data/raw/language",
    "data/lantra_corpus",
    "data/language_corpus",
)

RAW_TEXT_EXTENSIONS = frozenset({
    ".txt", ".text", ".md", ".markdown",
    ".html", ".htm", ".xhtml",
    ".docx", ".epub", ".pdf",
    ".json", ".jsonl",
})


class LantraCorpusError(RuntimeError):
    """Raised when a supported corpus source cannot be safely extracted."""


@dataclass(frozen=True)
class ExtractedRawDocument:
    source_path: str
    source_type: str
    source_sha256: str
    text: str
    title: Optional[str]
    extractor: str
    logical_index: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def discover_raw_text_files(configured_paths: Sequence[str]) -> List[Path]:
    # Explicit --raw-text paths override the automatic raw corpus roots.
    # With no explicit paths, data/library remains the default corpus.
    if configured_paths:
        candidates: List[Path] = [
            Path(value) for value in configured_paths
        ]
    else:
        candidates = [Path("data/library")]

        env_path = os.getenv("SLAI_LANTRA_RAW_TEXT", "").strip()
        if env_path:
            candidates.append(Path(env_path))
        else:
            candidates.extend(
                Path(value)
                for value in DEFAULT_RAW_TEXT_CANDIDATES
                if value != "data/library"
            )

    files: List[Path] = []
    seen: set[str] = set()

    for candidate in candidates:
        if not candidate.exists():
            continue

        discovered = (
            [candidate]
            if candidate.is_file()
            else sorted(candidate.rglob("*"))
        )

        for item in discovered:
            if (
                not item.is_file()
                or item.suffix.lower() not in RAW_TEXT_EXTENSIONS
            ):
                continue

            if (
                item.name.startswith("slai_corpus_")
                and item.name.endswith(".manifest.json")
            ):
                continue

            key = str(item.resolve())

            if key not in seen:
                seen.add(key)
                files.append(item)

    return sorted(files, key=lambda item: str(item))


def _decode_document_bytes(payload: bytes) -> str:
    """Decode ordinary text files without introducing a charset dependency."""
    if not payload:
        return ""
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp1252"):
        try:
            return payload.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return payload.decode("utf-8", errors="replace")


def _normalize_document_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    compact: List[str] = []
    blank = False
    for line in lines:
        if line:
            compact.append(line)
            blank = False
        elif compact and not blank:
            compact.append("")
            blank = True
    return "\n".join(compact).strip()


def _normalized_text_hash(text: str) -> str:
    canonical = " ".join(_normalize_document_text(text).split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _clean_markdown_text(text: str) -> str:
    # Preserve the actual prose/code while removing high-frequency presentation
    # syntax that otherwise becomes corpus noise.
    text = re.sub(r"(?ms)^---\s*$.*?^---\s*$", " ", text, count=1)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    text = re.sub(r"(?m)^\s*>\s?", "", text)
    text = re.sub(r"(?m)^\s*[-*_]{3,}\s*$", "", text)
    text = text.replace("```", "").replace("~~~", "")
    return _normalize_document_text(text)


class _VisibleHTMLTextExtractor(HTMLParser):
    _SKIP_TAGS = frozenset({"script", "style", "noscript", "svg", "canvas", "template"})
    _BLOCK_TAGS = frozenset({
        "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt",
        "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6",
        "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section", "table",
        "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.title_parts: List[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if not self._skip_depth and tag.lower() in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        value = data.strip()
        if not value:
            return
        if self._in_title:
            self.title_parts.append(value)
        self.parts.append(value)
        self.parts.append(" ")

    def result(self) -> Tuple[str, Optional[str]]:
        text = _normalize_document_text("".join(self.parts))
        title = " ".join(self.title_parts).strip() or None
        return text, title


def _extract_html_text(payload: str) -> Tuple[str, Optional[str]]:
    parser = _VisibleHTMLTextExtractor()
    parser.feed(payload)
    parser.close()
    return parser.result()


def _extract_docx(path: Path) -> Tuple[str, Optional[str], Dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        try:
            root = ET.fromstring(archive.read("word/document.xml"))
        except KeyError as exc:
            raise LantraCorpusError(f"DOCX is missing word/document.xml: {path}") from exc
        paragraphs: List[str] = []
        for paragraph in root.iter():
            if not paragraph.tag.endswith("}p"):
                continue
            text_nodes = [node.text or "" for node in paragraph.iter() if node.tag.endswith("}t")]
            paragraph_text = "".join(text_nodes).strip()
            if paragraph_text:
                paragraphs.append(paragraph_text)

        title: Optional[str] = None
        try:
            core_root = ET.fromstring(archive.read("docProps/core.xml"))
            for node in core_root.iter():
                if node.tag.endswith("}title") and (node.text or "").strip():
                    title = (node.text or "").strip()
                    break
        except (KeyError, ET.ParseError):
            pass

    return _normalize_document_text("\n\n".join(paragraphs)), title, {"paragraphs": len(paragraphs)}


def _safe_epub_member(base_dir: str, href: str) -> str:
    member = posixpath.normpath(posixpath.join(base_dir, href.split("#", 1)[0]))
    if member.startswith("../") or member.startswith("/"):
        raise LantraCorpusError(f"Unsafe EPUB member path: {href!r}")
    return member


def _extract_epub(path: Path) -> Tuple[str, Optional[str], Dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        opf_path: Optional[str] = None
        if "META-INF/container.xml" in names:
            container_root = ET.fromstring(archive.read("META-INF/container.xml"))
            rootfile = container_root.find(".//{*}rootfile")
            if rootfile is not None:
                opf_path = rootfile.attrib.get("full-path")

        ordered_members: List[str] = []
        title: Optional[str] = None
        if opf_path and opf_path in names:
            package_root = ET.fromstring(archive.read(opf_path))
            base_dir = posixpath.dirname(opf_path)
            manifest: Dict[str, Tuple[str, str, str]] = {}
            for item in package_root.findall(".//{*}manifest/{*}item"):
                item_id = item.attrib.get("id", "")
                href = item.attrib.get("href", "")
                media_type = item.attrib.get("media-type", "")
                properties = item.attrib.get("properties", "")
                if item_id and href:
                    manifest[item_id] = (href, media_type, properties)
            for itemref in package_root.findall(".//{*}spine/{*}itemref"):
                item_id = itemref.attrib.get("idref", "")
                if item_id not in manifest:
                    continue
                href, media_type, properties = manifest[item_id]
                if "nav" in properties.split():
                    continue
                if media_type in {"application/xhtml+xml", "text/html"} or href.lower().endswith((".xhtml", ".html", ".htm")):
                    member = _safe_epub_member(base_dir, href)
                    if member in names:
                        ordered_members.append(member)
            title_node = package_root.find(".//{http://purl.org/dc/elements/1.1/}title")
            if title_node is not None and (title_node.text or "").strip():
                title = (title_node.text or "").strip()

        if not ordered_members:
            ordered_members = sorted(
                name for name in names if name.lower().endswith((".xhtml", ".html", ".htm"))
            )

        chapters: List[str] = []
        seen_members: set[str] = set()
        for member in ordered_members:
            if member in seen_members:
                continue
            seen_members.add(member)
            text, chapter_title = _extract_html_text(_decode_document_bytes(archive.read(member)))
            if text:
                if chapter_title and not title:
                    title = chapter_title
                chapters.append(text)

    return _normalize_document_text("\n\n".join(chapters)), title, {"chapters": len(chapters)}


def _extract_pdf(path: Path) -> Tuple[str, Optional[str], Dict[str, Any]]:
    try:
        pypdf = importlib.import_module("pypdf")
    except ImportError as exc:
        raise LantraCorpusError(
            "PDF corpus ingestion requires pypdf. SLAI's root requirements.txt already declares pypdf; "
            "install the project requirements in the active virtual environment."
        ) from exc

    try:
        reader = pypdf.PdfReader(str(path), strict=False)
    except Exception as exc:
        raise LantraCorpusError(f"Unable to open PDF {path}: {exc}") from exc

    if getattr(reader, "is_encrypted", False):
        try:
            decrypted = reader.decrypt("")
        except Exception as exc:
            raise LantraCorpusError(f"Encrypted PDF cannot be decrypted without a password: {path}") from exc
        if not decrypted:
            raise LantraCorpusError(f"Encrypted PDF requires a password and cannot be used for training: {path}")

    pages: List[str] = []
    failed_pages = 0
    for page_number, page in enumerate(reader.pages, 1):
        try:
            try:
                text = page.extract_text(extraction_mode="layout") or ""
            except TypeError:
                text = page.extract_text() or ""
        except Exception as exc:
            failed_pages += 1
            LOGGER.warning("PDF text extraction failed for %s page %d: %s", path, page_number, exc)
            continue
        if text.strip():
            # Repair common line-end hyphenation before paragraph normalization.
            text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
            pages.append(text)

    title: Optional[str] = None
    metadata = getattr(reader, "metadata", None)
    if metadata is not None:
        candidate = getattr(metadata, "title", None)
        if candidate is None and isinstance(metadata, Mapping):
            candidate = metadata.get("/Title")
        if candidate:
            title = str(candidate).strip() or None

    return _normalize_document_text("\n\n".join(pages)), title, {
        "pages": len(getattr(reader, "pages", [])),
        "pages_with_text": len(pages),
        "failed_pages": failed_pages,
    }


def _json_text_candidates(value: Any) -> Iterator[Tuple[str, Optional[str], Dict[str, Any]]]:
    """Yield logical text documents from JSON without mistaking task labels for prose."""
    preferred = ("text", "content", "document", "body", "passage", "article", "paragraph")
    if isinstance(value, Mapping):
        emitted = False
        title = value.get("title") if isinstance(value.get("title"), str) else None
        for key in preferred:
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                emitted = True
                yield item, title, {"json_field": key}
        if not emitted:
            for key in ("records", "examples", "data", "documents", "items"):
                nested = value.get(key)
                if nested is not None:
                    yield from _json_text_candidates(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _json_text_candidates(item)
    elif isinstance(value, str) and value.strip():
        yield value, None, {"json_field": None}


def extract_raw_documents(path: Path, *, source_sha256: Optional[str] = None) -> Iterator[ExtractedRawDocument]:
    suffix = path.suffix.lower()
    source_hash = source_sha256 or sha256_file(path)
    source_type = suffix.lstrip(".") or "unknown"

    if suffix in {".txt", ".text"}:
        if path.name.startswith("slai_corpus_") and (path.with_suffix(".manifest.json").exists() or path.with_suffix(".ready").exists()):
            from src.slai_worm.storage.manifest import read_documents
            # Validate the entire committed shard before emitting any book.
            documents = read_documents(path)
            for logical_index, (body, entry) in enumerate(documents):
                yield ExtractedRawDocument(
                    str(path), source_type, entry["sha256"],
                    _normalize_document_text(body), entry["title"],
                    "slai-worm-manifest-v1", logical_index,
                    {**entry["metadata"], "gutenberg_id": entry["ebook_id"], "shard_sha256": source_hash},
                )
            return
        text = _normalize_document_text(_decode_document_bytes(path.read_bytes()))
        if text:
            yield ExtractedRawDocument(str(path), source_type, source_hash, text, None, "stdlib-text", 0)
        return

    if suffix in {".md", ".markdown"}:
        text = _clean_markdown_text(_decode_document_bytes(path.read_bytes()))
        if text:
            yield ExtractedRawDocument(str(path), source_type, source_hash, text, None, "stdlib-markdown", 0)
        return

    if suffix in {".html", ".htm", ".xhtml"}:
        text, title = _extract_html_text(_decode_document_bytes(path.read_bytes()))
        if text:
            yield ExtractedRawDocument(str(path), source_type, source_hash, text, title, "stdlib-html.parser", 0)
        return

    if suffix == ".docx":
        text, title, metadata = _extract_docx(path)
        if text:
            yield ExtractedRawDocument(str(path), source_type, source_hash, text, title, "stdlib-zipfile+xml", 0, metadata)
        return

    if suffix == ".epub":
        text, title, metadata = _extract_epub(path)
        if text:
            yield ExtractedRawDocument(str(path), source_type, source_hash, text, title, "stdlib-epub-zip+xml+html", 0, metadata)
        return

    if suffix == ".pdf":
        text, title, metadata = _extract_pdf(path)
        if text:
            yield ExtractedRawDocument(str(path), source_type, source_hash, text, title, "pypdf", 0, metadata)
        return

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            logical_index = 0
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LantraCorpusError(
                        f"Invalid raw-text JSONL in {path}:{line_number}: {exc.msg}."
                    ) from exc
                for text, title, metadata in _json_text_candidates(payload):
                    metadata = {**metadata, "jsonl_line": line_number}
                    yield ExtractedRawDocument(
                        str(path), source_type, source_hash, _normalize_document_text(text), title,
                        "stdlib-json", logical_index, metadata,
                    )
                    logical_index += 1
        return

    if suffix == ".json":
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise LantraCorpusError(f"Invalid raw-text JSON in {path}: {exc.msg}.") from exc
        for logical_index, (text, title, metadata) in enumerate(_json_text_candidates(payload)):
            yield ExtractedRawDocument(
                str(path), source_type, source_hash, _normalize_document_text(text), title,
                "stdlib-json", logical_index, metadata,
            )
        return

    raise LantraCorpusError(f"Unsupported raw corpus document type: {path}")


def segment_raw_document(text: str, *, min_chars: int, chunk_chars: int) -> Iterator[str]:
    normalized = " ".join(_normalize_document_text(text).split())
    if len(normalized) < min_chars:
        return
    if len(normalized) <= chunk_chars:
        yield normalized
        return

    words = normalized.split(" ")
    current: List[str] = []
    current_chars = 0
    for word in words:
        addition = len(word) + (1 if current else 0)
        if current and current_chars + addition > chunk_chars:
            chunk = " ".join(current).strip()
            if len(chunk) >= min_chars:
                yield chunk
            current = [word]
            current_chars = len(word)
        else:
            current.append(word)
            current_chars += addition
    if current:
        chunk = " ".join(current).strip()
        if len(chunk) >= min_chars:
            yield chunk



# Public names for consumers that should not depend on private implementation names.
def normalize_document_text(text: str) -> str:
    return _normalize_document_text(text)


def normalized_text_hash(text: str) -> str:
    return _normalized_text_hash(text)


__all__ = [
    "DEFAULT_RAW_TEXT_CANDIDATES",
    "RAW_TEXT_EXTENSIONS",
    "ExtractedRawDocument",
    "LantraCorpusError",
    "sha256_file",
    "discover_raw_text_files",
    "extract_raw_documents",
    "segment_raw_document",
    "normalize_document_text",
    "normalized_text_hash",
]
