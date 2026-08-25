"""Utility functions for DocuMaster GUI, conversion, analysis, and PDF merge plans."""

from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from docx import Document
from pypdf import PdfReader, PdfWriter

from ..services.document_extractor import DocumentExtractionError, DocumentExtractor


@dataclass
class DocumentStats:
    filename: str = ""
    word_count: int = 0
    char_count: int = 0
    sentence_count: int = 0
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0
    unique_word_count: int = 0
    readability_score: float = 0.0
    reading_level: str = "Unknown"
    top_words: List[Dict[str, Any]] = field(default_factory=list)
    preview: str = ""


@dataclass
class MergePageItem:
    source_file: str
    source_file_index: int
    page_number: int
    included: bool
    user_order_index: int
    page_label: str = ""
    preview_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MergePlan:
    status: str
    pages: List[MergePageItem]
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "pages": [page.to_dict() for page in self.pages],
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def count_syllables(word: str) -> int:
    word = word.lower()
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", word)
    word = re.sub(r"^y", "", word)
    syllables = re.findall(r"[aeiouy]{1,2}", word)
    return max(1, len(syllables))


def analyze_text(text: str, *, filename: str = "") -> DocumentStats:
    clean = re.sub(r"\s+", " ", text or "").strip()
    words = clean.split() if clean else []
    sentences = re.findall(r"[^.!?]+[.!?]+(?:\s|$)", clean) or ([clean] if clean else [])
    word_lengths = [len(w) for w in words]
    syllables = sum(count_syllables(w) for w in words)
    normalized = [re.sub(r"[^A-Za-zÀ-ÿ0-9]", "", w).lower() for w in words]
    normalized = [w for w in normalized if len(w) > 1]
    avg_sentence_length = len(words) / max(1, len(sentences))
    readability = 0.0
    if words and sentences:
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllables / len(words)))
    if readability >= 80:
        level = "Easy"
    elif readability >= 60:
        level = "Standard"
    elif readability >= 30:
        level = "Difficult"
    else:
        level = "Very Difficult"
    return DocumentStats(
        filename=filename,
        word_count=len(words),
        char_count=len(text or ""),
        sentence_count=len(sentences),
        avg_word_length=round(sum(word_lengths) / max(1, len(word_lengths)), 2),
        avg_sentence_length=round(avg_sentence_length, 2),
        unique_word_count=len(set(normalized)),
        readability_score=round(readability, 1),
        reading_level=level,
        top_words=[{"text": word, "value": count} for word, count in Counter(normalized).most_common(30)],
        preview=(text[:1000] + ("..." if len(text) > 1000 else "")) if text else "",
    )


class DocMasterFileService:
    """Document utilities kept separate from the GUI."""

    def __init__(self, extractor: Optional[DocumentExtractor] = None) -> None:
        self.extractor = extractor or DocumentExtractor()

    def count_file(self, path: str | Path) -> DocumentStats:
        extracted = self.extractor.extract_path(path)
        return analyze_text(extracted.text, filename=extracted.filename)

    def extract_text(self, path: str | Path) -> str:
        return self.extractor.extract_path(path).text

    def convert_file(self, path: str | Path, target_format: str, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        source = Path(path)
        target_format = target_format.lower().strip().lstrip(".")
        text = self.extract_text(source)
        output_path = output_dir / f"{source.stem}_converted.{target_format}"

        if target_format == "txt":
            output_path.write_text(text, encoding="utf-8")
        elif target_format == "html":
            output_path.write_text(f"<!doctype html><html><body>{html.escape(text).replace(chr(10), '<br>')}</body></html>", encoding="utf-8")
        elif target_format == "xml":
            output_path.write_text(f"<?xml version='1.0' encoding='UTF-8'?><document><content>{html.escape(text)}</content></document>", encoding="utf-8")
        elif target_format == "docx":
            doc = Document()
            for line in text.splitlines() or [text]:
                doc.add_paragraph(line)
            doc.save(str(output_path))
        else:
            raise DocumentExtractionError("Supported conversion targets in the desktop GUI are TXT, HTML, XML and DOCX.")
        return output_path

    # ------------------------------------------------------------------
    # Improved PDF merge/page organizer workflow
    # ------------------------------------------------------------------
    def build_pdf_merge_plan(self, paths: Iterable[str | Path]) -> MergePlan:
        pdf_paths = [Path(p) for p in paths]
        if len(pdf_paths) < 2:
            raise DocumentExtractionError("Select at least two PDF files to build a merge plan.")
        pages: List[MergePageItem] = []
        warnings: List[str] = []
        order_index = 0
        for source_index, path in enumerate(pdf_paths):
            self._validate_pdf_path(path)
            reader = PdfReader(str(path))
            if getattr(reader, "is_encrypted", False):
                raise DocumentExtractionError(f"Encrypted PDF cannot be merged: {path.name}")
            for page_idx, page in enumerate(reader.pages):
                preview = ""
                try:
                    preview = (page.extract_text() or "").strip().replace("\n", " ")[:160]
                except Exception:  # noqa: BLE001
                    warnings.append(f"Preview text could not be extracted for {path.name}, page {page_idx + 1}.")
                pages.append(
                    MergePageItem(
                        source_file=str(path),
                        source_file_index=source_index,
                        page_number=page_idx + 1,
                        included=True,
                        user_order_index=order_index,
                        page_label=f"{path.name} · page {page_idx + 1}",
                        preview_metadata={"preview": preview, "source_name": path.name},
                    )
                )
                order_index += 1
        return MergePlan(
            status="success",
            pages=pages,
            warnings=list(dict.fromkeys(warnings)),
            metadata={"file_count": len(pdf_paths), "page_count": len(pages), "included_page_count": len(pages)},
        )

    def merge_pdfs_with_plan(self, plan: Mapping[str, Any] | MergePlan | Sequence[Mapping[str, Any]], output_path: str | Path) -> Path:
        page_items = self.validate_merge_plan(plan)
        included = sorted([item for item in page_items if item.included], key=lambda item: item.user_order_index)
        if not included:
            raise DocumentExtractionError("The merge plan has no included pages.")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = PdfWriter()
        reader_cache: Dict[str, PdfReader] = {}
        for item in included:
            reader = reader_cache.get(item.source_file)
            if reader is None:
                reader = PdfReader(item.source_file)
                if getattr(reader, "is_encrypted", False):
                    raise DocumentExtractionError(f"Encrypted PDF cannot be merged: {Path(item.source_file).name}")
                reader_cache[item.source_file] = reader
            zero_index = item.page_number - 1
            if zero_index < 0 or zero_index >= len(reader.pages):
                raise DocumentExtractionError(f"Invalid page selection for {Path(item.source_file).name}: page {item.page_number}.")
            writer.add_page(reader.pages[zero_index])
        with output.open("wb") as handle:
            writer.write(handle)
        return output

    def merge_pdfs(self, paths: Iterable[str | Path], output_path: str | Path) -> Path:
        """Backward-compatible merge: all pages included in selected file order."""
        plan = self.build_pdf_merge_plan(paths)
        return self.merge_pdfs_with_plan(plan, output_path)

    def validate_merge_plan(self, plan: Mapping[str, Any] | MergePlan | Sequence[Mapping[str, Any]]) -> List[MergePageItem]:
        raw_pages: Sequence[Any]
        if isinstance(plan, MergePlan):
            raw_pages = plan.pages
        elif isinstance(plan, Mapping):
            raw_pages = plan.get("pages", []) if isinstance(plan.get("pages", []), Sequence) else []
        else:
            raw_pages = plan
        if not raw_pages:
            raise DocumentExtractionError("Merge plan is empty.")

        normalized: List[MergePageItem] = []
        page_counts: Dict[str, int] = {}
        seen_orders: set[int] = set()
        for index, raw in enumerate(raw_pages):
            item = raw if isinstance(raw, MergePageItem) else self._merge_item_from_mapping(raw, index)
            path = Path(item.source_file)
            self._validate_pdf_path(path)
            if item.source_file not in page_counts:
                reader = PdfReader(str(path))
                if getattr(reader, "is_encrypted", False):
                    raise DocumentExtractionError(f"Encrypted PDF cannot be merged: {path.name}")
                page_counts[item.source_file] = len(reader.pages)
            if item.page_number < 1 or item.page_number > page_counts[item.source_file]:
                raise DocumentExtractionError(f"Invalid page number {item.page_number} for {path.name}.")
            order = item.user_order_index
            if order in seen_orders:
                order = index
            seen_orders.add(order)
            normalized.append(
                MergePageItem(
                    source_file=str(path),
                    source_file_index=max(0, int(item.source_file_index)),
                    page_number=int(item.page_number),
                    included=bool(item.included),
                    user_order_index=int(order),
                    page_label=item.page_label or f"{path.name} · page {item.page_number}",
                    preview_metadata=dict(item.preview_metadata),
                )
            )
        if not any(item.included for item in normalized):
            raise DocumentExtractionError("At least one page must be included in the merge plan.")
        return normalized

    @staticmethod
    def _merge_item_from_mapping(raw: Any, fallback_order: int) -> MergePageItem:
        if not isinstance(raw, Mapping):
            raise DocumentExtractionError("Merge plan contains an invalid page item.")
        return MergePageItem(
            source_file=str(raw.get("source_file", "")),
            source_file_index=int(raw.get("source_file_index", 0) or 0),
            page_number=int(raw.get("page_number", 0) or 0),
            included=bool(raw.get("included", True)),
            user_order_index=int(raw.get("user_order_index", fallback_order) or fallback_order),
            page_label=str(raw.get("page_label", "")),
            preview_metadata=dict(raw.get("preview_metadata", {}) if isinstance(raw.get("preview_metadata", {}), Mapping) else {}),
        )

    @staticmethod
    def _validate_pdf_path(path: Path) -> None:
        if not path.exists() or not path.is_file():
            raise DocumentExtractionError(f"PDF does not exist: {path.name}")
        if path.suffix.lower() != ".pdf":
            raise DocumentExtractionError("PDF merge accepts PDF files only.")


def format_payload(payload: Mapping[str, Any]) -> str:
    result = payload.get("result", {}) if isinstance(payload, Mapping) else {}
    lines: List[str] = []
    if payload.get("status") not in {"success", "partial"}:
        for issue in result.get("issues", []):
            lines.append(f"ERROR: {issue.get('message', issue)}" if isinstance(issue, Mapping) else f"ERROR: {issue}")
        return "\n".join(lines) or "The request failed."

    if result.get("summary"):
        lines.append("SUMMARY\n" + str(result["summary"]))
    if result.get("answer"):
        lines.append("ANSWER\n" + str(result["answer"]))
    if result.get("key_points"):
        lines.append("KEY POINTS")
        lines.extend(f"- {item}" for item in result["key_points"])
    if result.get("headings"):
        lines.append("HEADINGS")
        lines.extend(f"- {item}" for item in result["headings"][:12])
    if result.get("issues"):
        lines.append("ISSUES")
        lines.extend(f"- {i.get('severity', 'info')}: {i.get('message', i)}" if isinstance(i, Mapping) else f"- {i}" for i in result["issues"])
    if result.get("suggestions"):
        lines.append("SUGGESTIONS")
        lines.extend(f"- {item}" for item in result["suggestions"])
    if result.get("structure"):
        structure = result["structure"]
        lines.append("STRUCTURE")
        lines.append(f"Headings detected: {structure.get('heading_count', 0)}")
        for heading in structure.get("headings", [])[:10]:
            lines.append(f"- {heading}")
    if not lines:
        lines.append("No result returned.")

    metadata = payload.get("metadata", {})
    privacy = payload.get("privacy", {})
    warnings = payload.get("warnings", [])
    footer = [
        "",
        "METADATA",
        f"Filename: {payload.get('filename', '-')}",
        f"Words: {metadata.get('word_count', 0)} | Characters: {metadata.get('char_count', 0)} | Pages: {metadata.get('page_count', 0)} | Type: {metadata.get('detected_type', '-')}",
        f"Reader Agent used: {metadata.get('reader_agent_used', False)} | SLAI available: {metadata.get('slai_available', False)}",
        f"Confidence: {result.get('confidence', 0.0):.2f}",
        f"Privacy: stored={privacy.get('stored', False)} | retention={privacy.get('retention', 'none_by_default')}",
    ]
    if warnings:
        footer.append("Warnings: " + "; ".join(map(str, warnings)))
    return "\n\n".join(lines + footer)
