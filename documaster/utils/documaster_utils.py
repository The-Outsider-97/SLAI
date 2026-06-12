"""Utility functions for DocMaster GUI and API routes."""

from __future__ import annotations

import io
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
    top_words = [
        {"text": word, "value": count}
        for word, count in Counter(normalized).most_common(30)
    ]
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
        top_words=top_words,
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
            body = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            output_path.write_text(f"<!doctype html><html><body>{body}</body></html>", encoding="utf-8")
        elif target_format == "xml":
            safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            output_path.write_text(f"<?xml version='1.0' encoding='UTF-8'?><document><content>{safe}</content></document>", encoding="utf-8")
        elif target_format == "docx":
            doc = Document()
            for line in text.splitlines() or [text]:
                doc.add_paragraph(line)
            doc.save(str(output_path))
        else:
            raise DocumentExtractionError("Supported conversion targets in the desktop GUI are TXT, HTML, XML and DOCX.")
        return output_path

    def merge_pdfs(self, paths: Iterable[str | Path], output_path: str | Path) -> Path:
        output_path = Path(output_path)
        writer = PdfWriter()
        pdf_paths = [Path(p) for p in paths]
        if len(pdf_paths) < 2:
            raise DocumentExtractionError("Select at least two PDF files to merge.")
        for path in pdf_paths:
            if path.suffix.lower() != ".pdf":
                raise DocumentExtractionError("Desktop merge currently accepts PDF files only.")
            reader = PdfReader(str(path))
            if getattr(reader, "is_encrypted", False):
                raise DocumentExtractionError(f"Encrypted PDF cannot be merged: {path.name}")
            for page in reader.pages:
                writer.add_page(page)
        with output_path.open("wb") as handle:
            writer.write(handle)
        return output_path


def format_payload(payload: Mapping[str, Any]) -> str:
    result = payload.get("result", {}) if isinstance(payload, Mapping) else {}
    lines: List[str] = []
    if payload.get("status") != "success":
        for issue in result.get("issues", []):
            lines.append(f"ERROR: {issue.get('message', issue)}")
        return "\n".join(lines) or "The request failed."

    if result.get("summary"):
        lines.append("SUMMARY\n" + str(result["summary"]))
    if result.get("answer"):
        lines.append("ANSWER\n" + str(result["answer"]))
    if result.get("key_points"):
        lines.append("KEY POINTS")
        lines.extend(f"- {item}" for item in result["key_points"])
    if result.get("issues"):
        lines.append("ISSUES")
        lines.extend(f"- {i.get('severity', 'info')}: {i.get('message', i)}" if isinstance(i, dict) else f"- {i}" for i in result["issues"])
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
        f"Words: {metadata.get('word_count', 0)} | Characters: {metadata.get('char_count', 0)} | Type: {metadata.get('detected_type', '-')}",
        f"Confidence: {result.get('confidence', 0.0):.2f}",
        f"Privacy: stored={privacy.get('stored', False)} | retention={privacy.get('retention', 'none_by_default')}",
    ]
    if warnings:
        footer.append("Warnings: " + "; ".join(map(str, warnings)))
    return "\n\n".join(lines + footer)
