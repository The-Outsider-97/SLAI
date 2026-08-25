from __future__ import annotations

"""Production-grade conversion and merge engine for the Reader subsystem.

The ConversionEngine is intentionally focused on Reader-domain rendering policy:
turn parsed Reader documents into deterministic output artifacts and merge
multiple parsed documents into one artifact. Shared concerns such as safe path
construction, atomic writes, hashing, JSON serialization, timestamping, and
parsed-document validation are delegated to ``reader_helpers.py`` wherever the
shared helper layer provides them.
"""

import csv
import html
import io
import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from xml.sax.saxutils import escape as xml_escape

from .utils.config_loader import get_config_section, load_global_config
from .utils.reader_error import *
from .utils.reader_helpers import *
from .reader_memory import ReaderMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Conversion Engine")
printer = PrettyPrinter()


DEFAULT_OUTPUT_FORMATS = ("txt", "md", "html", "xml", "json", "csv")


@dataclass(frozen=True)
class ConversionArtifact:
    """Structured result metadata for one conversion artifact."""

    status: str
    source: str
    target_format: str
    output_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "source": self.source,
            "target_format": self.target_format,
            "output_path": self.output_path,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MergeArtifact:
    """Structured result metadata for one merged artifact."""

    status: str
    output_path: str
    output_format: str
    inputs: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "output_path": self.output_path,
            "output_format": self.output_format,
            "inputs": list(self.inputs),
            "metadata": dict(self.metadata),
        }


class ConversionEngine:
    """Converts and merges parsed Reader documents using safe filesystem semantics."""

    def __init__(self, memory: Optional[ReaderMemory] = None) -> None:
        self.config = load_global_config()
        self.reader_config = get_config_section("reader")
        self.ce_config = get_config_section("conversion_engine") or {}

        configured = self.reader_config.get("supported_output_formats", DEFAULT_OUTPUT_FORMATS)
        self.supported_output_formats = {
            str(fmt).lower().lstrip(".")
            for fmt in configured
            if str(fmt).strip()
        }
        if not self.supported_output_formats:
            self.supported_output_formats = set(DEFAULT_OUTPUT_FORMATS)

        self.default_output_format = str(
            self.reader_config.get("default_output_format", "txt")
        ).lower().lstrip(".")

        if self.default_output_format not in self.supported_output_formats:
            logger.warning(
                "Configured default output format '%s' is not in supported formats; falling back to txt",
                self.default_output_format,
            )
            self.default_output_format = "txt"

        self.output_encoding = str(self.ce_config.get("output_encoding", "utf-8") or "utf-8")
        self.overwrite_existing = coerce_bool(self.ce_config.get("overwrite_existing"), False)
        self.include_source_headers = coerce_bool(self.ce_config.get("include_source_headers"), True)
        self.include_metadata = coerce_bool(self.ce_config.get("include_metadata"), True)
        self.include_content_hash = coerce_bool(self.ce_config.get("include_content_hash"), True)
        self.preserve_trailing_newline = coerce_bool(self.ce_config.get("preserve_trailing_newline"), True)
        self.max_output_bytes = coerce_int(
            self.ce_config.get("max_output_bytes"),
            100 * 1024 * 1024,
            minimum=0,
        )
        self.max_merge_documents = coerce_int(
            self.ce_config.get("max_merge_documents"),
            1_000,
            minimum=1,
        )
        self.max_merge_output_chars = coerce_int(
            self.ce_config.get("max_merge_output_chars"),
            10_000_000,
            minimum=0,
        )
        self.markdown_source_header_level = coerce_int(
            self.ce_config.get("markdown_source_header_level"),
            2,
            minimum=1,
        )
        self.text_source_header_prefix = str(
            self.ce_config.get("text_source_header_prefix", "# Source:") or "# Source:"
        )
        self.html_title = str(
            self.ce_config.get("html_title", "Reader Conversion Output") or "Reader Conversion Output"
        )
        self.html_escape_content = coerce_bool(self.ce_config.get("html_escape_content"), True)
        self.xml_root_tag = str(
            self.ce_config.get("xml_root_tag", "reader_documents") or "reader_documents"
        ).strip() or "reader_documents"
        self.xml_document_tag = str(
            self.ce_config.get("xml_document_tag", "document") or "document"
        ).strip() or "document"

        csv_delimiter = str(self.ce_config.get("csv_delimiter", ",") or ",")
        self.csv_delimiter = csv_delimiter[0] if csv_delimiter else ","
        self.csv_include_header = coerce_bool(self.ce_config.get("csv_include_header"), True)
        self.csv_line_mode = str(
            self.ce_config.get("csv_line_mode", "per_line") or "per_line"
        ).strip().lower()

        raw_json_indent = self.ce_config.get("json_indent", None)
        self.json_indent = None if raw_json_indent is None else coerce_int(
            raw_json_indent,
            2,
            minimum=0,
        )

        self.conversion_settings = {
            "output_encoding": self.output_encoding,
            "overwrite_existing": self.overwrite_existing,
            "include_source_headers": self.include_source_headers,
            "include_metadata": self.include_metadata,
            "include_content_hash": self.include_content_hash,
            "preserve_trailing_newline": self.preserve_trailing_newline,
            "max_output_bytes": self.max_output_bytes,
            "max_merge_documents": self.max_merge_documents,
            "max_merge_output_chars": self.max_merge_output_chars,
            "markdown_source_header_level": self.markdown_source_header_level,
            "text_source_header_prefix": self.text_source_header_prefix,
            "html_title": self.html_title,
            "html_escape_content": self.html_escape_content,
            "xml_root_tag": self.xml_root_tag,
            "xml_document_tag": self.xml_document_tag,
            "csv_delimiter": self.csv_delimiter,
            "csv_include_header": self.csv_include_header,
            "csv_line_mode": self.csv_line_mode,
            "json_indent": self.json_indent,
        }

        self.memory = memory

        self._renderer_map = {
            "txt": self._render_txt,
            "md": self._render_md,
            "html": self._render_html,
            "xml": self._render_xml,
            "json": self._render_json,
            "csv": self._render_csv,
        }

        logger.info(
            "Conversion Engine initialized with formats: %s",
            sorted(self.supported_output_formats),
        )

    def _normalize_target_format(self, target_format: str | None) -> str:
        normalized = normalize_output_format(
            target_format,
            supported_formats=self.supported_output_formats,
            default=self.default_output_format,
        )
        if normalized not in self._renderer_map:
            raise ConversionFailureError(
                source="unknown",
                target_format=normalized,
                message=f"No Reader renderer is registered for output format '{normalized}'",
            )
        return normalized

    def _prepare_output_path(self, output_dir: str | Path, stem: str, target_ext: str) -> Path:
        return prepare_output_path(
            output_dir,
            stem,
            target_ext,
            overwrite=self.overwrite_existing,
        )

    def supported_formats(self) -> List[str]:
        """Return configured output formats with registered renderers."""

        return sorted(
            fmt for fmt in self.supported_output_formats
            if fmt in self._renderer_map
        )

    def describe(self) -> Dict[str, Any]:
        """Return a safe operational description useful for health checks."""

        return {
            "status": "ok",
            "default_output_format": self.default_output_format,
            "supported_output_formats": self.supported_formats(),
            "settings": dict(self.conversion_settings),
        }

    def _document_payload(
        self,
        parsed_doc: Mapping[str, Any],
        *,
        include_content: bool = True,
    ) -> Dict[str, Any]:
        doc = validate_parsed_document(
            parsed_doc,
            require_content=False,
            require_metadata=False,
        )
        source = str(doc.get("source", "unknown"))
        content = str(doc.get("content", ""))
        metadata = dict(doc.get("metadata") or {})

        payload: Dict[str, Any] = {
            "source": source,
            "extension": doc.get("extension"),
            "warnings": list(doc.get("warnings") or []),
        }

        if self.include_metadata:
            payload["metadata"] = metadata
        if self.include_content_hash:
            payload["content_sha256"] = sha256_text(content)
        if include_content:
            payload["content"] = content

        return payload

    def _source_header(self, source: str, target_format: str) -> str:
        if not self.include_source_headers:
            return ""
        if target_format == "md":
            level = max(1, int(self.markdown_source_header_level))
            return f"{'#' * level} Source: {source}\n\n"
        return f"{self.text_source_header_prefix} {source}\n\n"

    def _render_txt(self, parsed_doc: Mapping[str, Any]) -> str:
        doc = validate_parsed_document(
            parsed_doc,
            require_content=False,
            require_metadata=False,
        )
        source = str(doc.get("source", "unknown"))
        content = str(doc.get("content", ""))
        return self._source_header(source, "txt") + content

    def _render_md(self, parsed_doc: Mapping[str, Any]) -> str:
        doc = validate_parsed_document(
            parsed_doc,
            require_content=False,
            require_metadata=False,
        )
        source = str(doc.get("source", "unknown"))
        content = str(doc.get("content", ""))
        return self._source_header(source, "md") + content

    def _render_html(self, parsed_doc: Mapping[str, Any]) -> str:
        payload = self._document_payload(parsed_doc)
        source = str(payload.get("source", "unknown"))
        content = str(payload.get("content", ""))

        rendered_content = html.escape(content) if self.html_escape_content else content
        title = html.escape(source or self.html_title)

        metadata_html = ""
        if self.include_metadata:
            metadata = html.escape(
                stable_json_dumps(payload.get("metadata", {}), indent=2)
            )
            metadata_html = (
                f"\n    <details><summary>Metadata</summary><pre>{metadata}</pre></details>"
            )

        return (
            "<!doctype html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "  <meta charset=\"utf-8\">\n"
            f"  <title>{title}</title>\n"
            "</head>\n"
            "<body>\n"
            f"  <article data-source=\"{html.escape(source)}\">\n"
            f"    <h1>Source: {html.escape(source)}</h1>{metadata_html}\n"
            f"    <pre>{rendered_content}</pre>\n"
            "  </article>\n"
            "</body>\n"
            "</html>"
        )

    def _render_xml(self, parsed_doc: Mapping[str, Any]) -> str:
        payload = self._document_payload(parsed_doc)
        tag = self.xml_document_tag
        source = xml_escape(str(payload.get("source", "unknown")))
        extension = xml_escape(str(payload.get("extension") or ""))
        content = xml_escape(str(payload.get("content", "")))
        metadata = xml_escape(
            stable_json_dumps(payload.get("metadata", {}), indent=None)
        )
        warnings = xml_escape(
            stable_json_dumps(payload.get("warnings", []), indent=None)
        )
        content_hash = xml_escape(str(payload.get("content_sha256", "")))

        return (
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            f"<{tag} source=\"{source}\" extension=\"{extension}\">\n"
            f"  <content_sha256>{content_hash}</content_sha256>\n"
            f"  <metadata>{metadata}</metadata>\n"
            f"  <warnings>{warnings}</warnings>\n"
            f"  <content>{content}</content>\n"
            f"</{tag}>"
        )

    def _render_json(self, parsed_doc: Mapping[str, Any]) -> str:
        payload = {
            "schema": "reader.conversion.v1",
            "generated_at": utc_now().isoformat(),
            "document": self._document_payload(parsed_doc),
        }
        return stable_json_dumps(payload, indent=self.json_indent)

    def _csv_rows_for_doc(
        self,
        parsed_doc: Mapping[str, Any],
        *,
        document_index: int = 0,
    ) -> List[Dict[str, Any]]:
        doc = validate_parsed_document(
            parsed_doc,
            require_content=False,
            require_metadata=False,
        )
        source = str(doc.get("source", "unknown"))
        extension = str(doc.get("extension") or "")
        content = str(doc.get("content", ""))

        if self.csv_line_mode == "single_row":
            return [
                {
                    "document_index": document_index,
                    "source": source,
                    "extension": extension,
                    "line_number": 1,
                    "content": content,
                }
            ]

        lines = content.splitlines() or [""]
        return [
            {
                "document_index": document_index,
                "source": source,
                "extension": extension,
                "line_number": index,
                "content": line,
            }
            for index, line in enumerate(lines, start=1)
        ]

    def _render_csv_rows(self, rows: Sequence[Mapping[str, Any]]) -> str:
        output = io.StringIO(newline="")
        fieldnames = [
            "document_index",
            "source",
            "extension",
            "line_number",
            "content",
        ]
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            delimiter=self.csv_delimiter,
            extrasaction="ignore",
        )

        if self.csv_include_header:
            writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

        return output.getvalue()

    def _render_csv(self, parsed_doc: Mapping[str, Any]) -> str:
        return self._render_csv_rows(
            self._csv_rows_for_doc(parsed_doc, document_index=1)
        )

    def _render_document(self, parsed_doc: Mapping[str, Any], target_format: str) -> str:
        renderer = self._renderer_map.get(target_format)
        if renderer is None:
            raise ConversionFailureError(
                source=str(parsed_doc.get("source", "unknown")),
                target_format=target_format,
                message=f"No renderer registered for target format '{target_format}'",
            )

        rendered = renderer(parsed_doc)
        return self._finalize_text(rendered)

    def _finalize_text(self, text: str) -> str:
        if self.preserve_trailing_newline and text and not text.endswith("\n"):
            return text + "\n"
        return text

    def _enforce_output_limits(
        self,
        payload: str,
        *,
        operation: str,
        source: str,
        target_format: str,
    ) -> None:
        max_bytes = int(self.max_output_bytes)
        if max_bytes <= 0:
            return

        size = len(payload.encode(self.output_encoding, errors="replace"))
        if size > max_bytes:
            raise ConversionFailureError(
                source=source,
                target_format=target_format,
                message=f"Reader {operation} output exceeds configured max_output_bytes ({max_bytes})",
            )

    def _artifact_metadata(
        self,
        *,
        output_path: Path,
        payload: str,
        source_extension: Any = None,
        document_count: int = 1,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        stat = output_path.stat()

        metadata: Dict[str, Any] = {
            "bytes_written": int(stat.st_size),
            "document_count": document_count,
            "source_extension": source_extension,
            "generated_at": utc_now().isoformat(),
            "output_sha256": sha256_text(payload),
            "quality": text_quality_report(payload).to_dict(),
        }

        if extra:
            metadata.update(dict(extra))

        return json_safe(metadata)

    def _settings_snapshot(self) -> Dict[str, Any]:
        """Return a deterministic subset of conversion settings for cache key stability."""
        return {
            "output_encoding": self.output_encoding,
            "include_source_headers": self.include_source_headers,
            "include_metadata": self.include_metadata,
            "include_content_hash": self.include_content_hash,
            "preserve_trailing_newline": self.preserve_trailing_newline,
            "markdown_source_header_level": self.markdown_source_header_level,
            "text_source_header_prefix": self.text_source_header_prefix,
            "html_title": self.html_title,
            "html_escape_content": self.html_escape_content,
            "xml_root_tag": self.xml_root_tag,
            "xml_document_tag": self.xml_document_tag,
            "csv_delimiter": self.csv_delimiter,
            "csv_include_header": self.csv_include_header,
            "csv_line_mode": self.csv_line_mode,
            "json_indent": self.json_indent,
        }

    def convert(self, parsed_doc: Dict[str, Any], target_format: str, output_dir: str) -> Dict[str, Any]:
        doc = validate_parsed_document(parsed_doc, require_content=False, require_metadata=False)
        source = str(doc.get("source", "unknown"))
        target_ext = self._normalize_target_format(target_format)

        # Prepare cache key payload only if memory is available
        cache_key_payload = None
        if self.memory is not None:
            cache_key_payload = {
                "source": source,
                "extension": doc.get("extension"),
                "content_sha256": sha256_text(str(doc.get("content", ""))),
                "target_format": target_ext,
                "output_encoding": self.output_encoding,
                "include_source_headers": self.include_source_headers,
                "include_metadata": self.include_metadata,
                "include_content_hash": self.include_content_hash,
                "settings_snapshot": self._settings_snapshot()
            }
            cached = self.memory.get_cache(cache_key_payload, namespace="conversion")
            if cached is not None:
                rendered = cached.get("rendered_text")
                if rendered is not None:
                    # Reuse cached rendered content
                    out_path = self._prepare_output_path(output_dir, Path(source).stem, target_ext)
                    if self.overwrite_existing or not out_path.exists():
                        atomic_write_text(out_path, rendered, encoding=self.output_encoding, purpose="output")
                    artifact = ConversionArtifact(
                        status="ok",
                        source=source,
                        target_format=target_ext,
                        output_path=str(out_path),
                        metadata=self._artifact_metadata(
                            output_path=out_path,
                            payload=rendered,
                            source_extension=doc.get("extension"),
                            document_count=1,
                            extra={"renderer": target_ext, "cached": True}
                        )
                    )
                    return artifact.to_dict()

        # ---- Normal conversion (render + write) ----
        out_path = self._prepare_output_path(output_dir, Path(source).stem, target_ext)
        payload = self._render_document(doc, target_ext)
        self._enforce_output_limits(payload, operation="conversion", source=source, target_format=target_ext)
        atomic_write_text(out_path, payload, encoding=self.output_encoding, purpose="output")

        artifact = ConversionArtifact(
            status="ok",
            source=source,
            target_format=target_ext,
            output_path=str(out_path),
            metadata=self._artifact_metadata(
                output_path=out_path,
                payload=payload,
                source_extension=doc.get("extension"),
                document_count=1,
                extra={"renderer": target_ext},
            ),
        )

        # ---- Cache the rendered text for future runs ----
        if self.memory is not None and cache_key_payload is not None:
            cache_value = {
                "rendered_text": payload,
                "artifact": artifact.to_dict()
            }
            self.memory.set_cache(cache_key_payload, cache_value, namespace="conversion", ttl_seconds=None)

        return artifact.to_dict()

    def _render_merged_payload(
        self,
        docs: Sequence[Mapping[str, Any]],
        output_format: str,
    ) -> str:
        if output_format == "json":
            payload = {
                "schema": "reader.merge.v1",
                "generated_at": utc_now().isoformat(),
                "document_count": len(docs),
                "documents": [self._document_payload(doc) for doc in docs],
            }
            return stable_json_dumps(payload, indent=self.json_indent)

        if output_format == "csv":
            rows: List[Dict[str, Any]] = []
            for index, doc in enumerate(docs, start=1):
                rows.extend(self._csv_rows_for_doc(doc, document_index=index))
            return self._render_csv_rows(rows)

        if output_format == "html":
            sections: List[str] = []
            for doc in docs:
                payload = self._document_payload(doc)
                source = str(payload.get("source", "unknown"))
                content = str(payload.get("content", ""))
                rendered_content = html.escape(content) if self.html_escape_content else content

                sections.append(
                    f"  <article data-source=\"{html.escape(source)}\">\n"
                    f"    <h2>Source: {html.escape(source)}</h2>\n"
                    f"    <pre>{rendered_content}</pre>\n"
                    "  </article>"
                )

            title = html.escape(self.html_title)
            return (
                "<!doctype html>\n"
                "<html lang=\"en\">\n"
                "<head>\n"
                "  <meta charset=\"utf-8\">\n"
                f"  <title>{title}</title>\n"
                "</head>\n"
                "<body>\n"
                f"  <h1>{title}</h1>\n"
                + "\n".join(sections)
                + "\n</body>\n</html>"
            )

        if output_format == "xml":
            root = self.xml_root_tag
            tag = self.xml_document_tag
            segments = [
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
                f"<{root} count=\"{len(docs)}\">",
            ]

            for index, doc in enumerate(docs, start=1):
                payload = self._document_payload(doc)
                source = xml_escape(str(payload.get("source", "unknown")))
                extension = xml_escape(str(payload.get("extension") or ""))
                content = xml_escape(str(payload.get("content", "")))
                content_hash = xml_escape(str(payload.get("content_sha256", "")))

                segments.append(
                    f"  <{tag} index=\"{index}\" source=\"{source}\" extension=\"{extension}\">\n"
                    f"    <content_sha256>{content_hash}</content_sha256>\n"
                    f"    <content>{content}</content>\n"
                    f"  </{tag}>"
                )

            segments.append(f"</{root}>")
            return "\n".join(segments)

        segments = []
        for doc in docs:
            source = str(doc.get("source", "unknown"))
            content = str(doc.get("content", ""))
            header = self._source_header(source, output_format)
            segments.append(header + content)

        separator = "\n\n" if output_format in {"txt", "md"} else "\n"
        return separator.join(segment.strip("\n") for segment in segments)

    def merge(
        self,
        parsed_docs: List[Dict[str, Any]],
        output_format: str,
        output_dir: str,
        filename: str = "merged",
    ) -> Dict[str, Any]:
        """Merge parsed Reader documents into one output artifact."""

        if not isinstance(parsed_docs, list) or not parsed_docs:
            raise MergeFailureError("Merge requires at least one parsed document")

        if len(parsed_docs) > self.max_merge_documents:
            raise MergeFailureError(
                message=f"Merge document count exceeds configured max_merge_documents ({self.max_merge_documents})",
                context={
                    "document_count": len(parsed_docs),
                    "max_merge_documents": self.max_merge_documents,
                },
            )

        target_ext = self._normalize_target_format(output_format)
        docs = [
            validate_parsed_document(
                doc,
                require_content=False,
                require_metadata=False,
            )
            for doc in parsed_docs
        ]
        out_path = self._prepare_output_path(output_dir, filename or "merged", target_ext)

        try:
            payload = self._finalize_text(
                self._render_merged_payload(docs, target_ext)
            )

            if self.max_merge_output_chars > 0 and len(payload) > self.max_merge_output_chars:
                raise MergeFailureError(
                    message="Merged Reader output exceeds configured max_merge_output_chars",
                    context={
                        "output_chars": len(payload),
                        "max_merge_output_chars": self.max_merge_output_chars,
                        "output_format": target_ext,
                    },
                )

            self._enforce_output_limits(
                payload,
                operation="merge",
                source="merged",
                target_format=target_ext,
            )

            atomic_write_text(
                out_path,
                payload,
                encoding=self.output_encoding,
                purpose="output",
            )

        except ReaderError:
            raise
        except Exception as exc:
            raise MergeFailureError(
                message=f"Failed merging {len(parsed_docs)} documents",
                context={
                    "output_path": str(out_path),
                    "output_format": target_ext,
                    "document_count": len(parsed_docs),
                },
                cause=exc,
            ) from exc

        inputs = [str(d.get("source", "unknown")) for d in docs]
        artifact = MergeArtifact(
            status="ok",
            output_path=str(out_path),
            output_format=target_ext,
            inputs=inputs,
            metadata=self._artifact_metadata(
                output_path=out_path,
                payload=payload,
                source_extension=None,
                document_count=len(docs),
                extra={
                    "renderer": target_ext,
                    "inputs_sha256": sha256_text("\n".join(inputs)),
                },
            ),
        )
        return artifact.to_dict()


if __name__ == "__main__":
    print("\n=== Running Conversion Engine ===\n")
    printer.status("TEST", "Conversion Engine initialized", "info")

    import tempfile

    with tempfile.TemporaryDirectory(prefix="reader_conversion_test_") as tmp_dir:
        engine = ConversionEngine()
        printer.pretty("CONFIG", engine.describe(), "info")

        parsed_doc = {
            "status": "ok",
            "source": "sample_reader_document.txt",
            "extension": ".txt",
            "content": "Alpha line\nBeta line\n",
            "metadata": {"size": 21, "encoding": "utf-8"},
            "warnings": [],
        }

        second_doc = {
            "status": "ok",
            "source": "notes.md",
            "extension": ".md",
            "content": "# Notes\nRecovered content stays factual.",
            "metadata": {"size": 38, "encoding": "utf-8"},
            "warnings": ["semantic_recovery_used"],
        }

        txt_result = engine.convert(parsed_doc, "txt", tmp_dir)
        assert txt_result["status"] == "ok"
        assert Path(txt_result["output_path"]).exists()
        assert txt_result["metadata"]["bytes_written"] > 0

        json_result = engine.convert(parsed_doc, "json", tmp_dir)
        assert json_result["target_format"] == "json"

        json_payload = json.loads(
            Path(json_result["output_path"]).read_text(
                encoding=engine.output_encoding,
            )
        )
        assert json_payload["schema"] == "reader.conversion.v1"
        assert json_payload["document"]["source"] == "sample_reader_document.txt"

        html_result = engine.convert(parsed_doc, "html", tmp_dir)
        assert "<html" in Path(html_result["output_path"]).read_text(
            encoding=engine.output_encoding,
        )

        merged_md = engine.merge(
            [parsed_doc, second_doc],
            "md",
            tmp_dir,
            filename="merged_reader_output",
        )
        assert merged_md["status"] == "ok"
        assert merged_md["metadata"]["document_count"] == 2
        assert Path(merged_md["output_path"]).exists()

        merged_csv = engine.merge(
            [parsed_doc, second_doc],
            "csv",
            tmp_dir,
            filename="merged_reader_rows",
        )
        assert merged_csv["output_format"] == "csv"
        assert "source" in Path(merged_csv["output_path"]).read_text(
            encoding=engine.output_encoding,
        )

        try:
            engine.convert(parsed_doc, "unsupported_format", tmp_dir)
            raise AssertionError("Unsupported format should have raised ConversionFailureError")
        except ConversionFailureError:
            pass

    print("\n=== Test ran successfully ===\n")
