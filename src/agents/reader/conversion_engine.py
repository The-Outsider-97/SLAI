from __future__ import annotations

import os

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.agents.reader.utils.reader_error import ConversionFailureError, MergeFailureError
from src.agents.reader.utils.config_loader import get_config_section


class ConversionEngine:
    """Converts and merges parsed reader documents using safe filesystem semantics."""

    def __init__(self) -> None:
        reader_config = get_config_section("reader")
        configured = reader_config.get("supported_output_formats", ["txt", "md", "html", "xml", "json", "csv"])
        self.supported_output_formats = {fmt.lower().lstrip(".") for fmt in configured}
        self.default_output_format = str(reader_config.get("default_output_format", "txt")).lower().lstrip(".")

    def _normalize_target_format(self, target_format: str) -> str:
        normalized = str(target_format or self.default_output_format).lower().strip().lstrip(".")
        if normalized not in self.supported_output_formats:
            raise ConversionFailureError(
                source="unknown",
                target_format=normalized,
                message=f"Unsupported output format '{normalized}'",
            )
        return normalized

    def _prepare_output_path(self, output_dir: str, stem: str, target_ext: str) -> Path:
        out_dir = Path(output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        candidate = out_dir / f"{stem}.{target_ext}"
        if not candidate.exists():
            return candidate

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        return out_dir / f"{stem}_{timestamp}.{target_ext}"

    def convert(self, parsed_doc: Dict[str, Any], target_format: str, output_dir: str) -> Dict[str, Any]:
        source = str(parsed_doc.get("source", "unknown"))
        target_ext = self._normalize_target_format(target_format)

        try:
            content = str(parsed_doc.get("content", ""))
            base_name = Path(source).stem or "document"
            out_path = self._prepare_output_path(output_dir, base_name, target_ext)
            out_path.write_text(content, encoding="utf-8")
        except Exception as exc:
            raise ConversionFailureError(
                source=source,
                target_format=target_ext,
                message=f"Failed converting '{source}' to '{target_ext}': {exc}",
                cause=exc,
            ) from exc

        return {
            "status": "ok",
            "source": source,
            "target_format": target_ext,
            "output_path": str(out_path),
            "metadata": {
                "bytes_written": out_path.stat().st_size,
                "source_extension": parsed_doc.get("extension"),
            },
        }

    def merge(
        self,
        parsed_docs: List[Dict[str, Any]],
        output_format: str,
        output_dir: str,
        filename: str = "merged",
    ) -> Dict[str, Any]:
        if not parsed_docs:
            raise MergeFailureError("Merge requires at least one parsed document")

        target_ext = self._normalize_target_format(output_format)
        out_path = self._prepare_output_path(output_dir, filename, target_ext)

        try:
            segments = []
            for doc in parsed_docs:
                source = str(doc.get("source", "unknown"))
                content = str(doc.get("content", ""))
                if target_ext == "md":
                    header = f"\n\n## Source: {source}\n\n"
                else:
                    header = f"\n\n# Source: {source}\n\n"
                segments.append(header + content)
            out_path.write_text("".join(segments).strip() + "\n", encoding="utf-8")
        except Exception as exc:
            raise MergeFailureError(
                message=f"Failed merging {len(parsed_docs)} documents",
                context={"output_path": str(out_path), "output_format": target_ext},
                cause=exc,
            ) from exc

        return {
            "status": "ok",
            "output_path": str(out_path),
            "output_format": target_ext,
            "inputs": [str(d.get("source", "unknown")) for d in parsed_docs],
            "metadata": {"bytes_written": out_path.stat().st_size, "document_count": len(parsed_docs)},
        }
