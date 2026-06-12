"""Service layer for DocuMaster document AI workflows."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from ..slai_adapter import ReaderExtraction, SLAIAdapter
from .document_extractor import DocumentExtractionError, DocumentExtractor, ExtractedDocument

logger = logging.getLogger(__name__)


class DocumentAIService:
    """Coordinates Reader extraction, privacy/safety gates, and document AI tasks."""

    VALID_TASKS = {"analyze", "summarize", "explain", "ask", "quality_check", "rewrite_suggestions", "key_points"}

    def __init__(self, *, extractor: Optional[DocumentExtractor] = None, adapter: Optional[SLAIAdapter] = None) -> None:
        self.extractor = extractor or DocumentExtractor()
        self.adapter = adapter or SLAIAdapter()

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "service": "documaster-ai",
            "slai": self.adapter.health(),
            "privacy": {"stored": False, "retention": "none_by_default"},
            "limits": {
                "max_upload_bytes": self.extractor.max_file_size_bytes,
                "max_pages": self.extractor.max_pages,
                "max_text_chars": self.extractor.max_text_chars,
                "allowed_file_types": sorted(self.extractor.allowed_extensions),
            },
        }

    def initialize_runtime(self) -> Dict[str, Any]:
        return self.adapter.initialize_runtime()

    def run_upload(
        self,
        *,
        task: str,
        file: FileStorage,
        question: str = "",
        options: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        safe_name = secure_filename(file.filename or "uploaded_document")
        suffix = Path(safe_name).suffix.lower()
        if suffix not in self.extractor.allowed_extensions:
            return self._error(task=task, message=f"Unsupported file type '{suffix or 'unknown'}'.", started=started)
        try:
            with tempfile.TemporaryDirectory(prefix="documaster_upload_") as tmp_dir:
                temp_path = Path(tmp_dir) / safe_name
                raw = file.read()
                temp_path.write_bytes(raw)
                extracted = self._extract_path_with_reader(temp_path, started=started)
                return self._run_extracted(task=task, extracted=extracted, question=question, options=options, started=started)
        except DocumentExtractionError as exc:
            return self._error(task=task, message=str(exc), started=started)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DocuMaster upload processing failed: %s", type(exc).__name__)
            return self._error(task=task, message=f"Document processing failed: {type(exc).__name__}.", started=started)

    def run_path(
        self,
        *,
        task: str,
        path: str | Path,
        question: str = "",
        options: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            extracted = self._extract_path_with_reader(Path(path), started=started)
            return self._run_extracted(task=task, extracted=extracted, question=question, options=options, started=started)
        except DocumentExtractionError as exc:
            return self._error(task=task, message=str(exc), started=started)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DocuMaster path processing failed: %s", type(exc).__name__)
            return self._error(task=task, message=f"Document processing failed: {type(exc).__name__}.", started=started)

    def _extract_path_with_reader(self, path: Path, *, started: float) -> ExtractedDocument:
        self._validate_path_before_reader(path)
        reader_extraction = self.adapter.read_path(path, recover=True, include_content=True)
        if reader_extraction is not None:
            return self._from_reader_extraction(reader_extraction)
        fallback = self.extractor.extract_path(path)
        fallback.metadata["reader_agent_used"] = False
        fallback.warnings.append("SLAI Reader Agent was unavailable; local extractor was used.")
        return fallback

    def _validate_path_before_reader(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            raise DocumentExtractionError("Selected file does not exist or is not a file.")
        if path.suffix.lower() not in self.extractor.allowed_extensions:
            raise DocumentExtractionError(
                f"Unsupported file type '{path.suffix.lower() or 'unknown'}'. Supported: {', '.join(sorted(self.extractor.allowed_extensions))}."
            )
        if path.stat().st_size > self.extractor.max_file_size_bytes:
            limit_mb = self.extractor.max_file_size_bytes / (1024 * 1024)
            raise DocumentExtractionError(f"File is too large. Maximum upload size is {limit_mb:.0f} MB.")

    @staticmethod
    def _from_reader_extraction(reader: ReaderExtraction) -> ExtractedDocument:
        metadata = dict(reader.metadata)
        metadata.setdefault("reader_agent_used", True)
        metadata.setdefault("word_count", len(reader.text.split()))
        metadata.setdefault("char_count", len(reader.text))
        metadata.setdefault("detected_type", reader.detected_type)
        return ExtractedDocument(
            filename=reader.filename,
            detected_type=reader.detected_type,
            text=reader.text,
            metadata=metadata,
            warnings=list(reader.warnings),
        )

    def _run_extracted(
        self,
        *,
        task: str,
        extracted: ExtractedDocument,
        question: str,
        options: Optional[Mapping[str, Any]],
        started: float,
    ) -> Dict[str, Any]:
        normalized = self._normalize_task(task)
        text = extracted.text
        warnings = list(extracted.warnings)
        metadata = self._metadata(extracted, started)

        privacy = self.adapter.privacy_check(text, metadata)
        if privacy.get("findings"):
            warnings.append("Potential sensitive data was detected. Content was not stored.")

        safety_pre = self.adapter.safety_check(text, task=normalized, metadata=metadata)
        if safety_pre.get("decision") == "block":
            return self._envelope(
                status="error",
                task=normalized,
                filename=extracted.filename,
                metadata=metadata,
                result={
                    "issues": [
                        {
                            "type": "safety_block",
                            "severity": "error",
                            "message": "The document task was blocked by the safety check.",
                        }
                    ],
                    "confidence": 0.0,
                    "safety": safety_pre,
                },
                warnings=warnings + ["Safety check blocked this document task."],
                privacy=privacy,
            )

        result = self.adapter.handle_document_task(normalized, text, question=question, metadata=metadata, options=options)
        safety_post = self.adapter.safety_check(str(result), task=f"{normalized}_output", metadata=metadata)
        if safety_post.get("decision") == "block":
            result = {
                "issues": [
                    {
                        "type": "safety_output_block",
                        "severity": "error",
                        "message": "The generated result was blocked by the safety check.",
                    }
                ],
                "confidence": 0.0,
            }
            warnings.append("Safety check blocked the generated output.")

        result.setdefault("privacy", privacy)
        result.setdefault("safety", {"pre": safety_pre, "post": safety_post})
        metadata.update(
            {
                "processing_time_ms": round((time.perf_counter() - started) * 1000),
                "slai_available": bool(self.adapter.health().get("available")),
                "reader_agent_used": bool(metadata.get("reader_agent_used")),
            }
        )
        status = "success" if result.get("confidence", 0.0) or any(result.get(key) for key in ("summary", "answer", "key_points", "issues", "suggestions")) else "partial"
        payload = self._envelope(
            status=status,
            task=normalized,
            filename=extracted.filename,
            metadata=metadata,
            result=result,
            warnings=warnings,
            privacy=privacy,
        )
        self.adapter.observe(
            {
                "task": normalized,
                "status": payload["status"],
                "filename": extracted.filename,
                "reader_agent_used": metadata.get("reader_agent_used"),
                "processing_time_ms": metadata.get("processing_time_ms"),
                "word_count": metadata.get("word_count"),
            }
        )
        logger.info("DocuMaster AI task completed: task=%s status=%s reader=%s", normalized, payload["status"], metadata.get("reader_agent_used"))
        return payload

    def _metadata(self, extracted: ExtractedDocument, started: float) -> Dict[str, Any]:
        metadata = {
            "word_count": extracted.metadata.get("word_count", len(extracted.text.split())),
            "char_count": extracted.metadata.get("char_count", len(extracted.text)),
            "page_count": extracted.metadata.get("page_count", 0),
            "detected_type": extracted.detected_type,
            "language": extracted.metadata.get("language", "unknown"),
            "processing_time_ms": round((time.perf_counter() - started) * 1000),
            "reader_agent_used": bool(extracted.metadata.get("reader_agent_used", False)),
            "slai_available": bool(self.adapter.health().get("available")),
        }
        for key, value in extracted.metadata.items():
            if key not in metadata and key != "page_texts":
                metadata[key] = value
        return metadata

    @classmethod
    def _normalize_task(cls, task: str) -> str:
        normalized = str(task or "analyze").strip().lower().replace("-", "_")
        return normalized if normalized in cls.VALID_TASKS else "analyze"

    @staticmethod
    def _envelope(
        *,
        status: str,
        task: str,
        filename: str,
        metadata: Mapping[str, Any],
        result: Mapping[str, Any],
        warnings: list[str],
        privacy: Mapping[str, Any],
    ) -> Dict[str, Any]:
        stable_result = {
            "summary": result.get("summary", ""),
            "answer": result.get("answer", ""),
            "key_points": result.get("key_points", []),
            "headings": result.get("headings", []),
            "issues": result.get("issues", []),
            "suggestions": result.get("suggestions", []),
            "merge_plan": result.get("merge_plan", []),
            "confidence": float(result.get("confidence", 0.0) or 0.0),
        }
        stable_result.update(
            {k: v for k, v in result.items() if k not in stable_result and k not in {"raw_text", "content", "document_text"}}
        )
        return {
            "status": status,
            "task": task,
            "filename": filename,
            "metadata": dict(metadata),
            "result": stable_result,
            "warnings": list(dict.fromkeys(str(item) for item in warnings if item)),
            "privacy": {"stored": False, "retention": "none_by_default", **dict(privacy)},
        }

    def _error(self, *, task: str, message: str, started: float) -> Dict[str, Any]:
        return self._envelope(
            status="error",
            task=self._normalize_task(task),
            filename="",
            metadata={
                "word_count": 0,
                "char_count": 0,
                "page_count": 0,
                "detected_type": "",
                "language": "unknown",
                "processing_time_ms": round((time.perf_counter() - started) * 1000),
                "reader_agent_used": False,
                "slai_available": bool(self.adapter.health().get("available")),
            },
            result={"issues": [{"type": "processing_error", "severity": "error", "message": message}], "confidence": 0.0},
            warnings=[message],
            privacy={"stored": False, "retention": "none_by_default"},
        )
