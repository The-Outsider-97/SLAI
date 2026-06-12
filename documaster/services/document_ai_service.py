"""Service layer for DocMaster document AI workflows."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from werkzeug.datastructures import FileStorage

from ..slai_adapter import SLAIAdapter
from .document_extractor import DocumentExtractionError, DocumentExtractor, ExtractedDocument


class DocumentAIService:
    """Coordinates extraction, privacy checks, quality checks, and SLAI/fallback AI."""

    VALID_TASKS = {"analyze", "summarize", "explain", "ask", "quality_check", "rewrite_suggestions", "key_points"}

    def __init__(self, *, extractor: Optional[DocumentExtractor] = None, adapter: Optional[SLAIAdapter] = None) -> None:
        self.extractor = extractor or DocumentExtractor()
        self.adapter = adapter or SLAIAdapter()

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "service": "docmaster-ai",
            "slai": self.adapter.health(),
            "privacy": {"stored": False, "retention": "none_by_default"},
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
        try:
            extracted = self.extractor.extract_upload(file)
            return self._run_extracted(task=task, extracted=extracted, question=question, options=options, started=started)
        except DocumentExtractionError as exc:
            return self._error(task=task, message=str(exc), started=started)
        except Exception as exc:  # noqa: BLE001
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
            extracted = self.extractor.extract_path(path)
            return self._run_extracted(task=task, extracted=extracted, question=question, options=options, started=started)
        except DocumentExtractionError as exc:
            return self._error(task=task, message=str(exc), started=started)
        except Exception as exc:  # noqa: BLE001
            return self._error(task=task, message=f"Document processing failed: {type(exc).__name__}.", started=started)

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
        privacy = self.adapter.privacy_check(text)
        if privacy.get("findings"):
            warnings.append("Potential sensitive data was detected. Content was not stored.")

        result = self._dispatch(normalized, text, question, options)
        metadata = {
            "word_count": extracted.metadata.get("word_count", len(text.split())),
            "char_count": extracted.metadata.get("char_count", len(text)),
            "detected_type": extracted.detected_type,
            "language": extracted.metadata.get("language", "unknown"),
            "processing_time_ms": round((time.perf_counter() - started) * 1000),
            **{k: v for k, v in extracted.metadata.items() if k not in {"word_count", "char_count", "detected_type", "language"}},
        }
        return self._envelope(
            status="success",
            task=normalized,
            filename=extracted.filename,
            metadata=metadata,
            result=result,
            warnings=warnings,
            privacy={"stored": False, "retention": "none_by_default", **privacy},
        )

    def _dispatch(self, task: str, text: str, question: str, options: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if task == "analyze":
            return self.adapter.analyze_document(text, options=options)
        if task == "summarize":
            return self.adapter.summarize_document(text, mode=str((options or {}).get("mode", "compact")))
        if task == "explain":
            return self.adapter.explain_document(text)
        if task == "ask":
            return self.adapter.answer_question(text, question)
        if task == "quality_check":
            return self.adapter.quality_check(text)
        if task == "rewrite_suggestions":
            return self.adapter.rewrite_suggestions(text)
        if task == "key_points":
            return {"key_points": self.adapter.extract_key_points(text), "confidence": 0.72}
        return {"suggestions": ["Unsupported AI task."], "confidence": 0.0}

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
        return {
            "status": status,
            "task": task,
            "filename": filename,
            "metadata": dict(metadata),
            "result": {
                "summary": result.get("summary", ""),
                "answer": result.get("answer", ""),
                "key_points": result.get("key_points", []),
                "issues": result.get("issues", []),
                "suggestions": result.get("suggestions", []),
                "structure": result.get("structure", {}),
                "confidence": float(result.get("confidence", 0.0) or 0.0),
                **{
                    k: v
                    for k, v in result.items()
                    if k not in {"summary", "answer", "key_points", "issues", "suggestions", "structure", "confidence"}
                },
            },
            "warnings": list(dict.fromkeys(warnings)),
            "privacy": dict(privacy),
        }

    def _error(self, *, task: str, message: str, started: float) -> Dict[str, Any]:
        return self._envelope(
            status="error",
            task=self._normalize_task(task),
            filename="",
            metadata={
                "word_count": 0,
                "char_count": 0,
                "detected_type": "",
                "language": "unknown",
                "processing_time_ms": round((time.perf_counter() - started) * 1000),
            },
            result={"issues": [{"type": "processing_error", "severity": "error", "message": message}], "confidence": 0.0},
            warnings=[message],
            privacy={"stored": False, "retention": "none_by_default"},
        )
