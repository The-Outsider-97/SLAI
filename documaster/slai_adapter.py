"""DocuMaster-to-SLAI adapter.

This module is the only place where DocuMaster touches SLAI internals.  The
adapter lazy-loads SLAI agents, keeps document content private by default, and
falls back only inside DocuMaster's own AI flow when SLAI is unavailable.

Normal document tasks intentionally avoid training, reinforcement-learning,
RSI/self-improvement, MAML, DQN, and other heavy learning loops.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import re
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

_SAFE_AGENT_ORDER = ("reader", "privacy", "safety", "quality", "observability", "language")
_OPTIONAL_AGENTS = ("knowledge", "reasoning", "evaluation")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "are", "was", "were", "not", "but",
    "het", "een", "voor", "met", "dat", "dit", "zijn", "wordt", "worden", "van", "en", "de", "niet", "maar",
}


@dataclass(frozen=True)
class DocuMasterAIConfig:
    """Runtime policy for DocuMaster AI integration."""

    enable_slai: bool = True
    enable_persistent_memory: bool = False
    enable_knowledge_agent: bool = False
    enable_reasoning_agent: bool = False
    enable_evaluation_agent: bool = False
    max_upload_size_bytes: int = 20 * 1024 * 1024
    max_pages: int = 250
    max_extracted_text_chars: int = 120_000
    allowed_file_types: tuple[str, ...] = (".pdf", ".docx", ".txt", ".html", ".htm", ".xml", ".odt")
    cleanup_temp_files: bool = True

    @classmethod
    def from_env(cls) -> "DocuMasterAIConfig":
        def flag(name: str, default: str = "0") -> bool:
            return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}

        def integer(name: str, default: int, minimum: int = 1) -> int:
            try:
                return max(minimum, int(os.getenv(name, str(default))))
            except (TypeError, ValueError):
                return default

        allowed = tuple(
            item if item.startswith(".") else f".{item}"
            for item in os.getenv("DOCMASTER_ALLOWED_FILE_TYPES", ".pdf,.docx,.txt,.html,.htm,.xml,.odt").split(",")
            if item.strip()
        )
        return cls(
            enable_slai=flag("DOCMASTER_ENABLE_SLAI", "1"),
            enable_persistent_memory=flag("DOCMASTER_ENABLE_PERSISTENT_MEMORY", "0"),
            enable_knowledge_agent=flag("DOCMASTER_ENABLE_KNOWLEDGE_AGENT", "0"),
            enable_reasoning_agent=flag("DOCMASTER_ENABLE_REASONING_AGENT", "0"),
            enable_evaluation_agent=flag("DOCMASTER_ENABLE_EVALUATION_AGENT", "0"),
            max_upload_size_bytes=integer("DOCMASTER_MAX_UPLOAD_BYTES", 20 * 1024 * 1024),
            max_pages=integer("DOCMASTER_MAX_PAGES", 250),
            max_extracted_text_chars=integer("DOCMASTER_MAX_TEXT_CHARS", 120_000),
            allowed_file_types=allowed or cls.allowed_file_types,
            cleanup_temp_files=flag("DOCMASTER_CLEANUP_TEMP_FILES", "1"),
        )

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "enable_slai": self.enable_slai,
            "enable_persistent_memory": self.enable_persistent_memory,
            "enable_knowledge_agent": self.enable_knowledge_agent,
            "enable_reasoning_agent": self.enable_reasoning_agent,
            "enable_evaluation_agent": self.enable_evaluation_agent,
            "max_upload_mb": round(self.max_upload_size_bytes / (1024 * 1024), 2),
            "max_pages": self.max_pages,
            "max_extracted_text_chars": self.max_extracted_text_chars,
            "allowed_file_types": list(self.allowed_file_types),
            "cleanup_temp_files": self.cleanup_temp_files,
        }


@dataclass
class SLAIRuntimeState:
    enabled: bool = True
    available: bool = False
    status: str = "not_initialized"
    error: Optional[str] = None
    agents: Dict[str, str] = field(default_factory=dict)
    initialized_at: Optional[float] = None


@dataclass
class ReaderExtraction:
    filename: str
    detected_type: str
    text: str
    metadata: Dict[str, Any]
    warnings: List[str]
    raw_reader_result: Dict[str, Any] = field(default_factory=dict)


class SLAIAdapter:
    """Stable adapter boundary used by DocuMaster services and GUI."""

    def __init__(self, *, config: Optional[DocuMasterAIConfig] = None, enable_slai: Optional[bool] = None) -> None:
        self.config = config or DocuMasterAIConfig.from_env()
        if enable_slai is not None:
            self.config = DocuMasterAIConfig(**{**self.config.__dict__, "enable_slai": bool(enable_slai)})
        self.state = SLAIRuntimeState(enabled=self.config.enable_slai)
        self.shared_memory: Any = None
        self.factory: Any = None
        self.agents: Dict[str, Any] = {}
        self._last_observability_event: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Runtime loading and health
    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        reader_status = self.state.agents.get("reader", "not_loaded")
        return {
            "enabled": self.config.enable_slai,
            "available": self.state.available,
            "status": self.state.status,
            "mode": "slai" if self.state.available else "degraded" if self.config.enable_slai else "disabled",
            "error": self.state.error,
            "reader_agent_available": reader_status == "loaded",
            "agents": dict(self.state.agents),
            "initialized_at": self.state.initialized_at,
            "config": self.config.to_public_dict(),
        }

    def initialize_runtime(self) -> Dict[str, Any]:
        """Lazy-load safe SLAI runtime components.

        No heavy learning or training components are imported here. Optional
        Knowledge/Reasoning/Evaluation agents only load behind explicit flags.
        """
        if not self.config.enable_slai:
            self.state.status = "disabled"
            self.state.error = "SLAI integration disabled by DocuMaster configuration."
            return self.health()
        if self.state.available:
            return self.health()

        try:
            root = os.getenv("SLAI_ROOT")
            if root and root not in os.sys.path:
                os.sys.path.insert(0, root)

            memory_mod = importlib.import_module("src.agents.collaborative.shared_memory")
            factory_mod = importlib.import_module("src.agents.agent_factory")
            SharedMemory = getattr(memory_mod, "SharedMemory")
            AgentFactory = getattr(factory_mod, "AgentFactory")

            self.shared_memory = SharedMemory()
            self.factory = AgentFactory(
                config={
                    "agent_factory": {
                        "create_dependencies": False,
                        "enable_out_of_process_fallback": False,
                        "block_torch_required_when_unavailable": True,
                        "diagnostics_import_check": False,
                        "diagnostics_constructor_check": False,
                    }
                }
            )
            self.state.agents["shared_memory"] = "loaded"
            self.state.agents["factory"] = "loaded"

            for agent_name in _SAFE_AGENT_ORDER:
                self._load_agent(agent_name)

            if self.config.enable_knowledge_agent:
                self._load_agent("knowledge")
            else:
                self.state.agents["knowledge"] = "not_loaded_by_policy"
            if self.config.enable_reasoning_agent:
                self._load_agent("reasoning")
            else:
                self.state.agents["reasoning"] = "not_loaded_by_policy"
            if self.config.enable_evaluation_agent:
                self._load_agent("evaluation")
            else:
                self.state.agents["evaluation"] = "not_loaded_by_policy_heavy"

            self.state.available = self.state.agents.get("reader") == "loaded"
            self.state.status = "ready" if self.state.available else "degraded_no_reader"
            self.state.error = None if self.state.available else "SLAI loaded, but Reader Agent is unavailable."
            self.state.initialized_at = time.time()
        except Exception as exc:  # noqa: BLE001
            self.state.available = False
            self.state.status = "unavailable"
            self.state.error = f"{type(exc).__name__}: {exc}"
            logger.warning("DocuMaster SLAI runtime unavailable: %s", type(exc).__name__)
        return self.health()

    def _load_agent(self, agent_name: str) -> None:
        if self.factory is None:
            self.state.agents[agent_name] = "factory_unavailable"
            return
        try:
            config = self._agent_config(agent_name)
            self.agents[agent_name] = self.factory.create(agent_name, self.shared_memory, config=config)
            self.state.agents[agent_name] = "loaded"
        except Exception as exc:  # noqa: BLE001
            self.state.agents[agent_name] = f"unavailable:{type(exc).__name__}"
            logger.info("SLAI agent unavailable for DocuMaster: %s (%s)", agent_name, type(exc).__name__)

    def _agent_config(self, agent_name: str) -> Dict[str, Any]:
        if agent_name == "reader":
            return {
                "output_dir": str(Path(tempfile.gettempdir()) / "documaster_reader"),
                "max_concurrency": 2,
                "fail_fast": False,
                "auto_recover_low_quality": True,
                "include_documents_by_default": False,
                "include_content_by_default": False,
                "use_shared_cache": self.config.enable_persistent_memory,
                "cache_parse_results": self.config.enable_persistent_memory,
                "cache_recovery_results": self.config.enable_persistent_memory,
                "write_checkpoints": self.config.enable_persistent_memory,
                "redact_checkpoints": True,
                "publish_events": True,
                "include_traceback_errors": False,
            }
        if agent_name == "quality":
            return {
                "include_record_previews_in_shared_memory": False,
                "shared_memory": {"enabled": False, "publish_notifications": False},
                "fail_closed_on_subsystem_error": False,
            }
        if agent_name == "privacy":
            return {"publish_to_shared_memory": False, "publish_notifications": False, "shared_ttl_seconds": 0}
        if agent_name == "safety":
            return {
                "collect_feedback": False,
                "enable_learnable_aggregation": False,
                "shared_memory": {"store_assessments": False, "store_audit_events": False},
                "compliance": {"evaluate_on_task": False},
            }
        if agent_name == "observability":
            return {"enable_shared_context_export": False, "auto_route_handler_on_warning": False}
        if agent_name == "knowledge":
            return {"retrieval_mode": "tfidf", "bias_detection_enabled": False}
        return {}

    # ------------------------------------------------------------------
    # Reader Agent integration
    # ------------------------------------------------------------------
    def read_path(self, path: str | Path, *, recover: bool = True, include_content: bool = True) -> Optional[ReaderExtraction]:
        """Read a file with SLAI Reader Agent and return normalized text.

        Returns None when SLAI/Reader is unavailable so DocumentAIService can use
        its local extractor without pretending SLAI handled the file.
        """
        health = self.initialize_runtime()
        if not health.get("reader_agent_available"):
            return None
        reader = self.agents.get("reader")
        if reader is None or not hasattr(reader, "perform_task"):
            return None

        file_path = Path(path)
        started = time.perf_counter()
        task = {
            "operation": "read",
            "mode": "read",
            "files": [str(file_path)],
            "recover": bool(recover),
            "include_documents": True,
            "include_content": bool(include_content),
            "use_cache": self.config.enable_persistent_memory,
            "write_checkpoints": self.config.enable_persistent_memory,
            "metadata": {"source": "documaster", "filename": file_path.name},
        }
        try:
            raw = reader.perform_task(task)
        except Exception as exc:  # noqa: BLE001
            self.state.agents["reader"] = f"runtime_error:{type(exc).__name__}"
            logger.warning("Reader Agent failed for %s: %s", self._fingerprint_path(file_path), type(exc).__name__)
            return None

        extraction = self._normalize_reader_result(raw, file_path=file_path)
        if extraction is not None:
            extraction.metadata.setdefault("reader_processing_time_ms", round((time.perf_counter() - started) * 1000))
            extraction.metadata["reader_agent_used"] = True
        return extraction

    def _normalize_reader_result(self, raw: Any, *, file_path: Path) -> Optional[ReaderExtraction]:
        if not isinstance(raw, Mapping):
            return None
        status = str(raw.get("status", "")).lower()
        if status not in {"ok", "partial", "success"}:
            return None
        payload = raw.get("payload", {}) if isinstance(raw.get("payload", {}), Mapping) else {}
        documents = payload.get("documents") if isinstance(payload.get("documents"), list) else []
        if not documents:
            # Some Reader versions only return parsed summaries unless include_documents is honored.
            parsed = payload.get("parsed", {}) if isinstance(payload.get("parsed", {}), Mapping) else {}
            documents = parsed.get("documents") if isinstance(parsed.get("documents"), list) else []
        chunks: List[str] = []
        warnings: List[str] = []
        page_count = 0
        quality_scores: List[float] = []
        for document in documents:
            if not isinstance(document, Mapping):
                continue
            content = str(document.get("content") or document.get("text") or document.get("content_preview") or "")
            if content.strip():
                chunks.append(content.strip())
            metadata = document.get("metadata", {}) if isinstance(document.get("metadata", {}), Mapping) else {}
            if metadata.get("page_count"):
                with _suppress_value_error():
                    page_count += int(metadata.get("page_count") or 0)
            if metadata.get("quality_score") is not None:
                with _suppress_value_error():
                    quality_scores.append(float(metadata.get("quality_score")))
            if document.get("warnings"):
                warnings.extend([str(item) for item in _as_list(document.get("warnings"))])
        text = self._clean_text("\n\n".join(chunks))
        if not text:
            return None
        if len(text) > self.config.max_extracted_text_chars:
            text = text[: self.config.max_extracted_text_chars]
            warnings.append(f"Reader Agent text was truncated to {self.config.max_extracted_text_chars:,} characters.")
        metadata = dict(raw.get("metadata", {}) if isinstance(raw.get("metadata", {}), Mapping) else {})
        metadata.update(
            {
                "word_count": len(text.split()),
                "char_count": len(text),
                "detected_type": file_path.suffix.lower().lstrip("."),
                "page_count": page_count or metadata.get("page_count", 0),
                "language": self._guess_language(text),
                "reader_status": status,
                "reader_quality_score": round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else None,
            }
        )
        return ReaderExtraction(
            filename=file_path.name,
            detected_type=file_path.suffix.lower().lstrip("."),
            text=text,
            metadata=metadata,
            warnings=list(dict.fromkeys(warnings + [str(w) for w in _as_list(raw.get("warnings"))])),
            raw_reader_result=self._strip_private_reader_payload(dict(raw)),
        )

    @staticmethod
    def _strip_private_reader_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
        payload = raw.get("payload")
        if isinstance(payload, Mapping):
            safe_payload = dict(payload)
            docs = safe_payload.get("documents")
            if isinstance(docs, list):
                safe_payload["documents"] = [
                    {key: value for key, value in dict(doc).items() if key not in {"content", "text"}}
                    for doc in docs
                    if isinstance(doc, Mapping)
                ]
            raw["payload"] = safe_payload
        return raw

    # ------------------------------------------------------------------
    # Purposeful SLAI agent gates
    # ------------------------------------------------------------------
    def privacy_check(self, text: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        agent = self._get_agent_if_ready("privacy")
        if agent is not None and hasattr(agent, "evaluate_privacy"):
            try:
                payload = {
                    "content": text,
                    "metadata": dict(metadata or {}),
                    "purpose": "documaster_document_processing",
                    "retention": "none_by_default",
                }
                decision = agent.evaluate_privacy(
                    payload,
                    purpose="documaster_document_processing",
                    action="process",
                    source_context="documaster_upload",
                    destination_context="documaster_runtime",
                    retention_days=1,
                    context={"store_document_content": False},
                )
                return self._normalize_privacy_result(decision)
            except Exception as exc:  # noqa: BLE001
                logger.info("Privacy Agent unavailable during document check: %s", type(exc).__name__)
        return self._local_privacy_check(text)

    def safety_check(self, text: str, *, task: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        agent = self._get_agent_if_ready("safety")
        if agent is not None and hasattr(agent, "perform_task"):
            try:
                result = agent.perform_task(
                    {"text": text[:12_000], "task": task, "metadata": dict(metadata or {})},
                    context={"type": "documaster_document_processing", "run_compliance": False},
                )
                decision = str(result.get("decision", "allow")).lower() if isinstance(result, Mapping) else "allow"
                return {
                    "decision": decision,
                    "is_safe": decision != "block",
                    "risk_score": float(result.get("risk_score", 0.0) or 0.0) if isinstance(result, Mapping) else 0.0,
                    "warnings": [str(item) for item in _as_list(result.get("warnings"))] if isinstance(result, Mapping) else [],
                    "blockers": [str(item) for item in _as_list(result.get("blockers"))] if isinstance(result, Mapping) else [],
                    "agent_used": True,
                }
            except Exception as exc:  # noqa: BLE001
                logger.info("Safety Agent unavailable during document check: %s", type(exc).__name__)
        return {"decision": "allow", "is_safe": True, "risk_score": 0.0, "warnings": [], "blockers": [], "agent_used": False}

    def quality_gate(self, text: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        agent = self._get_agent_if_ready("quality")
        if agent is not None and hasattr(agent, "evaluate_batch"):
            try:
                decision = agent.evaluate_batch(
                    records=[{"text": text, "metadata": dict(metadata or {})}],
                    dataset_id="documaster_document",
                    source_id=str((metadata or {}).get("filename", "document")),
                    context={"task": "document_quality", "store_content": False},
                )
                return {
                    "verdict": str(decision.get("verdict", "warn")),
                    "batch_score": float(decision.get("batch_score", 0.0) or 0.0),
                    "flags": [str(item) for item in _as_list(decision.get("flags"))],
                    "remediation_actions": [str(item) for item in _as_list(decision.get("remediation_actions"))],
                    "agent_used": True,
                }
            except Exception as exc:  # noqa: BLE001
                logger.info("Quality Agent unavailable during document check: %s", type(exc).__name__)
        local = self.quality_check(text)
        return {
            "verdict": "pass" if not local.get("issues") else "warn",
            "batch_score": local.get("confidence", 0.0),
            "flags": [item.get("type", "quality") for item in local.get("issues", []) if isinstance(item, Mapping)],
            "remediation_actions": local.get("suggestions", []),
            "agent_used": False,
        }

    def observe(self, event: Mapping[str, Any]) -> None:
        self._last_observability_event = dict(event)
        agent = self._get_agent_if_ready("observability")
        if agent is None or not hasattr(agent, "perform_task"):
            return
        try:
            safe_event = {k: v for k, v in dict(event).items() if k not in {"text", "content", "raw"}}
            agent.perform_task(
                {
                    "task_name": "documaster_document_task",
                    "agent_name": "DocuMaster",
                    "operation_name": str(safe_event.get("task", "document_task")),
                    "events": [safe_event],
                    "metadata": safe_event,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Observability Agent event skipped: %s", type(exc).__name__)

    def _get_agent_if_ready(self, agent_name: str) -> Any:
        if not self.state.available:
            self.initialize_runtime()
        return self.agents.get(agent_name)

    # ------------------------------------------------------------------
    # Document task logic. Heuristics are local fallback/utility, not fake SLAI.
    # ------------------------------------------------------------------
    def handle_document_task(
        self,
        task: str,
        text: str,
        *,
        question: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        task = str(task or "analyze").strip().lower().replace("-", "_")
        if task == "summarize":
            return self.summarize_document(text, mode=str((options or {}).get("mode", "compact")))
        if task == "explain":
            return self.explain_document(text)
        if task == "ask":
            return self.answer_question(text, question)
        if task == "quality_check":
            quality = self.quality_check(text)
            quality["slai_quality_gate"] = self.quality_gate(text, metadata)
            return quality
        if task == "rewrite_suggestions":
            return self.rewrite_suggestions(text)
        if task == "key_points":
            return {"key_points": self.extract_key_points(text), "confidence": 0.72}
        return self.analyze_document(text, metadata=metadata, options=options)

    def analyze_document(
        self,
        text: str,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        structure = self.detect_structure(text)
        quality = self.quality_check(text)
        return {
            "summary": self.summarize_document(text, mode="compact").get("summary", ""),
            "key_points": self.extract_key_points(text),
            "headings": structure.get("headings", []),
            "structure": structure,
            "issues": quality.get("issues", []),
            "suggestions": quality.get("suggestions", []),
            "slai_quality_gate": self.quality_gate(text, metadata),
            "confidence": self._combine_confidence(0.72, quality.get("confidence", 0.6)),
        }

    def summarize_document(self, text: str, mode: str = "compact") -> Dict[str, Any]:
        sentences = self._sentences(text)
        if not sentences:
            return {"summary": "", "confidence": 0.0}
        target = 3 if mode == "compact" else 6
        ranked = self._rank_sentences(text, sentences)
        selected = sorted(ranked[:target], key=lambda item: item[0])
        summary = " ".join(sentence for _idx, _score, sentence in selected)
        return {"summary": summary, "confidence": 0.68 if summary else 0.0}

    def explain_document(self, text: str) -> Dict[str, Any]:
        summary = self.summarize_document(text, mode="compact").get("summary", "")
        structure = self.detect_structure(text)
        if not summary:
            return {"summary": "No readable explanation could be created from the extracted text.", "confidence": 0.0}
        topics = ", ".join(structure.get("topics", [])[:6])
        return {
            "summary": f"Plain-language explanation: {summary}" + (f"\n\nMain topics detected: {topics}." if topics else ""),
            "confidence": 0.64,
        }

    def answer_question(self, text: str, question: str) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {"answer": "Add a question before running document Q&A.", "confidence": 0.0, "evidence": []}
        sentences = self._sentences(text)
        q_terms = set(self._keywords(question))
        ranked: List[tuple[int, int, str]] = []
        for index, sentence in enumerate(sentences):
            overlap = len(q_terms.intersection(set(self._keywords(sentence))))
            if overlap:
                ranked.append((overlap, index, sentence))
        if not ranked:
            return {"answer": "I could not find a clear answer in the readable document text.", "confidence": 0.25, "evidence": []}
        ranked.sort(key=lambda item: (-item[0], item[1]))
        evidence = [item[2] for item in ranked[:3]]
        return {"answer": " ".join(evidence), "confidence": min(0.85, 0.45 + 0.12 * ranked[0][0]), "evidence": evidence}

    def quality_check(self, text: str) -> Dict[str, Any]:
        words = text.split()
        sentences = self._sentences(text)
        issues: List[Dict[str, Any]] = []
        suggestions: List[str] = []
        avg_sentence_len = len(words) / max(1, len(sentences))
        unique_ratio = len({w.lower().strip(".,;:!?()[]{}") for w in words}) / max(1, len(words))

        if avg_sentence_len > 28:
            issues.append({"type": "readability", "severity": "warning", "message": "Average sentence length is high."})
            suggestions.append("Split long sentences into shorter, clearer statements.")
        if unique_ratio < 0.35 and len(words) > 120:
            issues.append({"type": "repetition", "severity": "warning", "message": "The document may contain repeated wording."})
            suggestions.append("Check repeated terms and merge duplicate statements where possible.")
        if len(self.detect_structure(text).get("headings", [])) < 2 and len(words) > 300:
            issues.append({"type": "structure", "severity": "info", "message": "Few headings were detected."})
            suggestions.append("Add clearer headings or section breaks to improve scanability.")
        if not issues:
            suggestions.append("No major readability or structure issues were detected in the extracted text.")
        return {"issues": issues, "suggestions": suggestions, "confidence": max(0.25, min(0.95, 0.92 - (0.02 * len(issues))))}

    def rewrite_suggestions(self, text: str) -> Dict[str, Any]:
        quality = self.quality_check(text)
        suggestions = list(quality.get("suggestions", []))
        suggestions.extend(
            [
                "Keep the original document unchanged and apply suggestions manually after review.",
                "Use active voice where the subject and action are unclear.",
                "Move background details to a separate section if they interrupt the main argument.",
                "Check that each section has one clear function and does not repeat another section.",
            ]
        )
        return {"suggestions": suggestions[:10], "confidence": quality.get("confidence", 0.6)}

    def extract_key_points(self, text: str, *, limit: int = 6) -> List[str]:
        ranked = self._rank_sentences(text, self._sentences(text))
        return [sentence for _idx, _score, sentence in sorted(ranked[:limit], key=lambda item: item[0])]

    def detect_structure(self, text: str) -> Dict[str, Any]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        headings: List[str] = []
        for line in lines[:300]:
            if len(line) <= 100 and (
                re.match(r"^\d+(\.\d+)*\s+\S+", line)
                or re.match(r"^[A-ZÀ-Ÿ0-9][A-ZÀ-Ÿ0-9\s\-:]{4,}$", line)
                or (line.istitle() and len(line.split()) <= 10)
            ):
                headings.append(line)
        return {
            "heading_count": len(headings),
            "headings": headings[:30],
            "section_count_estimate": max(1, len(headings)),
            "topics": [word for word, _count in Counter(self._keywords(text)).most_common(12)],
        }

    def _normalize_privacy_result(self, decision: Any) -> Dict[str, Any]:
        if not isinstance(decision, Mapping):
            return self._local_privacy_check("")
        stages = decision.get("stages", {}) if isinstance(decision.get("stages", {}), Mapping) else {}
        findings: List[Dict[str, Any]] = []
        for stage_name, stage in stages.items():
            if isinstance(stage, Mapping):
                for key in ("findings", "entities", "violations"):
                    for item in _as_list(stage.get(key)):
                        findings.append({"stage": stage_name, "type": key, "message": str(item)[:240], "severity": "review"})
        return {
            "stored": False,
            "retention": "none_by_default",
            "decision": str(decision.get("decision", "allow")),
            "summary": str(decision.get("summary", "Privacy check completed.")),
            "findings": findings,
            "agent_used": True,
        }

    @staticmethod
    def _local_privacy_check(text: str) -> Dict[str, Any]:
        patterns = {
            "email": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            "phone_like": r"\b(?:\+?\d[\d\s().-]{7,}\d)\b",
            "secret_like": r"(?i)\b(api[_-]?key|token|password|secret|credential)\s*[:=]",
        }
        findings = []
        for name, pattern in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                findings.append({"type": name, "severity": "review", "message": f"Potential {name} content detected."})
        return {
            "stored": False,
            "retention": "none_by_default",
            "decision": "review" if findings else "allow",
            "findings": findings,
            "recommendation": "Review sensitive content before sharing extracted text externally." if findings else "No obvious sensitive pattern detected.",
            "agent_used": False,
        }

    @staticmethod
    def _sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
        return [p.strip() for p in parts if len(p.strip()) > 20]

    @staticmethod
    def _keywords(text: str) -> List[str]:
        words = re.findall(r"[A-Za-zÀ-ÿ0-9]{3,}", text.lower())
        return [word for word in words if word not in _STOPWORDS]

    def _rank_sentences(self, text: str, sentences: Iterable[str]) -> List[tuple[int, float, str]]:
        frequency = Counter(self._keywords(text))
        ranked = []
        for index, sentence in enumerate(sentences):
            terms = self._keywords(sentence)
            score = sum(frequency.get(t, 0) for t in terms) / max(1, len(terms))
            if any(marker in sentence.lower() for marker in ("conclusion", "therefore", "important", "doel", "belangrijk", "conclusie")):
                score += 2
            ranked.append((index, score, sentence))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _guess_language(text: str) -> str:
        sample = f" {text[:3000].lower()} "
        nl = sum(marker in sample for marker in (" de ", " het ", " een ", " voor ", " zijn ", " worden ", " niet "))
        en = sum(marker in sample for marker in (" the ", " and ", " for ", " with ", " this ", " that ", " not "))
        if nl > en:
            return "nl"
        if en > nl:
            return "en"
        return "unknown"

    @staticmethod
    def _combine_confidence(*scores: Any) -> float:
        values = [float(score) for score in scores if isinstance(score, (int, float))]
        return round(sum(values) / len(values), 4) if values else 0.0

    @staticmethod
    def _fingerprint_path(path: Path) -> str:
        return hashlib.sha256(str(path).encode("utf-8", errors="ignore")).hexdigest()[:12]


class _suppress_value_error:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return exc_type in {ValueError, TypeError}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]
