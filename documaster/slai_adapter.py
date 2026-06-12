"""Safe SLAI adapter for DocMaster.

This adapter isolates DocMaster from SLAI internals. Heavy agents are lazily loaded
and never initialized during import. Normal document requests use inference-safe
operations only; training, RSI, MAML, DQN, and adaptive learning loops are not used.
"""

from __future__ import annotations

import importlib
import os
import re

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass
class SLAIRuntimeState:
    enabled: bool = False
    available: bool = False
    status: str = "disabled"
    error: Optional[str] = None
    agents: Dict[str, str] = field(default_factory=dict)


class SLAIAdapter:
    """Stable document-AI facade used by DocMaster services and GUI."""

    SAFE_AGENT_TYPES = ("reader", "language", "reasoning", "knowledge", "quality", "privacy", "safety", "observability")

    def __init__(self, *, enable_slai: Optional[bool] = None) -> None:
        self.enable_slai = bool(os.getenv("DOCMASTER_ENABLE_SLAI", "0") == "1") if enable_slai is None else enable_slai
        self.state = SLAIRuntimeState(enabled=self.enable_slai)
        self.shared_memory = None
        self.factory = None
        self.agents: Dict[str, Any] = {}

    def health(self) -> Dict[str, Any]:
        return {
            "enabled": self.enable_slai,
            "available": self.state.available,
            "status": self.state.status,
            "error": self.state.error,
            "agents": dict(self.state.agents),
            "mode": "slai" if self.state.available else "safe_fallback",
        }

    def initialize_runtime(self) -> Dict[str, Any]:
        """Lazy-load safe SLAI components.

        Returns a health payload. It does not raise to GUI/API callers.
        """
        if not self.enable_slai:
            self.state.status = "disabled"
            self.state.error = "Set DOCMASTER_ENABLE_SLAI=1 to use SLAI runtime."
            return self.health()
        if self.state.available:
            return self.health()

        try:
            root = os.getenv("SLAI_ROOT")
            if root and root not in os.sys.path:
                os.sys.path.insert(0, root)

            shared_memory_mod = importlib.import_module("src.agents.collaborative.shared_memory")
            factory_mod = importlib.import_module("src.agents.agent_factory")
            SharedMemory = getattr(shared_memory_mod, "SharedMemory")
            AgentFactory = getattr(factory_mod, "AgentFactory")
            self.shared_memory = SharedMemory()
            self.factory = AgentFactory()
            self.state.agents["factory"] = "loaded"
            self.state.agents["shared_memory"] = "loaded"

            for agent_type in self.SAFE_AGENT_TYPES:
                try:
                    self.agents[agent_type] = self.factory.create(agent_type, self.shared_memory)
                    self.state.agents[agent_type] = "loaded"
                except Exception as exc:  # noqa: BLE001
                    self.state.agents[agent_type] = f"fallback · {type(exc).__name__}: {exc}"

            self.state.available = True
            self.state.status = "ready"
            self.state.error = None
            return self.health()
        except Exception as exc:  # noqa: BLE001
            self.state.available = False
            self.state.status = "fallback"
            self.state.error = f"{type(exc).__name__}: {exc}"
            return self.health()

    def analyze_document(self, text: str, *, options: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        structure = self.detect_structure(text)
        key_points = self.extract_key_points(text)
        quality = self.quality_check(text)
        return {
            "summary": self.summarize_document(text, mode="compact").get("summary", ""),
            "key_points": key_points,
            "structure": structure,
            "issues": quality.get("issues", []),
            "suggestions": quality.get("suggestions", []),
            "confidence": 0.72,
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
        return {
            "summary": f"In clear language: {summary}" if summary else "No readable explanation could be created.",
            "confidence": 0.64 if summary else 0.0,
        }

    def answer_question(self, text: str, question: str) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {"answer": "Add a question before running document Q&A.", "confidence": 0.0}
        sentences = self._sentences(text)
        q_terms = self._keywords(question)
        ranked = []
        for index, sentence in enumerate(sentences):
            s_terms = set(self._keywords(sentence))
            overlap = len(q_terms.intersection(s_terms))
            if overlap:
                ranked.append((overlap, index, sentence))
        if not ranked:
            return {
                "answer": "I could not find a clear answer in the readable document text.",
                "confidence": 0.25,
                "evidence": [],
            }
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

        score = max(0.25, min(0.95, 0.92 - (0.02 * len(issues))))
        return {"issues": issues, "suggestions": suggestions, "confidence": score}

    def rewrite_suggestions(self, text: str) -> Dict[str, Any]:
        quality = self.quality_check(text)
        suggestions = list(quality.get("suggestions", []))
        suggestions.extend(
            [
                "Keep the original document unchanged and apply suggestions manually after review.",
                "Use active voice where the subject and action are unclear.",
                "Move background details to a separate section if they interrupt the main argument.",
            ]
        )
        return {"suggestions": suggestions[:8], "confidence": quality.get("confidence", 0.6)}

    def privacy_check(self, text: str) -> Dict[str, Any]:
        patterns = {
            "email": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            "phone_like": r"\b(?:\+?\d[\d\s().-]{7,}\d)\b",
            "secret_like": r"(?i)\b(api[_-]?key|token|password|secret)\s*[:=]",
        }
        findings = []
        for name, pattern in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                findings.append({"type": name, "severity": "review", "message": f"Potential {name} content detected."})
        return {
            "stored": False,
            "retention": "none_by_default",
            "findings": findings,
            "recommendation": "Review sensitive content before sharing extracted text with external services." if findings else "No obvious sensitive pattern detected.",
        }

    def build_document_context(self, text: str, metadata: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "metadata": dict(metadata),
            "structure": self.detect_structure(text),
            "key_points": self.extract_key_points(text),
            "privacy": self.privacy_check(text),
        }

    def extract_key_points(self, text: str, *, limit: int = 6) -> List[str]:
        sentences = self._sentences(text)
        ranked = self._rank_sentences(text, sentences)
        return [sentence for _idx, _score, sentence in sorted(ranked[:limit], key=lambda item: item[0])]

    def detect_structure(self, text: str) -> Dict[str, Any]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        headings = []
        for line in lines[:250]:
            if len(line) <= 90 and (re.match(r"^\d+(\.\d+)*\s+\S+", line) or line.isupper() or line.istitle()):
                headings.append(line)
        keywords = [word for word, _count in Counter(self._keywords(text)).most_common(12)]
        return {
            "heading_count": len(headings),
            "headings": headings[:20],
            "section_count_estimate": max(1, len(headings)),
            "topics": keywords,
        }

    @staticmethod
    def _sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
        return [p.strip() for p in parts if len(p.strip()) > 20]

    @staticmethod
    def _keywords(text: str) -> set[str]:
        stop = {
            "the", "and", "for", "with", "that", "this", "from", "into", "are", "was", "were", "het", "een", "voor", "met", "dat", "dit", "zijn", "wordt", "worden", "van", "en", "de",
        }
        words = re.findall(r"[A-Za-zÀ-ÿ0-9]{3,}", text.lower())
        return {word for word in words if word not in stop}

    def _rank_sentences(self, text: str, sentences: Iterable[str]) -> List[tuple[int, float, str]]:
        words = [w for w in re.findall(r"[A-Za-zÀ-ÿ0-9]{3,}", text.lower()) if w not in self._keywords("")]
        frequency = Counter(words)
        ranked = []
        for index, sentence in enumerate(sentences):
            terms = re.findall(r"[A-Za-zÀ-ÿ0-9]{3,}", sentence.lower())
            score = sum(frequency.get(t, 0) for t in terms) / max(1, len(terms))
            if any(marker in sentence.lower() for marker in ("conclusion", "therefore", "important", "doel", "belangrijk", "conclusie")):
                score += 2
            ranked.append((index, score, sentence))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked
