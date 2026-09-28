from __future__ import annotations

"""SLAI-LM: LANTRA-first interactive chatbot for SLAI v2.3.

This root-level entry point composes SLAI's existing LANTRA inference runtime,
SharedMemory, LanguageMemory, and selectively invoked agents.  LANTRA remains
the only normal natural-language generator.  Supporting agents contribute
retrieval, reasoning, browsing/reading context, safety, and alignment evidence;
they do not replace LANTRA as the response model.

Human feedback is persisted through LanguageMemory.  Explicitly validated
feedback can affect later turns immediately through retrieval-based context and
can be exported/queued in the current train_lantra dialogue schema.  This file
does not perform online gradient updates or retrain LANTRA after a message.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from logs.logger import LoggingSettings, configure_logging, get_logger, shutdown_logging
from src.agents.agent_factory import AgentFactory
from src.agents.collaborative.shared_memory import SharedMemory
from src.agents.language.lantra_runtime import LantraRuntime, LantraTextResult
from src.agents.language.language_memory import (
    LanguageMemory,
    MemoryKind,
    MemoryQuery,
    MemoryRole,
    MemoryScope,
)
from src.agents.language.utils.config_loader import get_config_section as get_language_config_section
from src.agents.language.utils.language_helpers import compact_text, json_safe, stable_hash


__version__ = "1.0.0"

LOGGER = get_logger("SLAI-LM")

CONVERSATION_SOURCE = "slailm.conversation"
FEEDBACK_SOURCE = "slailm.feedback"
SESSION_STATS_PREFIX = "slailm:session"
DEFAULT_CONTEXT_TURNS = 10
DEFAULT_KNOWLEDGE_RESULTS = 3
DEFAULT_FEEDBACK_RECALL = 12
DEFAULT_MAX_REGENERATIONS = 1
DEFAULT_MAX_SUPPORT_CHARS = 1800
DEFAULT_MAX_SNIPPET_CHARS = 650


# This policy is intentionally explicit.  It documents the current v2.3 agent
# decision rather than creating ceremonial calls merely to increase agent count.
AGENT_POLICY: Dict[str, Dict[str, str]] = {
    "adaptive": {
        "integration": "excluded",
        "role": "RL/meta-learning environment; not a LANTRA text-learning interface.",
        "invocation": "—",
    },
    "alignment": {
        "integration": "conditional",
        "role": "Advisory alignment assessment for normatively sensitive prompts and validated feedback.",
        "invocation": "conditional",
    },
    "browser": {
        "integration": "conditional",
        "role": "Explicit live browser/search evidence via /browse.",
        "invocation": "explicit",
    },
    "collaborative": {
        "integration": "infrastructure",
        "role": "SharedMemory collaboration fabric is reused; facade delegation is not needed by the CLI.",
        "invocation": "infrastructure",
    },
    "evaluation": {
        "integration": "excluded",
        "role": "Current evaluator needs benchmark/ground-truth style inputs; no arbitrary ungolded chat score is fabricated.",
        "invocation": "—",
    },
    "execution": {
        "integration": "excluded",
        "role": "Side-effect/robot/task execution is outside a text chatbot response cycle.",
        "invocation": "—",
    },
    "handler": {
        "integration": "excluded",
        "role": "Cross-agent failure recovery is not used to conceal optional agent/API defects in this entry point.",
        "invocation": "—",
    },
    "knowledge": {
        "integration": "conditional",
        "role": "Factual retrieval and grounding context.",
        "invocation": "conditional",
    },
    "language": {
        "integration": "subsystem-direct",
        "role": "LantraRuntime and LanguageMemory are used directly to avoid a second LANTRA load/template NLG takeover.",
        "invocation": "every turn",
    },
    "learning": {
        "integration": "excluded",
        "role": "DQN/MAML/RSI/RL learning is not the supervised LANTRA feedback pathway.",
        "invocation": "—",
    },
    "network": {
        "integration": "excluded",
        "role": "Transport relay is unnecessary; BrowserAgent owns explicit browser workflows.",
        "invocation": "—",
    },
    "observability": {
        "integration": "excluded",
        "role": "SLAI logger, LANTRA stats, SharedMemory, and explicit session metrics cover this local CLI without auto-routing incidents.",
        "invocation": "—",
    },
    "perception": {
        "integration": "excluded",
        "role": "Multimodal perception/training is unnecessary for this text-only root CLI.",
        "invocation": "—",
    },
    "planning": {
        "integration": "excluded",
        "role": "Planner expects structured Task objects; free-form text is not converted into invented planner contracts.",
        "invocation": "—",
    },
    "privacy": {
        "integration": "excluded",
        "role": "PrivacyAgent expects explicit data-flow/retention semantics; SafetyAgent sanitization is used before persisted training reuse.",
        "invocation": "—",
    },
    "qnn": {
        "integration": "excluded",
        "role": "Variational state-vector simulation has no legitimate generic chat contribution.",
        "invocation": "—",
    },
    "quality": {
        "integration": "excluded",
        "role": "QualityAgent is a data-quality gate, not an ungolded generative-response judge; train_lantra remains schema authority.",
        "invocation": "—",
    },
    "reader": {
        "integration": "conditional",
        "role": "Explicit local document parsing/context via /read.",
        "invocation": "explicit",
    },
    "reasoning": {
        "integration": "conditional",
        "role": "Public reason(..., reasoning_type='auto') evidence for complex prompts.",
        "invocation": "conditional",
    },
    "safety": {
        "integration": "integrated",
        "role": "Input/output safety assessment and feedback-reuse validation.",
        "invocation": "every turn when available",
    },
    "simulation": {
        "integration": "excluded",
        "role": "Requires explicit simulation models/requests; no text-to-model formalization is invented.",
        "invocation": "—",
    },
    "verification": {
        "integration": "excluded",
        "role": "Requires explicit formal artifacts/properties; free-form responses are not fake-formalized.",
        "invocation": "—",
    },
}


@dataclass(frozen=True)
class SLAILMSettings:
    """Entry-point settings layered over existing SLAI/LANTRA configuration."""

    checkpoint: Optional[str] = None
    device: Optional[str] = None
    max_output_tokens: Optional[int] = None
    context_turns: int = DEFAULT_CONTEXT_TURNS
    knowledge_results: int = DEFAULT_KNOWLEDGE_RESULTS
    max_regenerations: int = DEFAULT_MAX_REGENERATIONS
    max_support_chars: int = DEFAULT_MAX_SUPPORT_CHARS
    max_snippet_chars: int = DEFAULT_MAX_SNIPPET_CHARS
    feedback_prompt: bool = True
    auto_export_feedback: bool = True
    strict_agents: bool = False
    debug: bool = False
    session_id: Optional[str] = None


@dataclass
class SessionStats:
    """Observable, non-invented runtime counters for one SLAI-LM session."""

    started_at: str
    turns: int = 0
    lantra_generation_calls: int = 0
    regenerations: int = 0
    total_turn_latency_ms: float = 0.0
    feedback_total: int = 0
    feedback_positive: int = 0
    feedback_negative: int = 0
    feedback_corrections: int = 0
    feedback_validations: int = 0
    exported_training_examples: int = 0
    optional_agent_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        denominator = (
            self.feedback_positive
            + self.feedback_negative
            + self.feedback_corrections
            + self.feedback_validations
        )
        payload["human_approval_rate"] = (
            (self.feedback_positive + self.feedback_validations) / denominator
            if denominator
            else None
        )
        payload["mean_turn_latency_ms"] = (
            self.total_turn_latency_ms / self.turns if self.turns else None
        )
        return payload


@dataclass
class SupportBundle:
    """Bounded support evidence supplied to LANTRA for one turn."""

    validated_feedback: List[str] = field(default_factory=list)
    knowledge: List[str] = field(default_factory=list)
    reasoning: Dict[str, Any] = field(default_factory=dict)
    browser: str = ""
    reader: str = ""
    safety: Dict[str, Any] = field(default_factory=dict)
    failures: Dict[str, str] = field(default_factory=dict)
    participating_agents: List[str] = field(default_factory=list)

    def mark_agent(self, name: str) -> None:
        if name not in self.participating_agents:
            self.participating_agents.append(name)


@dataclass
class TurnResult:
    """Structured result retained for feedback, diagnostics, and provenance."""

    turn_id: str
    session_id: str
    user_input: str
    response: str
    original_lantra_response: str
    training_history: List[Dict[str, str]]
    participating_agents: List[str]
    evaluation_results: Dict[str, Any]
    generation_metadata: Dict[str, Any]
    context_summary: Dict[str, Any]
    latency_ms: float
    regeneration_count: int
    failsafe_used: bool = False


@dataclass
class FeedbackRecord:
    """Structured human-supervision record stored in LanguageMemory."""

    schema: str
    feedback_id: str
    session_id: str
    timestamp: str
    turn_id: str
    feedback_type: str
    quality_signal: str
    original_user_input: str
    original_lantra_response: str
    validated_response: Optional[str]
    conversation_history: List[Dict[str, str]]
    participating_agents: List[str]
    evaluation_results: Dict[str, Any]
    model_checkpoint: str
    generation_metadata: Dict[str, Any]
    human_validated: bool
    safety_validated: bool
    runtime_adaptation_eligible: bool
    training_eligible: bool
    validation_notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(frozen=True)
class SafetyAssessment:
    """Small projection of SafetyAgent output used by the orchestrator."""

    available: bool
    decision: str
    blocked: bool
    review: bool
    sanitized_text: str
    risk_score: Optional[float]
    blockers: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    degraded_components: Tuple[str, ...] = ()
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(frozen=True)
class AlignmentAssessment:
    """Small projection of AlignmentAgent output used by the orchestrator."""

    available: bool
    approved: Optional[bool]
    requires_review: bool
    status: Optional[str]
    correction_action: Optional[str]
    reasons: Tuple[str, ...] = ()
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        converted = value.to_dict()
        return dict(converted) if isinstance(converted, Mapping) else {"value": converted}
    return {"value": json_safe(value)}


def _compact(value: Any, limit: int) -> str:
    """Create a bounded JSON-safe text projection for prompt context."""

    if isinstance(value, str):
        return compact_text(value, max_length=limit)
    try:
        text = json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        text = str(value)
    return compact_text(text, max_length=limit)


def _collect_text_fields(value: Any, *, max_chars: int) -> str:
    """Collect document/search text without copying subsystem parsing logic."""

    collected: List[str] = []
    remaining = max_chars

    def visit(node: Any, depth: int = 0) -> None:
        nonlocal remaining
        if remaining <= 0 or depth > 6:
            return
        if isinstance(node, str):
            text = node.strip()
            if text:
                piece = text[:remaining]
                collected.append(piece)
                remaining -= len(piece)
            return
        if isinstance(node, Mapping):
            preferred = ("content", "text", "snippet", "summary", "title", "message", "url", "link")
            handled = set()
            for key in preferred:
                if key in node:
                    handled.add(key)
                    visit(node[key], depth + 1)
            for key, item in node.items():
                if key not in handled and key in {"data", "documents", "results", "items", "payload"}:
                    visit(item, depth + 1)
            return
        if isinstance(node, Sequence) and not isinstance(node, (bytes, bytearray, str)):
            for item in node:
                visit(item, depth + 1)
                if remaining <= 0:
                    break

    visit(value)
    return compact_text("\n".join(collected), max_length=max_chars)


def _configured_checkpoint() -> Optional[Path]:
    config = get_language_config_section("lantra_runtime") or {}
    raw = str(config.get("checkpoint_path") or "").strip()
    return Path(raw).expanduser() if raw else None


def _discover_checkpoint(directory: Path) -> Optional[Path]:
    """Reuse the trainer's current checkpoint-resume selection semantics."""

    from train_lantra import discover_resume_checkpoint

    return discover_resume_checkpoint(directory, {})


def list_checkpoint_candidates(explicit: Optional[str] = None) -> List[Path]:
    """List local LANTRA checkpoint files without hardcoding a dated filename."""

    target: Optional[Path]
    if explicit:
        target = Path(explicit).expanduser()
    else:
        target = _configured_checkpoint()

    if target is None:
        return []
    directory = target if target.is_dir() else target.parent
    if not directory.exists():
        return []
    candidates = [
        path
        for pattern in ("lantra*.pt", "lantra*.pth", "lantra*.ckpt")
        for path in directory.glob(pattern)
        if path.is_file()
    ]
    return sorted(set(candidates), key=lambda item: item.stat().st_mtime_ns, reverse=True)


def resolve_checkpoint(explicit: Optional[str]) -> Path:
    """Resolve CLI/configured LANTRA checkpoint, falling back to trainer discovery."""

    if explicit:
        requested = Path(explicit).expanduser()
        if requested.is_file():
            return requested
        if requested.is_dir():
            discovered = _discover_checkpoint(requested)
            if discovered is not None:
                return discovered
            raise FileNotFoundError(f"No LANTRA checkpoint found in directory: {requested}")
        raise FileNotFoundError(f"LANTRA checkpoint does not exist: {requested}")

    configured = _configured_checkpoint()
    if configured is None:
        raise FileNotFoundError(
            "language_config.yaml does not define lantra_runtime.checkpoint_path; use --checkpoint PATH."
        )
    if configured.is_file():
        return configured

    discovered = _discover_checkpoint(configured.parent)
    if discovered is not None:
        LOGGER.warning(
            "Configured LANTRA checkpoint is missing; using trainer-compatible discovered checkpoint %s",
            discovered,
        )
        return discovered

    raise FileNotFoundError(
        "Configured LANTRA checkpoint was not found and no trainer-compatible checkpoint could be discovered "
        f"in {configured.parent}. Configured path: {configured}"
    )


class SLAILM:
    """LANTRA-first SLAI v2.3 chatbot orchestration layer."""

    def __init__(self, settings: SLAILMSettings) -> None:
        self.settings = settings
        self.session_id = settings.session_id or f"slailm-{uuid.uuid4().hex[:12]}"
        self.shared_memory = SharedMemory()
        self.agent_factory = AgentFactory()
        self.language_memory = LanguageMemory()
        self._agents: Dict[str, Any] = {}
        self._agent_init_failures: Dict[str, str] = {}
        self.last_turn: Optional[TurnResult] = None
        self.last_used_agents: List[str] = []
        self.debug = settings.debug

        checkpoint = resolve_checkpoint(settings.checkpoint)
        runtime_override: Dict[str, Any] = {"checkpoint_path": str(checkpoint)}
        if settings.device:
            runtime_override["device"] = settings.device
        self.lantra = LantraRuntime(runtime_override)
        self.checkpoint_path = Path(self.lantra.checkpoint_path)
        self.stats = SessionStats(started_at=utc_now_iso())

        LOGGER.info(
            "SLAI-LM session initialized session=%s checkpoint=%s device=%s",
            self.session_id,
            self.checkpoint_path,
            self.lantra.device,
        )
        self._publish_stats()

    # ------------------------------------------------------------------
    # Optional agent lifecycle
    # ------------------------------------------------------------------
    def _agent(self, name: str) -> Any:
        """Lazily obtain one canonical AgentFactory-managed optional agent."""

        if name in self._agents:
            return self._agents[name]
        if name in self._agent_init_failures:
            return None

        try:
            agent = self.agent_factory.create(name, shared_memory=self.shared_memory)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self._agent_init_failures[name] = message
            self.stats.optional_agent_failures += 1
            LOGGER.warning("Optional SLAI-LM agent unavailable: %s | %s", name, message)
            if self.settings.strict_agents:
                raise
            self._publish_stats()
            return None

        self._agents[name] = agent
        LOGGER.info("SLAI-LM agent initialized: %s (%s)", name, type(agent).__name__)
        return agent

    def close(self) -> None:
        """Persist durable language memory and close initialized runtime resources."""

        LOGGER.info("Closing SLAI-LM session %s", self.session_id)
        self._publish_stats()
        self.language_memory.save_to_disk(force=True)

        for name, agent in list(self._agents.items()):
            shutdown = getattr(agent, "shutdown", None)
            close = getattr(agent, "close", None)
            method = shutdown if callable(shutdown) else close if callable(close) else None
            if method is None:
                continue
            try:
                method()
            except Exception as exc:
                LOGGER.warning("Agent shutdown failed: %s | %s: %s", name, type(exc).__name__, exc)

        close_shared = getattr(self.shared_memory, "close", None)
        if callable(close_shared):
            close_shared()

    # ------------------------------------------------------------------
    # Conversation memory
    # ------------------------------------------------------------------
    def _remember_turn(self, role: MemoryRole, content: str, *, turn_id: str) -> None:
        self.language_memory.remember_turn(
            role,
            content,
            scope=MemoryScope.SESSION,
            source=CONVERSATION_SOURCE,
            tags=("slailm", f"session:{self.session_id}"),
            metadata={"session_id": self.session_id, "turn_id": turn_id},
        )

    def _session_history(self, *, limit: Optional[int] = None) -> List[Dict[str, str]]:
        effective_limit = limit or self.settings.context_turns
        matches = self.language_memory.recall(
            MemoryQuery(
                kinds=(MemoryKind.TURN,),
                scopes=(MemoryScope.SESSION,),
                source=CONVERSATION_SOURCE,
                top_k=max(1, effective_limit),
                min_score=0.0,
                metadata_filter=lambda metadata: metadata.get("session_id") == self.session_id,
            )
        )
        records = sorted((match.record for match in matches), key=lambda item: item.created_at)
        history: List[Dict[str, str]] = []
        for record in records:
            if isinstance(record.value, Mapping):
                role = str(record.value.get("role") or record.role.value)
                content = str(record.value.get("content") or record.text)
            else:
                role = record.role.value
                content = record.text
            if content.strip():
                history.append({"role": role, "content": content})
        return history

    def clear_session(self) -> str:
        """Start a fresh conversational scope while retaining validated long-term feedback."""

        previous = self.session_id
        self.session_id = f"slailm-{uuid.uuid4().hex[:12]}"
        self.last_turn = None
        self.last_used_agents = []
        LOGGER.info("SLAI-LM session scope cleared old=%s new=%s", previous, self.session_id)
        self._publish_stats()
        return self.session_id

    # ------------------------------------------------------------------
    # Dynamic capability selection
    # ------------------------------------------------------------------
    @staticmethod
    def _is_simple_social(text: str) -> bool:
        normalized = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()
        if len(normalized) > 80:
            return False
        tokens = set(normalized.split())
        social = {
            "hi", "hello", "hey", "thanks", "thank", "you", "bye", "goodbye",
            "morning", "afternoon", "evening", "yo", "sup",
        }
        return bool(tokens) and tokens.issubset(social)

    @classmethod
    def _needs_knowledge(cls, text: str) -> bool:
        if cls._is_simple_social(text):
            return False
        lowered = text.lower().strip()
        question_starts = (
            "what ", "who ", "when ", "where ", "which ", "define ", "explain ",
            "tell me about ", "how many ", "how much ", "is ", "are ", "was ", "were ",
        )
        return "?" in text or lowered.startswith(question_starts)

    @classmethod
    def _needs_reasoning(cls, text: str) -> bool:
        if cls._is_simple_social(text):
            return False
        lowered = text.lower()
        markers = (
            "why ", "how ", "compare", "analyse", "analyze", "reason", "solve",
            "trade-off", "tradeoff", "pros and cons", "derive", "infer", "evaluate",
            "step by step", "plan", "because", "implication", "consequence",
        )
        return len(text) >= 180 or any(marker in lowered for marker in markers)

    @staticmethod
    def _needs_alignment(text: str) -> bool:
        lowered = text.lower()
        markers = (
            "ethical", "ethics", "moral", "fairness", "fair ", "bias", "biased",
            "discrimination", "rights", "consent", "harm", "values", "alignment",
            "protected group", "sensitive attribute",
        )
        return any(marker in lowered for marker in markers)

    # ------------------------------------------------------------------
    # Agent adapters using real v2.3 public APIs
    # ------------------------------------------------------------------
    def _safety_assess(self, text: str, *, context_type: str) -> SafetyAssessment:
        agent = self._agent("safety")
        if agent is None:
            return SafetyAssessment(
                available=False,
                decision="unavailable",
                blocked=False,
                review=True,
                sanitized_text=text,
                risk_score=None,
                error=self._agent_init_failures.get("safety", "SafetyAgent unavailable"),
            )

        try:
            raw = agent.perform_task(
                text,
                context={
                    "context_type": context_type,
                    "source": "slailm",
                    "session_id": self.session_id,
                },
            )
            result = _ensure_mapping(raw)
            decision = str(result.get("decision") or result.get("status") or "allow").strip().lower()
            blockers_raw = result.get("blockers") or []
            warnings_raw = result.get("warnings") or []
            degraded_raw = result.get("degraded_components") or []
            blockers = tuple(str(item) for item in blockers_raw) if isinstance(blockers_raw, Sequence) and not isinstance(blockers_raw, str) else ()
            warnings = tuple(str(item) for item in warnings_raw) if isinstance(warnings_raw, Sequence) and not isinstance(warnings_raw, str) else ()
            degraded = tuple(str(item) for item in degraded_raw) if isinstance(degraded_raw, Sequence) and not isinstance(degraded_raw, str) else ()
            is_safe = result.get("is_safe")
            blocked = decision in {"block", "blocked", "deny", "denied"} or bool(blockers)
            if is_safe is False:
                blocked = True
            review = decision in {"review", "review_required", "warn", "warning"} or bool(warnings) or bool(degraded)
            sanitized = str(result.get("sanitized_text") or text)
            risk_value = result.get("risk_score")
            risk_score = float(risk_value) if isinstance(risk_value, (int, float)) else None
            return SafetyAssessment(
                available=True,
                decision=decision,
                blocked=blocked,
                review=review,
                sanitized_text=sanitized,
                risk_score=risk_score,
                blockers=blockers,
                warnings=warnings,
                degraded_components=degraded,
            )
        except Exception as exc:
            LOGGER.warning("SafetyAgent assessment failed: %s: %s", type(exc).__name__, exc)
            self.stats.optional_agent_failures += 1
            if self.settings.strict_agents:
                raise
            return SafetyAssessment(
                available=False,
                decision="error",
                blocked=False,
                review=True,
                sanitized_text=text,
                risk_score=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _knowledge_context(self, query: str) -> List[str]:
        agent = self._agent("knowledge")
        if agent is None:
            return []
        try:
            retrieved = agent.retrieve(query, k=self.settings.knowledge_results)
        except Exception as exc:
            LOGGER.warning("Knowledge retrieval failed: %s: %s", type(exc).__name__, exc)
            self.stats.optional_agent_failures += 1
            if self.settings.strict_agents:
                raise
            return []

        snippets: List[str] = []
        for item in retrieved or []:
            score: Any = None
            document: Any = item
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)) and len(item) >= 2:
                score, document = item[0], item[1]
            if isinstance(document, Mapping):
                text = document.get("text") or document.get("content") or document.get("summary")
                doc_id = document.get("doc_id") or document.get("id")
            else:
                text = getattr(document, "text", None) or getattr(document, "content", None)
                doc_id = getattr(document, "doc_id", None) or getattr(document, "id", None)
            if not text:
                continue
            prefix = f"[{doc_id}] " if doc_id else ""
            if isinstance(score, (int, float)):
                prefix += f"retrieval_score={float(score):.4f} "
            snippets.append(
                compact_text(prefix + str(text), max_length=self.settings.max_snippet_chars)
            )
        return snippets

    def _reasoning_context(self, query: str, knowledge: Sequence[str], history: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
        agent = self._agent("reasoning")
        if agent is None:
            return {}
        context = {
            "knowledge": list(knowledge),
            "conversation": list(history[-4:]),
            "source": "slailm",
            "session_id": self.session_id,
        }
        try:
            raw = agent.reason(query, reasoning_type="auto", context=context)
        except Exception as exc:
            LOGGER.warning("ReasoningAgent failed: %s: %s", type(exc).__name__, exc)
            self.stats.optional_agent_failures += 1
            if self.settings.strict_agents:
                raise
            return {}

        result = _ensure_mapping(raw)
        projection: Dict[str, Any] = {}
        for key in (
            "conclusion", "result", "answer", "outcome", "confidence", "validation",
            "reasoning_type", "selected_reasoning_type", "strategy", "status",
        ):
            if key in result and result[key] not in (None, "", [], {}):
                projection[key] = json_safe(result[key])
        return projection or {"summary": _compact(result, self.settings.max_snippet_chars)}

    def _browser_context(self, query: str) -> Tuple[str, Optional[str]]:
        agent = self._agent("browser")
        if agent is None:
            return "", self._agent_init_failures.get("browser", "BrowserAgent unavailable")
        try:
            result = agent.perform_task(
                {
                    "task": "search",
                    "query": query,
                    "max_results": max(3, self.settings.knowledge_results),
                }
            )
        except Exception as exc:
            LOGGER.warning("BrowserAgent search failed: %s: %s", type(exc).__name__, exc)
            self.stats.optional_agent_failures += 1
            if self.settings.strict_agents:
                raise
            return "", f"{type(exc).__name__}: {exc}"
        return _collect_text_fields(result, max_chars=self.settings.max_support_chars), None

    def _reader_context(self, path: str) -> Tuple[str, Optional[str]]:
        agent = self._agent("reader")
        if agent is None:
            return "", self._agent_init_failures.get("reader", "ReaderAgent unavailable")
        try:
            result = agent.perform_task(
                {
                    "operation": "read",
                    "files": [path],
                    "instruction": "Read this document as bounded context for a LANTRA dialogue response.",
                    "include_documents": True,
                    "include_content": True,
                    "recover": True,
                    "merge": False,
                }
            )
        except Exception as exc:
            LOGGER.warning("ReaderAgent failed: %s: %s", type(exc).__name__, exc)
            self.stats.optional_agent_failures += 1
            if self.settings.strict_agents:
                raise
            return "", f"{type(exc).__name__}: {exc}"
        return _collect_text_fields(result, max_chars=self.settings.max_support_chars), None

    def _alignment_assess(self, user_input: str, candidate: str) -> AlignmentAssessment:
        agent = self._agent("alignment")
        if agent is None:
            return AlignmentAssessment(
                available=False,
                approved=None,
                requires_review=False,
                status=None,
                correction_action=None,
                error=self._agent_init_failures.get("alignment", "AlignmentAgent unavailable"),
            )
        try:
            raw = agent.align(
                input_data=user_input,
                predictions=candidate,
                task_context={
                    "task_id": f"slailm-{uuid.uuid4().hex[:12]}",
                    "context": {
                        "source": "slailm",
                        "session_id": self.session_id,
                    },
                },
            )
            result = _ensure_mapping(raw)
            decision = result.get("decision") if isinstance(result.get("decision"), Mapping) else {}
            reasons_raw = decision.get("reasons") or []
            reasons = tuple(str(item) for item in reasons_raw) if isinstance(reasons_raw, Sequence) and not isinstance(reasons_raw, str) else ()
            approved_raw = decision.get("approved")
            approved = bool(approved_raw) if isinstance(approved_raw, bool) else None
            return AlignmentAssessment(
                available=True,
                approved=approved,
                requires_review=bool(decision.get("requires_review", False)),
                status=str(decision.get("alignment_status")) if decision.get("alignment_status") is not None else None,
                correction_action=str(decision.get("correction_action")) if decision.get("correction_action") is not None else None,
                reasons=reasons,
            )
        except Exception as exc:
            LOGGER.warning("AlignmentAgent assessment failed: %s: %s", type(exc).__name__, exc)
            self.stats.optional_agent_failures += 1
            if self.settings.strict_agents:
                raise
            return AlignmentAssessment(
                available=False,
                approved=None,
                requires_review=False,
                status=None,
                correction_action=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    # ------------------------------------------------------------------
    # Persistent human-guided adaptation
    # ------------------------------------------------------------------
    def _relevant_validated_feedback(self, query: str) -> List[str]:
        matches = self.language_memory.recall(
            MemoryQuery(
                text=query,
                kinds=(MemoryKind.NOTE,),
                scopes=(MemoryScope.USER,),
                source=FEEDBACK_SOURCE,
                top_k=DEFAULT_FEEDBACK_RECALL,
                min_score=0.0,
                metadata_filter=lambda metadata: bool(metadata.get("runtime_adaptation_eligible")),
            )
        )

        # LanguageMemory exposes the contribution of lexical overlap.  Requiring
        # actual overlap prevents recency alone from injecting unrelated feedback.
        relevant = [match for match in matches if float(match.reasons.get("lexical", 0.0)) > 0.0]
        if not relevant:
            return []

        candidates = [match.record.text for match in relevant if match.record.text.strip()]
        if len(candidates) <= 3:
            return candidates

        try:
            ranked = self.lantra.rerank(query, candidates, top_k=3)
            return [item.text for item in ranked.candidates]
        except Exception as exc:
            LOGGER.warning("LANTRA feedback reranking unavailable; using LanguageMemory ranking: %s", exc)
            return candidates[:3]

    def _store_feedback(self, record: FeedbackRecord) -> None:
        target = record.validated_response or record.original_lantra_response
        if record.feedback_type == "correction":
            adaptation_text = (
                "Human-validated correction for a prior similar request. "
                f"Request: {record.original_user_input}\nValidated answer: {target}"
            )
        elif record.feedback_type == "validation":
            adaptation_text = (
                "Human explicitly validated this prior response for a similar request. "
                f"Request: {record.original_user_input}\nValidated answer: {target}"
            )
        else:
            adaptation_text = (
                f"Human feedback ({record.feedback_type}) for request: {record.original_user_input}"
            )

        self.language_memory.remember(
            MemoryKind.NOTE,
            key=f"feedback:{record.feedback_id}",
            value=record.to_dict(),
            text=adaptation_text,
            scope=MemoryScope.USER,
            role=MemoryRole.USER,
            source=FEEDBACK_SOURCE,
            confidence=1.0 if record.human_validated else 0.8,
            salience=0.9 if record.human_validated else 0.6,
            priority=2 if record.human_validated else 1,
            tags=(
                "slailm-feedback",
                record.feedback_type,
                "human-validated" if record.human_validated else "preference-signal",
            ),
            metadata={
                "feedback_id": record.feedback_id,
                "session_id": record.session_id,
                "turn_id": record.turn_id,
                "feedback_type": record.feedback_type,
                "human_validated": record.human_validated,
                "safety_validated": record.safety_validated,
                "runtime_adaptation_eligible": record.runtime_adaptation_eligible,
                "training_eligible": record.training_eligible,
            },
            replace_existing=True,
        )
        self.language_memory.save_to_disk(force=True)
        LOGGER.info(
            "Human feedback recorded id=%s type=%s runtime_adaptation=%s training_eligible=%s",
            record.feedback_id,
            record.feedback_type,
            record.runtime_adaptation_eligible,
            record.training_eligible,
        )

    def record_feedback(
        self,
        feedback_type: str,
        *,
        correction: Optional[str] = None,
        turn: Optional[TurnResult] = None,
    ) -> FeedbackRecord:
        """Validate and persist explicit human feedback for a completed turn."""

        target_turn = turn or self.last_turn
        if target_turn is None:
            raise ValueError("No completed SLAI-LM turn is available for feedback.")

        normalized = feedback_type.strip().lower()
        aliases = {
            "good": "positive",
            "+": "positive",
            "yes": "positive",
            "bad": "negative",
            "-": "negative",
            "no": "negative",
            "correct": "correction",
            "correction": "correction",
            "validate": "validation",
            "validated": "validation",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"positive", "negative", "correction", "validation"}:
            raise ValueError("Feedback must be positive, negative, correction, or validation.")
        if normalized == "correction" and not str(correction or "").strip():
            raise ValueError("Correction feedback requires non-empty corrected text.")

        if normalized == "correction":
            requested_target = str(correction).strip()
        elif normalized == "validation":
            requested_target = target_turn.response
        else:
            requested_target = None

        notes: List[str] = []
        safety_validated = False
        safe_target = requested_target
        alignment_result: Dict[str, Any] = {}

        if requested_target is not None:
            safety = self._safety_assess(requested_target, context_type="slailm_feedback")
            notes.append(f"safety:{safety.decision}")
            if safety.available and not safety.blocked:
                safety_validated = True
                safe_target = safety.sanitized_text
            elif safety.available and safety.blocked:
                notes.append("feedback retained but excluded from runtime adaptation/training by SafetyAgent")
            else:
                notes.append("feedback retained but excluded from runtime adaptation/training because SafetyAgent validation was unavailable")

            if self._needs_alignment(target_turn.user_input) or normalized in {"correction", "validation"}:
                alignment = self._alignment_assess(target_turn.user_input, safe_target or requested_target)
                alignment_result = alignment.to_dict()
                if alignment.available and alignment.requires_review:
                    notes.append(
                        f"alignment advisory review: {alignment.correction_action or alignment.status or 'review_required'}"
                    )

        human_validated = normalized in {"correction", "validation"}
        reusable = bool(human_validated and safety_validated and safe_target)
        quality_signal = {
            "positive": "accepted_preference",
            "negative": "rejected_response",
            "correction": "human_corrected",
            "validation": "human_validated",
        }[normalized]

        feedback_id = stable_hash(
            {
                "session": target_turn.session_id,
                "turn": target_turn.turn_id,
                "type": normalized,
                "target": safe_target,
            },
            length=24,
        )
        feedback = FeedbackRecord(
            schema="slai.slailm.feedback.v1",
            feedback_id=feedback_id,
            session_id=target_turn.session_id,
            timestamp=utc_now_iso(),
            turn_id=target_turn.turn_id,
            feedback_type=normalized,
            quality_signal=quality_signal,
            original_user_input=target_turn.user_input,
            original_lantra_response=target_turn.original_lantra_response,
            validated_response=safe_target if human_validated else None,
            conversation_history=target_turn.training_history,
            participating_agents=list(target_turn.participating_agents),
            evaluation_results=json_safe(
                {
                    **target_turn.evaluation_results,
                    "feedback_alignment": alignment_result,
                }
            ),
            model_checkpoint=self._model_identifier(),
            generation_metadata=json_safe(target_turn.generation_metadata),
            human_validated=human_validated,
            safety_validated=safety_validated,
            runtime_adaptation_eligible=reusable,
            training_eligible=reusable,
            validation_notes=notes,
        )
        self._store_feedback(feedback)

        self.stats.feedback_total += 1
        if normalized == "positive":
            self.stats.feedback_positive += 1
        elif normalized == "negative":
            self.stats.feedback_negative += 1
        elif normalized == "correction":
            self.stats.feedback_corrections += 1
        elif normalized == "validation":
            self.stats.feedback_validations += 1

        if feedback.training_eligible and self.settings.auto_export_feedback:
            exported = self.export_feedback(records=[feedback])
            self.stats.exported_training_examples += exported["added"]

        self._publish_stats()
        return feedback

    # ------------------------------------------------------------------
    # LANTRA training-compatible export
    # ------------------------------------------------------------------
    @staticmethod
    def _default_feedback_export_path() -> Path:
        from train_lantra import DEFAULT_DATA_CANDIDATES

        return Path(DEFAULT_DATA_CANDIDATES[0]) / "slailm_feedback" / "validated_dialogue.jsonl"

    def _eligible_feedback_records(self) -> List[FeedbackRecord]:
        matches = self.language_memory.recall(
            MemoryQuery(
                kinds=(MemoryKind.NOTE,),
                scopes=(MemoryScope.USER,),
                source=FEEDBACK_SOURCE,
                top_k=max(1, int(self.language_memory.settings.max_records)),
                min_score=0.0,
                metadata_filter=lambda metadata: bool(metadata.get("training_eligible")),
            )
        )
        records: Dict[str, FeedbackRecord] = {}
        for match in matches:
            value = match.record.value
            if not isinstance(value, Mapping):
                continue
            try:
                record = FeedbackRecord(**dict(value))
            except TypeError:
                continue
            if record.training_eligible:
                records[record.feedback_id] = record
        return sorted(records.values(), key=lambda item: (item.timestamp, item.feedback_id))

    def _feedback_training_record(self, feedback: FeedbackRecord) -> Dict[str, Any]:
        target = feedback.validated_response or feedback.original_lantra_response
        history = [
            {"role": str(turn["role"]), "content": str(turn["content"])}
            for turn in feedback.conversation_history
            if isinstance(turn, Mapping) and turn.get("role") and turn.get("content")
        ]
        if not history or history[-1].get("role") != "user" or history[-1].get("content") != feedback.original_user_input:
            history.append({"role": "user", "content": feedback.original_user_input})

        record: Dict[str, Any] = {
            "id": f"slailm-{feedback.feedback_id}",
            "task": "dialogue",
            "split": "train",
            "history": history,
            "target": target,
            "metadata": {
                "source": "slailm_human_feedback",
                "feedback_id": feedback.feedback_id,
                "feedback_type": feedback.feedback_type,
                "human_validated": feedback.human_validated,
                "safety_validated": feedback.safety_validated,
                "model_checkpoint": feedback.model_checkpoint,
                "participating_agents": list(feedback.participating_agents),
                "recorded_at": feedback.timestamp,
            },
        }

        # train_lantra.normalize_record is the current trainer schema authority.
        from train_lantra import normalize_record

        normalize_record(record, "<slailm-feedback>", 1)
        return record

    @staticmethod
    def _read_existing_jsonl(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                value = json.loads(stripped)
                if not isinstance(value, Mapping):
                    raise ValueError(f"Existing training record at {path}:{line_number} is not an object.")
                records.append(dict(value))
        return records

    @staticmethod
    def _atomic_write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(json_safe(record), ensure_ascii=False, sort_keys=True, allow_nan=False))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def export_feedback(
        self,
        path: Optional[Path] = None,
        *,
        records: Optional[Sequence[FeedbackRecord]] = None,
    ) -> Dict[str, Any]:
        """Export only human- and SafetyAgent-validated examples to LANTRA dialogue JSONL."""

        target = path or self._default_feedback_export_path()
        source_records = list(records) if records is not None else self._eligible_feedback_records()
        training_records = [self._feedback_training_record(item) for item in source_records if item.training_eligible]

        existing = self._read_existing_jsonl(target)
        existing_ids = {str(item.get("id")) for item in existing if item.get("id") is not None}
        added = 0
        for record in training_records:
            record_id = str(record.get("id"))
            if record_id in existing_ids:
                continue
            existing.append(record)
            existing_ids.add(record_id)
            added += 1

        if added:
            self._atomic_write_jsonl(target, existing)
            LOGGER.info("Queued %s validated SLAI-LM dialogue example(s) at %s", added, target)

        return {
            "path": str(target),
            "eligible": len(training_records),
            "added": added,
            "total": len(existing),
        }

    # ------------------------------------------------------------------
    # Context construction and generation
    # ------------------------------------------------------------------
    def _build_support(
        self,
        query: str,
        history: Sequence[Mapping[str, str]],
        *,
        input_safety: Optional[SafetyAssessment] = None,
        force_browser: bool = False,
        browser_query: Optional[str] = None,
        reader_path: Optional[str] = None,
    ) -> SupportBundle:
        support = SupportBundle()

        safety = input_safety or self._safety_assess(query, context_type="slailm_input")
        support.safety = safety.to_dict()
        if safety.available:
            support.mark_agent("safety")
        elif safety.error:
            support.failures["safety"] = safety.error

        support.validated_feedback = self._relevant_validated_feedback(query)

        if self._needs_knowledge(query):
            knowledge = self._knowledge_context(query)
            if knowledge:
                support.knowledge = knowledge
                support.mark_agent("knowledge")
            elif "knowledge" in self._agent_init_failures:
                support.failures["knowledge"] = self._agent_init_failures["knowledge"]

        if self._needs_reasoning(query):
            reasoning = self._reasoning_context(query, support.knowledge, history)
            if reasoning:
                support.reasoning = reasoning
                support.mark_agent("reasoning")
            elif "reasoning" in self._agent_init_failures:
                support.failures["reasoning"] = self._agent_init_failures["reasoning"]

        if force_browser:
            browser_text, error = self._browser_context(browser_query or query)
            if browser_text:
                support.browser = browser_text
                support.mark_agent("browser")
            if error:
                support.failures["browser"] = error

        if reader_path:
            reader_text, error = self._reader_context(reader_path)
            if reader_text:
                support.reader = reader_text
                support.mark_agent("reader")
            if error:
                support.failures["reader"] = error

        return support

    def _support_text(self, support: SupportBundle) -> str:
        sections: List[str] = [
            "SLAI support context for LANTRA. Use only relevant evidence. "
            "Human-validated corrections have priority over automated support. "
            "Do not claim certainty when evidence is insufficient, and do not mention internal agent names unless asked."
        ]

        safety = support.safety
        if safety and (safety.get("blocked") or safety.get("review")):
            sections.append(
                "Safety constraint: the request/output requires a safe, non-harmful response. "
                f"Decision={safety.get('decision')}; blockers={safety.get('blockers')}; warnings={safety.get('warnings')}"
            )

        if support.validated_feedback:
            sections.append(
                "Human-validated prior feedback:\n"
                + "\n".join(f"- {item}" for item in support.validated_feedback)
            )
        if support.knowledge:
            sections.append("Retrieved knowledge:\n" + "\n".join(f"- {item}" for item in support.knowledge))
        if support.reasoning:
            sections.append("Reasoning support:\n" + _compact(support.reasoning, self.settings.max_snippet_chars))
        if support.browser:
            sections.append("Browser evidence:\n" + support.browser)
        if support.reader:
            sections.append("Reader document context:\n" + support.reader)
        if support.failures:
            unavailable = ", ".join(sorted(support.failures))
            sections.append(
                f"Some optional evidence sources were unavailable ({unavailable}). Do not invent their missing evidence."
            )

        return compact_text("\n\n".join(sections), max_length=self.settings.max_support_chars)

    def _lantra_history(
        self,
        prior_history: Sequence[Mapping[str, str]],
        user_input: str,
        support: SupportBundle,
        *,
        correction_instruction: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        support_text = self._support_text(support)
        if correction_instruction:
            support_text = compact_text(
                support_text + "\n\nRegeneration instruction: " + correction_instruction,
                max_length=self.settings.max_support_chars,
            )

        # Keep the dialogue comfortably below LANTRA's configured source budget.
        # This is a conservative character budget; LANTRA remains responsible for
        # the actual tokenizer truncation and sequence validation.
        char_budget = max(800, int(self.lantra.source_max_length) * 3)
        reserved = len(user_input) + len(support_text) + 80
        remaining = max(0, char_budget - reserved)

        selected_reversed: List[Dict[str, str]] = []
        for turn in reversed(list(prior_history)):
            role = str(turn.get("role") or "unknown")
            content = str(turn.get("content") or "")
            cost = len(role) + len(content) + 4
            if cost > remaining:
                continue
            selected_reversed.append({"role": role, "content": content})
            remaining -= cost
        selected = list(reversed(selected_reversed))

        if support_text and len(user_input) < char_budget:
            selected.append({"role": "system", "content": support_text})
        selected.append({"role": "user", "content": user_input})
        return selected

    def _evaluate_candidate(
        self,
        user_input: str,
        candidate: str,
        *,
        alignment_required: bool,
    ) -> Tuple[str, Dict[str, Any], List[str], List[str]]:
        """Return sanitized candidate, evaluation evidence, defects, and used agents."""

        defects: List[str] = []
        used_agents: List[str] = []
        evaluations: Dict[str, Any] = {}

        if not candidate.strip():
            defects.append("LANTRA returned an empty response")

        safety = self._safety_assess(candidate or "empty response", context_type="slailm_output")
        evaluations["safety"] = safety.to_dict()
        if safety.available:
            used_agents.append("safety")
        if safety.blocked:
            defects.append(
                "SafetyAgent blocked the draft"
                + (f": {', '.join(safety.blockers)}" if safety.blockers else "")
            )
        elif safety.review:
            evaluations["safety_review"] = True
        sanitized = safety.sanitized_text if safety.available and not safety.blocked else candidate

        if alignment_required:
            alignment = self._alignment_assess(user_input, sanitized)
            evaluations["alignment"] = alignment.to_dict()
            if alignment.available:
                used_agents.append("alignment")
            if alignment.available and alignment.requires_review:
                reason = alignment.correction_action or alignment.status or "review_required"
                defects.append(f"AlignmentAgent requested review: {reason}")

        return sanitized, evaluations, defects, used_agents

    def respond(
        self,
        user_input: str,
        *,
        force_browser: bool = False,
        browser_query: Optional[str] = None,
        reader_path: Optional[str] = None,
    ) -> TurnResult:
        """Execute one bounded multi-agent LANTRA dialogue cycle."""

        raw_input = str(user_input).strip()
        if not raw_input:
            raise ValueError("User input cannot be empty.")

        started = time.perf_counter()
        turn_id = f"turn-{uuid.uuid4().hex[:16]}"
        prior_history = self._session_history(limit=self.settings.context_turns)

        # Input safety can redact sensitive spans before they become durable
        # feedback/training provenance.  If unavailable, the original input is
        # still usable for dialogue, but future feedback reuse fails closed.
        input_safety = self._safety_assess(raw_input, context_type="slailm_input")
        safe_input = input_safety.sanitized_text if input_safety.available else raw_input
        self._remember_turn(MemoryRole.USER, safe_input, turn_id=turn_id)
        training_history = [dict(turn) for turn in prior_history] + [
            {"role": "user", "content": safe_input}
        ]

        support = self._build_support(
            safe_input,
            prior_history,
            input_safety=input_safety,
            force_browser=force_browser,
            browser_query=browser_query,
            reader_path=reader_path,
        )

        alignment_required = self._needs_alignment(safe_input) or bool(input_safety.review)
        history = self._lantra_history(prior_history, safe_input, support)

        original_lantra_response = ""
        final_response = ""
        generation_metadata: Dict[str, Any] = {}
        evaluation_results: Dict[str, Any] = {}
        regeneration_count = 0
        failsafe_used = False
        used_agents = ["language", *support.participating_agents]
        correction_instruction: Optional[str] = None

        for attempt in range(self.settings.max_regenerations + 1):
            if attempt > 0:
                history = self._lantra_history(
                    prior_history,
                    safe_input,
                    support,
                    correction_instruction=correction_instruction,
                )

            LOGGER.info(
                "LANTRA generation started session=%s turn=%s attempt=%s agents=%s",
                self.session_id,
                turn_id,
                attempt + 1,
                sorted(set(used_agents)),
            )
            result: LantraTextResult = self.lantra.dialogue(
                history,
                max_length=self.settings.max_output_tokens,
            )
            self.stats.lantra_generation_calls += 1
            candidate = result.text.strip()
            if attempt == 0:
                original_lantra_response = candidate
            generation_metadata = result.to_dict()

            sanitized, evaluations, defects, evaluation_agents = self._evaluate_candidate(
                safe_input,
                candidate,
                alignment_required=alignment_required,
            )
            for name in evaluation_agents:
                if name not in used_agents:
                    used_agents.append(name)
            evaluation_results[f"attempt_{attempt + 1}"] = evaluations

            if not defects:
                final_response = sanitized
                break

            LOGGER.warning(
                "LANTRA draft requires correction session=%s turn=%s attempt=%s defects=%s",
                self.session_id,
                turn_id,
                attempt + 1,
                defects,
            )
            if attempt >= self.settings.max_regenerations:
                final_response = sanitized
                final_safety = evaluations.get("safety") if isinstance(evaluations.get("safety"), Mapping) else {}
                if final_safety and bool(final_safety.get("blocked")):
                    final_response = "I can't provide that response safely."
                    failsafe_used = True
                break

            regeneration_count += 1
            self.stats.regenerations += 1
            correction_instruction = (
                "Correct the previous draft without repeating it. Address these evaluator findings: "
                + "; ".join(defects[:4])
                + ". Answer the user's request directly while respecting the supplied safety/alignment context."
            )

        if not final_response.strip():
            final_response = "I could not produce a usable LANTRA response for that request."
            failsafe_used = True

        self._remember_turn(MemoryRole.ASSISTANT, final_response, turn_id=turn_id)

        latency_ms = (time.perf_counter() - started) * 1000.0
        self.stats.turns += 1
        self.stats.total_turn_latency_ms += latency_ms
        self.last_used_agents = sorted(set(used_agents))
        context_summary = {
            "validated_feedback_count": len(support.validated_feedback),
            "knowledge_count": len(support.knowledge),
            "reasoning_used": bool(support.reasoning),
            "browser_used": bool(support.browser),
            "reader_used": bool(support.reader),
            "optional_failures": sorted(support.failures),
        }

        turn = TurnResult(
            turn_id=turn_id,
            session_id=self.session_id,
            user_input=safe_input,
            response=final_response,
            original_lantra_response=original_lantra_response,
            training_history=training_history,
            participating_agents=self.last_used_agents,
            evaluation_results=json_safe(evaluation_results),
            generation_metadata=json_safe(generation_metadata),
            context_summary=context_summary,
            latency_ms=latency_ms,
            regeneration_count=regeneration_count,
            failsafe_used=failsafe_used,
        )
        self.last_turn = turn
        self._publish_turn(turn)
        self._publish_stats()
        LOGGER.info(
            "LANTRA generation completed session=%s turn=%s latency_ms=%.2f regenerations=%s",
            self.session_id,
            turn_id,
            latency_ms,
            regeneration_count,
        )
        return turn

    # ------------------------------------------------------------------
    # Observability/status through existing logger + SharedMemory
    # ------------------------------------------------------------------
    def _publish_turn(self, turn: TurnResult) -> None:
        payload = {
            "turn_id": turn.turn_id,
            "session_id": turn.session_id,
            "latency_ms": round(turn.latency_ms, 3),
            "participating_agents": list(turn.participating_agents),
            "regeneration_count": turn.regeneration_count,
            "failsafe_used": turn.failsafe_used,
            "context_summary": json_safe(turn.context_summary),
        }
        self.shared_memory.set(
            f"{SESSION_STATS_PREFIX}:{self.session_id}:last_turn",
            payload,
        )

    def _publish_stats(self) -> None:
        self.shared_memory.set(
            f"{SESSION_STATS_PREFIX}:{self.session_id}:stats",
            self.stats.to_dict(),
        )

    def _model_identifier(self) -> str:
        return str(self.checkpoint_path.name or self.checkpoint_path)

    def status(self) -> Dict[str, Any]:
        memory = self.language_memory.stats_snapshot()
        return {
            "version": __version__,
            "session_id": self.session_id,
            "model": self._model_identifier(),
            "device": str(self.lantra.device),
            "last_used_agents": list(self.last_used_agents),
            "initialized_agents": sorted(self._agents),
            "unavailable_agents": dict(self._agent_init_failures),
            "session_stats": self.stats.to_dict(),
            "language_memory": memory.get("counts", {}),
        }

    def model_status(self) -> Dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint_path),
            "runtime": self.lantra.stats().to_dict(),
        }

    def agent_status(self) -> Dict[str, Any]:
        return {
            name: {
                **policy,
                "initialized": name in self._agents,
                "initialization_error": self._agent_init_failures.get(name),
                "used_last_turn": name in self.last_used_agents,
            }
            for name, policy in AGENT_POLICY.items()
        }

    # ------------------------------------------------------------------
    # Evaluation mode: LANTRA alone vs LANTRA + SLAI-LM
    # ------------------------------------------------------------------
    def compare(self, prompt: str) -> Dict[str, Any]:
        """Compare raw LANTRA and the agent-assisted path without inventing a quality winner."""

        direct_started = time.perf_counter()
        direct = self.lantra.dialogue(
            [{"role": "user", "content": prompt}],
            max_length=self.settings.max_output_tokens,
        )
        direct_wall_ms = (time.perf_counter() - direct_started) * 1000.0

        assisted = self.respond(prompt)
        return {
            "prompt": prompt,
            "lantra_alone": {
                "response": direct.text,
                "runtime_latency_ms": direct.latency_ms,
                "wall_latency_ms": direct_wall_ms,
            },
            "slailm": {
                "response": assisted.response,
                "wall_latency_ms": assisted.latency_ms,
                "participating_agents": assisted.participating_agents,
                "regenerations": assisted.regeneration_count,
                "context_summary": assisted.context_summary,
            },
            "note": (
                "No automatic quality winner is declared because the current EvaluationAgent does not provide "
                "a ground-truth-free generative quality metric. Use human feedback or an external held-out target."
            ),
        }


# ----------------------------------------------------------------------
# CLI presentation
# ----------------------------------------------------------------------
HELP_TEXT = """Commands:
  /help                       Show this help.
  /exit                       Exit SLAI-LM cleanly.
  /clear                      Start a fresh conversation scope; validated feedback remains reusable.
  /status                     Show model/session/memory status.
  /model                      Show LANTRA checkpoint/runtime stats.
  /agents                     Show current v2.3 agent integration policy and runtime state.
  /feedback                   Give feedback on the latest response.
  /browse <query>             Search with BrowserAgent, then let LANTRA answer from the evidence.
  /read <path> [:: question]  Read a local document with ReaderAgent, then let LANTRA answer.
  /compare <prompt>            Compare LANTRA alone with LANTRA + SLAI-LM architecture.
  /export [path]              Export validated feedback in train_lantra dialogue JSONL format.
  /debug on|off               Toggle compact per-turn diagnostics.
"""


def _print_json(payload: Any) -> None:
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True))


def _prompt_feedback(bot: SLAILM) -> None:
    if bot.last_turn is None:
        print("No response is available for feedback.")
        return
    try:
        raw = input("Feedback [good/bad/correct/validate/skip] > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not raw or raw in {"skip", "s"}:
        return

    correction: Optional[str] = None
    if raw in {"correct", "correction"}:
        try:
            correction = input("Correction > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not correction:
            print("Correction cancelled: empty text.")
            return

    try:
        feedback = bot.record_feedback(raw, correction=correction)
    except ValueError as exc:
        print(f"Feedback error: {exc}")
        return

    print(
        "Feedback recorded "
        f"(type={feedback.feedback_type}, runtime_adaptation={feedback.runtime_adaptation_eligible}, "
        f"training_eligible={feedback.training_eligible})."
    )


def _handle_command(bot: SLAILM, raw: str) -> bool:
    """Handle one slash command. Return False when the CLI should exit."""

    command, _, argument = raw.partition(" ")
    command = command.strip().lower()
    argument = argument.strip()

    if command in {"/exit", "/quit"}:
        return False
    if command == "/help":
        print(HELP_TEXT)
        return True
    if command == "/clear":
        print(f"New session: {bot.clear_session()}")
        return True
    if command == "/status":
        _print_json(bot.status())
        return True
    if command == "/model":
        _print_json(bot.model_status())
        return True
    if command == "/agents":
        _print_json(bot.agent_status())
        return True
    if command == "/feedback":
        _prompt_feedback(bot)
        return True
    if command == "/debug":
        value = argument.lower()
        if value not in {"on", "off"}:
            print(f"Debug is currently {'on' if bot.debug else 'off'}. Use /debug on or /debug off.")
        else:
            bot.debug = value == "on"
            print(f"Debug {'enabled' if bot.debug else 'disabled'}.")
        return True
    if command == "/browse":
        if not argument:
            print("Usage: /browse <query>")
            return True
        turn = bot.respond(argument, force_browser=True, browser_query=argument)
        print(f"SLAI-LM > {turn.response}")
        if bot.debug:
            _print_json(
                {
                    "agents": turn.participating_agents,
                    "context": turn.context_summary,
                    "latency_ms": turn.latency_ms,
                    "regenerations": turn.regeneration_count,
                }
            )
        return True
    if command == "/read":
        if not argument:
            print("Usage: /read <path> [:: question]")
            return True
        path_text, separator, question = argument.partition("::")
        path = Path(path_text.strip().strip('"')).expanduser()
        if not path.is_file():
            print(f"File not found: {path}")
            return True
        prompt = question.strip() if separator and question.strip() else f"Summarize and explain the relevant content of {path.name}."
        turn = bot.respond(prompt, reader_path=str(path))
        print(f"SLAI-LM > {turn.response}")
        if bot.debug:
            _print_json(
                {
                    "agents": turn.participating_agents,
                    "context": turn.context_summary,
                    "latency_ms": turn.latency_ms,
                    "regenerations": turn.regeneration_count,
                }
            )
        return True
    if command == "/compare":
        if not argument:
            print("Usage: /compare <prompt>")
            return True
        _print_json(bot.compare(argument))
        return True
    if command == "/export":
        path = Path(argument).expanduser() if argument else None
        result = bot.export_feedback(path=path)
        _print_json(result)
        return True

    print(f"Unknown command: {command}. Use /help.")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SLAI-LM: LANTRA-first multi-agent chatbot for SLAI v2.3",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="LANTRA checkpoint file or checkpoint directory. Defaults to language_config.yaml with trainer-compatible discovery fallback.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override LANTRA device policy (for example cpu, cuda, cuda:0, mps).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Optional per-response LANTRA max length; otherwise use lantra_runtime.inference.target_max_length.",
    )
    parser.add_argument(
        "--context-turns",
        type=int,
        default=DEFAULT_CONTEXT_TURNS,
        help=f"Maximum recent persisted session turns considered for dialogue (default: {DEFAULT_CONTEXT_TURNS}).",
    )
    parser.add_argument(
        "--max-regenerations",
        type=int,
        default=DEFAULT_MAX_REGENERATIONS,
        help=f"Bounded LANTRA correction attempts after safety/alignment findings (default: {DEFAULT_MAX_REGENERATIONS}).",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Optional explicit session identifier.",
    )
    parser.add_argument(
        "--no-feedback-prompt",
        action="store_true",
        help="Do not ask for feedback after each normal response; /feedback remains available.",
    )
    parser.add_argument(
        "--no-auto-export-feedback",
        action="store_true",
        help="Persist validated feedback in LanguageMemory but do not automatically queue training-compatible JSONL records.",
    )
    parser.add_argument(
        "--strict-agents",
        action="store_true",
        help="Fail the turn instead of degrading when an optional support agent fails.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging and compact per-turn diagnostics.")
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List discoverable local LANTRA checkpoints and exit without loading the model.",
    )
    return parser


def _settings_from_args(args: argparse.Namespace) -> SLAILMSettings:
    if args.context_turns < 1:
        raise ValueError("--context-turns must be >= 1")
    if args.max_regenerations < 0:
        raise ValueError("--max-regenerations must be >= 0")
    if args.max_output_tokens is not None and args.max_output_tokens < 2:
        raise ValueError("--max-output-tokens must be >= 2")
    return SLAILMSettings(
        checkpoint=args.checkpoint,
        device=args.device,
        max_output_tokens=args.max_output_tokens,
        context_turns=args.context_turns,
        max_regenerations=args.max_regenerations,
        feedback_prompt=not args.no_feedback_prompt,
        auto_export_feedback=not args.no_auto_export_feedback,
        strict_agents=args.strict_agents,
        debug=args.debug,
        session_id=args.session,
    )


def run_cli(bot: SLAILM) -> int:
    print(
        f"SLAI-LM v{__version__} | LANTRA={bot._model_identifier()} | device={bot.lantra.device} | "
        f"session={bot.session_id}"
    )
    print("Type /help for commands. LANTRA remains the response generator; support agents are selective.\n")

    while True:
        try:
            raw = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        if raw.startswith("/"):
            if not _handle_command(bot, raw):
                break
            continue

        try:
            turn = bot.respond(raw)
        except Exception as exc:
            LOGGER.error("SLAI-LM turn failed: %s: %s", type(exc).__name__, exc, exc_info=bot.debug)
            print(f"SLAI-LM error > {type(exc).__name__}: {exc}")
            continue

        print(f"SLAI-LM > {turn.response}")
        if bot.debug:
            _print_json(
                {
                    "turn_id": turn.turn_id,
                    "agents": turn.participating_agents,
                    "context": turn.context_summary,
                    "latency_ms": turn.latency_ms,
                    "regenerations": turn.regeneration_count,
                    "failsafe_used": turn.failsafe_used,
                }
            )
        if bot.settings.feedback_prompt:
            _prompt_feedback(bot)

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging_level = logging.DEBUG if args.debug else logging.INFO
    configure_logging(LoggingSettings(level=logging_level))

    if args.list_checkpoints:
        candidates = list_checkpoint_candidates(args.checkpoint)
        if not candidates:
            print("No LANTRA checkpoints discovered.")
            shutdown_logging()
            return 1
        for index, path in enumerate(candidates, 1):
            print(f"{index:>2}. {path}")
        shutdown_logging()
        return 0

    try:
        settings = _settings_from_args(args)
        bot = SLAILM(settings)
    except Exception as exc:
        LOGGER.error("SLAI-LM startup failed: %s: %s", type(exc).__name__, exc, exc_info=args.debug)
        print(f"SLAI-LM startup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        shutdown_logging()
        return 2

    try:
        return run_cli(bot)
    finally:
        bot.close()
        shutdown_logging()


if __name__ == "__main__":
    raise SystemExit(main())
