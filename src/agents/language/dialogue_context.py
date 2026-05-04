"""
Production-grade dialogue context module for the language subsystem.

Core Function:
Maintains conversational state between the language agent's pipeline stages and
keeps the agent coherent across turns.

Pipeline Position:
User Input
↓
OrthographyProcessor -> spellcheck + normalize
↓
NLPEngine -> tokenize, lemmatize, POS, etc.
↓
GrammarProcessor -> parse + grammar checks
↓
NLUEngine -> determine intent/entities
↓
DialogueContext -> assemble recent context, slots, unresolved issues, summary,
                   intent history, topic state, and personalization for NLG
↓
NLGEngine -> generate response
↓
DialogueContext -> log user/agent turn and persist updated state
↓
Agent Output

Responsibilities:
- Store short-term dialogue history with stable typed records and backward-
  compatible dict views for the current LanguageAgent.
- Track slots, entities, intents, unresolved issues, pending clarification state,
  user preferences, topic focus, and environment state.
- Provide compact, production-safe context strings for NLG without owning NLG.
- Integrate with LanguageMemory for durable, queryable language memory while
  preserving local in-process history for fast agent access.
- Summarize long conversations using a registered summarizer or deterministic
  extractive fallback.
- Persist and restore context state using language_config.yaml settings.
- Use language_helpers.py and language_error.py instead of duplicating generic
  normalization, serialization, redaction, config, ID, and diagnostics logic.

Why it matters:
The language agent should not behave as a stateless request/response wrapper.
DialogueContext gives the pipeline continuity while remaining a coordinator of
state, not a replacement for orthography, NLP, grammar, NLU, NLG, cache, or the
longer-term LanguageMemory layer.
"""

from __future__ import annotations

import os
import re
import tempfile
import time as time_module

from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.linguistic_frame import LinguisticFrame, SpeechActType
from .utils.language_cache import LanguageCache
from .utils.language_error import *
from .utils.language_helpers import *
from .language_memory import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Dialogue Context")
printer = PrettyPrinter()

ContextDict = Dict[str, Any]
SummarizerFn = Callable[[List[Dict[str, Any]], Optional[str]], str]
SimilarityFn = Callable[[str, str], float]


class DialogueRole(str, Enum):
    """Normalized dialogue roles used by the language agent."""

    USER = "user"
    AGENT = "agent"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    UNKNOWN = "unknown"


class ConversationPhase(str, Enum):
    """High-level conversation phase labels."""

    OPENING = "opening"
    ACTIVE = "active"
    CLARIFYING = "clarifying"
    RESOLVING = "resolving"
    CLOSING = "closing"
    STALE = "stale"


@dataclass(frozen=True)
class DialogueContextConfig:
    """Resolved config for DialogueContext from language_config.yaml."""

    version: str = "2.0"
    memory_limit: int = 20
    message_limit: int = 200
    threshold: float = 0.7
    required_slots: Tuple[str, ...] = ()
    include_summary: bool = True
    include_history: bool = True
    include_slots: bool = True
    include_unresolved: bool = True
    include_intents: bool = True
    include_preferences: bool = True
    include_environment: bool = False
    enable_summarization: bool = True
    enable_memory: bool = True
    enable_cache: bool = True
    enable_persistence: bool = True
    enable_topic_detection: bool = True
    enable_follow_up_detection: bool = True
    default_history_window: int = 8
    default_turn_window: int = 4
    max_context_chars: int = 6000
    text_preview_length: int = 240
    default_initial_history: Tuple[Any, ...] = ("System: Hello! How can I assist you today?",)
    initial_history: Tuple[Any, ...] = ()
    default_initial_summary: str = "The conversation has just begun."
    initial_summary: Optional[str] = None
    initial_environment_state: ContextDict = field(default_factory=dict)
    default_initial_environment_state: ContextDict = field(default_factory=lambda: {
        "session_id": None,
        "user_preferences": {},
        "last_intent": None,
    })
    follow_up_patterns_path: Optional[Path] = None
    follow_up_patterns: Tuple[str, ...] = ()
    topic_similarity_threshold: float = 0.65
    topic_lookback_window: int = 4
    topic_min_text_length: int = 3
    retain_last_messages: int = 2
    summary_update_strategy: str = "accumulate"
    max_summary_length: int = 500
    summary_cache_namespace: str = "dialogue_context"
    auto_save_interval: float = 300.0
    default_save_path: Optional[Path] = None
    session_timeout_seconds: float = 1800.0
    time_reference_format: str = "ISO"
    slot_validation_rules: ContextDict = field(default_factory=dict)
    strict_slots: bool = False
    auto_register_entities_as_slots: bool = True
    role_aliases: ContextDict = field(default_factory=dict)
    redact_persistence: bool = False
    persistence_pretty: bool = True

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "DialogueContextConfig":
        section = dict(config or {})
        topic_detection = ensure_mapping(section.get("topic_detection", {}), field_name="dialogue_context.topic_detection", allow_none=True)
        summarization = ensure_mapping(section.get("summarization", {}), field_name="dialogue_context.summarization", allow_none=True)
        persistence = ensure_mapping(section.get("persistence", {}), field_name="dialogue_context.persistence", allow_none=True)
        temporal = ensure_mapping(section.get("temporal", {}), field_name="dialogue_context.temporal", allow_none=True)

        initial_env = ensure_mapping(section.get("initial_environment_state", {}), field_name="dialogue_context.initial_environment_state", allow_none=True)
        default_env = ensure_mapping(section.get("default_initial_environment_state", {}), field_name="dialogue_context.default_initial_environment_state", allow_none=True)
        role_aliases = ensure_mapping(section.get("role_aliases", {}), field_name="dialogue_context.role_aliases", allow_none=True)
        slot_rules = ensure_mapping(section.get("slot_validation_rules", {}), field_name="dialogue_context.slot_validation_rules", allow_none=True)

        follow_path_raw = section.get("follow_up_patterns_path")
        follow_path = None if follow_path_raw in (None, "", "none", "None") else resolve_path(follow_path_raw, field_name="dialogue_context.follow_up_patterns_path")
        save_path_raw = persistence.get("default_save_path", section.get("default_save_path"))
        save_path = None if save_path_raw in (None, "", "none", "None") else resolve_path(save_path_raw, field_name="dialogue_context.persistence.default_save_path")

        return cls(
            version=ensure_text(section.get("version", "2.0")),
            memory_limit=coerce_int(section.get("memory_limit", 20), default=20, minimum=1),
            message_limit=coerce_int(section.get("message_limit", section.get("history_limit", 200)), default=200, minimum=2),
            threshold=coerce_float(section.get("threshold", 0.7), default=0.7, minimum=0.0, maximum=1.0),
            required_slots=tuple(ensure_text(slot) for slot in ensure_list(section.get("required_slots", [])) if ensure_text(slot)),
            include_summary=coerce_bool(section.get("include_summary", True), default=True),
            include_history=coerce_bool(section.get("include_history", True), default=True),
            include_slots=coerce_bool(section.get("include_slots", True), default=True),
            include_unresolved=coerce_bool(section.get("include_unresolved", True), default=True),
            include_intents=coerce_bool(section.get("include_intents", True), default=True),
            include_preferences=coerce_bool(section.get("include_preferences", True), default=True),
            include_environment=coerce_bool(section.get("include_environment", False), default=False),
            enable_summarization=coerce_bool(section.get("enable_summarization", True), default=True),
            enable_memory=coerce_bool(section.get("enable_memory", True), default=True),
            enable_cache=coerce_bool(section.get("enable_cache", True), default=True),
            enable_persistence=coerce_bool(section.get("enable_persistence", True), default=True),
            enable_topic_detection=coerce_bool(section.get("enable_topic_detection", True), default=True),
            enable_follow_up_detection=coerce_bool(section.get("enable_follow_up_detection", True), default=True),
            default_history_window=coerce_int(section.get("default_history_window", 8), default=8, minimum=1),
            default_turn_window=coerce_int(section.get("default_turn_window", 4), default=4, minimum=1),
            max_context_chars=coerce_int(section.get("max_context_chars", 6000), default=6000, minimum=512),
            text_preview_length=coerce_int(section.get("text_preview_length", 240), default=240, minimum=40),
            default_initial_history=tuple(ensure_list(section.get("default_initial_history", ["System: Hello! How can I assist you today?"]))),
            initial_history=tuple(ensure_list(section.get("initial_history", []))),
            default_initial_summary=ensure_text(section.get("default_initial_summary", "The conversation has just begun.")),
            initial_summary=None if section.get("initial_summary") is None else ensure_text(section.get("initial_summary")),
            initial_environment_state=initial_env,
            default_initial_environment_state=default_env or {"session_id": None, "user_preferences": {}, "last_intent": None},
            follow_up_patterns_path=follow_path,
            follow_up_patterns=tuple(ensure_text(item) for item in ensure_list(section.get("follow_up_patterns", [])) if ensure_text(item)),
            topic_similarity_threshold=coerce_float(topic_detection.get("similarity_threshold", 0.65), default=0.65, minimum=0.0, maximum=1.0),
            topic_lookback_window=coerce_int(topic_detection.get("lookback_window", 4), default=4, minimum=1),
            topic_min_text_length=coerce_int(topic_detection.get("min_text_length", 3), default=3, minimum=1),
            retain_last_messages=coerce_int(summarization.get("retain_last_messages", 2), default=2, minimum=0),
            summary_update_strategy=normalize_identifier_component(summarization.get("summary_update_strategy", "accumulate"), default="accumulate", lowercase=True),
            max_summary_length=coerce_int(summarization.get("max_summary_length", 500), default=500, minimum=80),
            summary_cache_namespace=normalize_identifier_component(summarization.get("cache_namespace", "dialogue_context"), default="dialogue_context", lowercase=True),
            auto_save_interval=coerce_float(persistence.get("auto_save_interval", 300), default=300.0, minimum=0.0),
            default_save_path=save_path,
            session_timeout_seconds=coerce_float(temporal.get("session_timeout", 1800), default=1800.0, minimum=0.0),
            time_reference_format=ensure_text(temporal.get("time_reference_format", "ISO")),
            slot_validation_rules=slot_rules,
            strict_slots=coerce_bool(section.get("strict_slots", False), default=False),
            auto_register_entities_as_slots=coerce_bool(section.get("auto_register_entities_as_slots", True), default=True),
            role_aliases=role_aliases,
            redact_persistence=coerce_bool(persistence.get("redact", False), default=False),
            persistence_pretty=coerce_bool(persistence.get("pretty", True), default=True),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["required_slots"] = list(self.required_slots)
        payload["default_initial_history"] = list(self.default_initial_history)
        payload["initial_history"] = list(self.initial_history)
        payload["follow_up_patterns_path"] = str(self.follow_up_patterns_path) if self.follow_up_patterns_path else None
        payload["follow_up_patterns"] = list(self.follow_up_patterns)
        payload["default_save_path"] = str(self.default_save_path) if self.default_save_path else None
        return json_safe(payload)


@dataclass
class DialogueMessage:
    """Single dialogue message in local context history."""

    role: DialogueRole
    content: str
    timestamp: str = field(default_factory=lambda: utc_timestamp())
    message_id: str = field(default_factory=lambda: generate_language_id("dlg_msg"))
    frame: Optional[LinguisticFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_user(self) -> bool:
        return self.role == DialogueRole.USER

    @property
    def is_agent(self) -> bool:
        return self.role in {DialogueRole.AGENT, DialogueRole.ASSISTANT}

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(
            {
                "role": self.role.value,
                "content": self.content,
                "timestamp": self.timestamp,
                "message_id": self.message_id,
                "frame": frame_to_dict(self.frame),
                "metadata": json_safe(self.metadata),
            },
            drop_empty=True,
        )

    def to_history_dict(self) -> Dict[str, str]:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass
class DialogueTurn:
    """User/agent turn pair with optional semantic and diagnostics context."""

    turn_id: str
    user_message: DialogueMessage
    agent_message: Optional[DialogueMessage] = None
    frame: Optional[LinguisticFrame] = None
    grammar_issues: Tuple[Any, ...] = ()
    context_snapshot_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.agent_message is not None

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(
            {
                "turn_id": self.turn_id,
                "user_message": self.user_message.to_dict(),
                "agent_message": self.agent_message.to_dict() if self.agent_message else None,
                "frame": frame_to_dict(self.frame),
                "grammar_issues": json_safe(self.grammar_issues),
                "context_snapshot_id": self.context_snapshot_id,
                "metadata": json_safe(self.metadata),
                "complete": self.complete,
            },
            drop_empty=True,
        )


@dataclass
class SlotValue:
    """Tracked slot value with provenance."""

    name: str
    value: Any
    updated_at: str = field(default_factory=lambda: utc_timestamp())
    source: Optional[str] = None
    confidence: float = 1.0
    frame_intent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(
            {
                "name": self.name,
                "value": json_safe(self.value),
                "updated_at": self.updated_at,
                "source": self.source,
                "confidence": self.confidence,
                "frame_intent": self.frame_intent,
                "metadata": json_safe(self.metadata),
            },
            drop_empty=True,
        )


@dataclass
class IntentTrace:
    """Intent history item."""

    name: str
    confidence: float
    turn: int
    timestamp: str = field(default_factory=lambda: utc_timestamp())
    frame: Optional[LinguisticFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(
            {
                "name": self.name,
                "confidence": self.confidence,
                "turn": self.turn,
                "timestamp": self.timestamp,
                "frame": frame_to_dict(self.frame),
                "metadata": json_safe(self.metadata),
            },
            drop_empty=True,
        )


@dataclass
class UnresolvedIssueRecord:
    """Pending dialogue issue that may require clarification or slot filling."""

    description: str
    slot: Optional[str] = None
    turn_number: int = 0
    created_at: str = field(default_factory=lambda: utc_timestamp())
    severity: str = "warning"
    attempts: int = 1
    frame: Optional[LinguisticFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(
            {
                "description": self.description,
                "slot": self.slot,
                "turn_number": self.turn_number,
                "created_at": self.created_at,
                "severity": self.severity,
                "attempts": self.attempts,
                "frame": frame_to_dict(self.frame),
                "metadata": json_safe(self.metadata),
            },
            drop_empty=True,
        )


@dataclass(frozen=True)
class DialogueContextStats:
    """Operational snapshot for DialogueContext."""

    context_id: str
    session_id: Optional[str]
    message_count: int
    turn_count: int
    slot_count: int
    unresolved_count: int
    intent_count: int
    summary_length: int
    diagnostics_count: int
    last_interaction_seconds: float
    memory_enabled: bool
    cache_enabled: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DialogueContextSnapshot:
    """Serializable dialogue context state."""

    context_id: str
    created_at: str
    summary: str
    history: Tuple[DialogueMessage, ...]
    turns: Tuple[DialogueTurn, ...]
    slots: Dict[str, SlotValue]
    unresolved_issues: Tuple[UnresolvedIssueRecord, ...]
    intent_history: Tuple[IntentTrace, ...]
    environment_state: Dict[str, Any]
    diagnostics: Tuple[LanguageIssue, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_id": self.context_id,
            "created_at": self.created_at,
            "summary": self.summary,
            "history": [message.to_dict() for message in self.history],
            "turns": [turn.to_dict() for turn in self.turns],
            "slots": {key: value.to_dict() for key, value in self.slots.items()},
            "unresolved_issues": [issue.to_dict() for issue in self.unresolved_issues],
            "intent_history": [intent.to_dict() for intent in self.intent_history],
            "environment_state": json_safe(self.environment_state),
            "diagnostics": [issue.to_dict() for issue in self.diagnostics],
            "metadata": json_safe(self.metadata),
        }


class DialogueContext:
    """Production dialogue-state coordinator for the language-agent pipeline."""

    def __init__(self) -> None:
        self.config = load_global_config()
        self.dialogue_config = get_config_section("dialogue_context") or {}
        self.settings = DialogueContextConfig.from_mapping(self.dialogue_config)

        self.context_id = generate_language_id("dialogue_context")
        self.created_at = utc_timestamp()
        self.diagnostics = LanguageDiagnostics()
        self._events: Deque[Dict[str, Any]] = deque(maxlen=max(self.settings.message_limit, 20))
        self._compiled_follow_up_patterns: Tuple[re.Pattern[str], ...] = ()
        self._similarity_fn: Optional[SimilarityFn] = None
        self.summarizer_fn: Optional[SummarizerFn] = None
        self._last_save_at = 0.0

        self.cache: Optional[LanguageCache] = LanguageCache() if self.settings.enable_cache else None
        self.memory: Optional[LanguageMemory] = LanguageMemory() if self.settings.enable_memory else None

        self.history: List[Dict[str, Any]] = []
        self._messages: List[DialogueMessage] = []
        self._turns: List[DialogueTurn] = []
        self.slot_records: Dict[str, SlotValue] = {}
        self.slot_values: Dict[str, Any] = {}
        self.unresolved_issue_records: List[UnresolvedIssueRecord] = []
        self.unresolved_issues: List[Dict[str, Any]] = []
        self.intent_records: List[IntentTrace] = []
        self.intent_history: List[Dict[str, Any]] = []
        self.summary = self.settings.initial_summary or self.settings.default_initial_summary or ""
        self.environment_state = self._initial_environment_state()

        self._load_follow_up_patterns()
        self._initialize_history()
        self._record_event("init", settings=self.settings.to_dict())

        logger.info(
            "DialogueContext initialized. memory_limit=%s summarization=%s memory=%s",
            self.settings.memory_limit,
            self.settings.enable_summarization,
            self.settings.enable_memory,
        )
        printer.status("INIT", "Dialogue Context ready", "success")

    # ------------------------------------------------------------------
    # Setup and normalization
    # ------------------------------------------------------------------
    def _initial_environment_state(self) -> Dict[str, Any]:
        env = merge_mappings(
            self.settings.default_initial_environment_state,
            self.settings.initial_environment_state,
        )
        env.setdefault("session_id", None)
        env.setdefault("user_preferences", {})
        env.setdefault("last_intent", None)
        env.setdefault("conversation_phase", ConversationPhase.OPENING.value)
        env.setdefault("current_topic", "general")
        env.setdefault("modality", "text")
        return env

    def _initialize_history(self) -> None:
        raw_items = list(self.settings.initial_history or self.settings.default_initial_history)
        for item in raw_items:
            role, content, metadata = self._parse_initial_history_item(item)
            if content:
                self.add_message(role=role, content=content, metadata=metadata, persist=False, summarize=False)

    def _parse_initial_history_item(self, item: Any) -> Tuple[str, str, Dict[str, Any]]:
        if isinstance(item, Mapping):
            role = ensure_text(item.get("role", "system"))
            content = ensure_text(item.get("content", ""))
            metadata = ensure_mapping(item.get("metadata", {}), field_name="initial_history.metadata", allow_none=True)
            return role, content, metadata
        text = ensure_text(item)
        if ":" in text:
            role, content = text.split(":", 1)
            return role.strip(), content.strip(), {"source": "initial_history"}
        return "system", text, {"source": "initial_history", "unparsed": True}

    def _normalize_role(self, role: Any) -> DialogueRole:
        raw = normalize_identifier_component(role, default="unknown", lowercase=True)
        alias_map = {
            "bot": "agent",
            "ai": "agent",
            "assistant": "assistant",
            "agent": "agent",
            "system": "system",
            "user": "user",
            "human": "user",
            "tool": "tool",
        }
        alias_map.update({normalize_identifier_component(k, default="unknown", lowercase=True): normalize_identifier_component(v, default="unknown", lowercase=True) for k, v in self.settings.role_aliases.items()})
        normalized = alias_map.get(raw, raw)
        try:
            return DialogueRole(normalized)
        except ValueError:
            issue = ContextIssue(
                code=LanguageErrorCode.CONTEXT_ROLE_MISMATCH,
                message="Unknown dialogue role; using 'unknown'.",
                severity=Severity.WARNING,
                module="DialogueContext",
                recoverable=True,
                details={"role": role},
            )
            self._add_issue(issue)
            return DialogueRole.UNKNOWN

    def _memory_role(self, role: DialogueRole) -> MemoryRole:
        if role == DialogueRole.USER:
            return MemoryRole.USER
        if role == DialogueRole.AGENT:
            return MemoryRole.AGENT
        if role == DialogueRole.ASSISTANT:
            return MemoryRole.ASSISTANT
        if role == DialogueRole.SYSTEM:
            return MemoryRole.SYSTEM
        if role == DialogueRole.TOOL:
            return MemoryRole.TOOL
        return MemoryRole.UNKNOWN

    def _load_follow_up_patterns(self) -> None:
        patterns: List[str] = list(self.settings.follow_up_patterns)
        if self.settings.follow_up_patterns_path and self.settings.follow_up_patterns_path.exists():
            data = load_json_file(self.settings.follow_up_patterns_path)
            if isinstance(data, Mapping):
                patterns.extend(ensure_text(item) for item in ensure_list(data.get("follow_up_patterns", [])))
            elif isinstance(data, list):
                patterns.extend(ensure_text(item) for item in data)
        if not patterns:
            patterns = [
                r"^(and|also|what about|how about|then|so|but)\b",
                r"\b(that|this|those|these|it|they|them)\b.*\?*$",
                r"^(why|how|when|where)\s+(is|are|did|does|can|would|should)\s+(that|it|they)\b",
                r"^(yes|no|sure|okay|ok|right|exactly)\b",
            ]
        compiled: List[re.Pattern[str]] = []
        for pattern in patterns:
            text = ensure_text(pattern)
            if not text:
                continue
            compiled.append(re.compile(text, re.IGNORECASE))
        self._compiled_follow_up_patterns = tuple(compiled)

    # ------------------------------------------------------------------
    # Diagnostics and events
    # ------------------------------------------------------------------
    def _add_issue(self, issue: Union[LanguageIssue, LanguageError]) -> None:
        self.diagnostics.add(issue)
        logger.warning(issue.to_json() if isinstance(issue, LanguageIssue) else issue.to_json())

    def _record_event(self, event_type: str, **payload: Any) -> None:
        event = {
            "timestamp": utc_timestamp(),
            "event_type": event_type,
            "payload": json_safe(payload),
        }
        self._events.append(event)

    # ------------------------------------------------------------------
    # Core message and turn API
    # ------------------------------------------------------------------
    def add_turn(
        self,
        user_input: str,
        agent_response: str,
        *,
        frame: Optional[LinguisticFrame] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> DialogueTurn:
        """Add a user/agent turn pair and return the typed turn record."""
        self.add_message("user", user_input, frame=frame, metadata=metadata, summarize=False)
        user_message = self._messages[-1]
        self.add_message("agent", agent_response, metadata=metadata, summarize=False)
        agent_message = self._messages[-1]
        turn = DialogueTurn(
            turn_id=generate_language_id("dlg_turn"),
            user_message=user_message,
            agent_message=agent_message,
            frame=frame,
            metadata=dict(metadata or {}),
        )
        self._turns.append(turn)
        self._record_event("add_turn", turn=turn.to_dict())
        self._maybe_summarize()
        self._autosave_if_due()
        return turn

    def add_message(
        self,
        role: str,
        content: str,
        *,
        frame: Optional[LinguisticFrame] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        persist: bool = True,
        summarize: bool = True,
    ) -> Dict[str, Any]:
        """Add a single message to local context, memory, and history views."""
        text = ensure_text(content)
        if not text:
            raise ContextError(
                ContextIssue(
                    code=LanguageErrorCode.CONTEXT_STATE_SERIALIZATION_FAILED,
                    message="Dialogue message content cannot be empty.",
                    severity=Severity.ERROR,
                    module="DialogueContext",
                    recoverable=True,
                    details={"role": role},
                )
            )

        normalized_role = self._normalize_role(role)
        message = DialogueMessage(
            role=normalized_role,
            content=text,
            frame=frame,
            metadata=dict(metadata or {}),
        )
        self._messages.append(message)
        self.history.append(message.to_history_dict())
        self._trim_history()

        if normalized_role == DialogueRole.USER:
            self.environment_state["last_user_message"] = text
            self.environment_state["conversation_phase"] = ConversationPhase.ACTIVE.value
        elif normalized_role in {DialogueRole.AGENT, DialogueRole.ASSISTANT}:
            self.environment_state["last_agent_message"] = text

        if persist and self.memory is not None:
            self.memory.remember_turn(
                role=self._memory_role(normalized_role),
                content=text,
                frame=frame,
                scope=MemoryScope.SESSION,
                source="DialogueContext.add_message",
                metadata={"message_id": message.message_id, **dict(metadata or {})},
            )

        self._record_event("add_message", message=message.to_dict())
        if summarize:
            self._maybe_summarize()
        self._autosave_if_due()
        return message.to_history_dict()

    def record_user_turn(
        self,
        user_input: str,
        *,
        frame: Optional[LinguisticFrame] = None,
        grammar_result: Optional[Any] = None,
        nlu_result: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Pipeline-facing method called after NLU and before NLG."""
        details = dict(metadata or {})
        if grammar_result is not None:
            details["grammar_result"] = self._grammar_summary(grammar_result)
        if nlu_result is not None:
            details["nlu_result"] = json_safe(nlu_result)
        message = self.add_message("user", user_input, frame=frame, metadata=details, summarize=False)
        if frame is not None:
            self.register_intent(frame.intent, frame.confidence, frame=frame)
            if self.settings.auto_register_entities_as_slots and frame.entities:
                self.update_slots_from_entities(frame.entities, frame=frame)
        return message

    def record_agent_turn(
        self,
        agent_response: str,
        *,
        frame: Optional[LinguisticFrame] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Pipeline-facing method called after NLG returns a final response."""
        message = self.add_message("agent", agent_response, frame=frame, metadata=metadata, summarize=True)
        self._pair_recent_turn(frame=frame, metadata=metadata)
        return message

    def record_pipeline_turn(self, user_input: str, agent_response: str, *, frame: Optional[LinguisticFrame] = None,
                             grammar_result: Optional[Any] = None, metadata: Optional[Mapping[str, Any]] = None) -> DialogueTurn:
        self.record_user_turn(user_input, frame=frame, grammar_result=grammar_result, metadata=metadata)
        self.record_agent_turn(agent_response, frame=frame, metadata=metadata)
        return self._turns[-1]

    def _pair_recent_turn(self, *, frame: Optional[LinguisticFrame], metadata: Optional[Mapping[str, Any]]) -> None:
        if len(self._messages) < 2:
            return
        user_message = None
        agent_message = self._messages[-1]
        for candidate in reversed(self._messages[:-1]):
            if candidate.role == DialogueRole.USER:
                user_message = candidate
                break
        if user_message is None or not agent_message.is_agent:
            return
        if any(turn.user_message.message_id == user_message.message_id and turn.agent_message and turn.agent_message.message_id == agent_message.message_id for turn in self._turns):
            return
        self._turns.append(
            DialogueTurn(
                turn_id=generate_language_id("dlg_turn"),
                user_message=user_message,
                agent_message=agent_message,
                frame=frame,
                metadata=dict(metadata or {}),
            )
        )

    def _message_from_dict(self, payload: Mapping[str, Any]) -> DialogueMessage:
        return DialogueMessage(
            role=self._normalize_role(payload.get("role", "unknown")),
            content=ensure_text(payload.get("content", "")),
            timestamp=ensure_text(payload.get("timestamp", utc_timestamp())),
            metadata=ensure_mapping(payload.get("metadata", {}), field_name="message.metadata", allow_none=True),
        )

    def _trim_history(self) -> None:
        overflow = len(self.history) - self.settings.message_limit
        if overflow <= 0:
            return
        self.history = self.history[overflow:]
        self._messages = self._messages[overflow:]

    # ------------------------------------------------------------------
    # Summaries and NLG context
    # ------------------------------------------------------------------
    def register_summarizer(self, summarizer_fn: SummarizerFn) -> None:
        if not callable(summarizer_fn):
            raise PipelineContractError("summarizer_fn must be callable", expected="callable", received=type(summarizer_fn).__name__)
        self.summarizer_fn = summarizer_fn
        self._record_event("register_summarizer")

    def _maybe_summarize(self) -> None:
        user_messages = [msg for msg in self.history if msg.get("role") == DialogueRole.USER.value]
        if not self.settings.enable_summarization or len(user_messages) <= self.settings.memory_limit:
            return
        self._summarize()

    def _summarize(self) -> None:
        retain_last = self.settings.retain_last_messages
        messages_to_summarize = self.history[:-retain_last] if retain_last else list(self.history)
        messages_to_keep = self.history[-retain_last:] if retain_last else []
        if not messages_to_summarize:
            return
        try:
            if self.summarizer_fn:
                new_summary = self.summarizer_fn(messages_to_summarize, self.summary)
            else:
                new_summary = self._fallback_summary(messages_to_summarize, self.summary)
            self.summary = compact_text(new_summary, max_length=self.settings.max_summary_length)
            self.history = messages_to_keep
            keep_ids = {msg.get("message_id") for msg in messages_to_keep if msg.get("message_id")}
            if keep_ids:
                self._messages = [message for message in self._messages if message.message_id in keep_ids]
            else:
                self._messages = self._messages[-retain_last:] if retain_last else []
            if self.memory is not None:
                self.memory.remember_summary(
                    key="dialogue_summary",
                    summary=self.summary,
                    scope=MemoryScope.SESSION,
                    source="DialogueContext._summarize",
                    tags=("summary", "dialogue"),
                )
            if self.cache is not None:
                session_id = ensure_text(self.environment_state.get("session_id") or self.context_id)
                self.cache.set_summary(session_id, self.summary, metadata={"source": "DialogueContext"})
            self._record_event("summarize", summary_preview=compact_text(self.summary, max_length=160))
        except Exception as exc:
            issue = issue_from_exception(
                exc,
                code=LanguageErrorCode.CONTEXT_SUMMARY_FAILED,
                stage="context",
                module="DialogueContext",
                severity=Severity.ERROR,
            )
            self._add_issue(issue)
            raise ContextError(issue, cause=exc) from exc

    def _fallback_summary(self, messages: Sequence[Mapping[str, Any]], previous_summary: Optional[str]) -> str:
        lines = []
        if previous_summary:
            lines.append(ensure_text(previous_summary))
        user_mentions = [ensure_text(msg.get("content", "")) for msg in messages if msg.get("role") == DialogueRole.USER.value]
        agent_mentions = [ensure_text(msg.get("content", "")) for msg in messages if msg.get("role") in {DialogueRole.AGENT.value, DialogueRole.ASSISTANT.value}]
        if user_mentions:
            lines.append("User discussed: " + "; ".join(compact_text(text, max_length=120) for text in user_mentions[-3:]))
        if agent_mentions:
            lines.append("Agent responded about: " + "; ".join(compact_text(text, max_length=120) for text in agent_mentions[-2:]))
        if self.intent_history:
            recent_intents = [
                str(item.get("name"))
                for item in self.intent_history[-3:]
                if item.get("name") is not None
            ]
            if recent_intents:
                lines.append("Recent intents: " + ", ".join(recent_intents))
        if self.slot_values:
            lines.append("Known slots: " + ", ".join(f"{key}={compact_text(value, max_length=60)}" for key, value in list(self.slot_values.items())[-5:]))
        return " ".join(line for line in lines if line).strip() or self.settings.default_initial_summary

    def get_summary(self) -> Optional[str]:
        return self.summary

    def get_context_for_prompt(
        self,
        include_summary: Optional[bool] = None,
        include_history: Optional[bool] = None,
        history_messages_window: Optional[int] = None,
    ) -> str:
        """Return a prompt-ready context string."""
        return self.get_relevant_context(
            history_window=history_messages_window if history_messages_window is not None else self.settings.default_history_window,
            include_summary=self.settings.include_summary if include_summary is None else include_summary,
            include_history=self.settings.include_history if include_history is None else include_history,
            include_entities=True,
        )

    def get_relevant_context(
        self,
        *,
        history_window: Optional[int] = None,
        include_summary: Optional[bool] = None,
        include_history: Optional[bool] = None,
        include_entities: bool = True,
        sentiment_threshold: float = 0.0,
        query: Optional[str] = None,
        max_chars: Optional[int] = None,
        include_memory: bool = True,
    ) -> str:
        """Return compact context for NLGEngine._build_neural_prompt."""
        del sentiment_threshold
        use_summary = self.settings.include_summary if include_summary is None else include_summary
        use_history = self.settings.include_history if include_history is None else include_history
        limit = coerce_int(history_window if history_window is not None else self.settings.default_history_window, default=self.settings.default_history_window, minimum=1)
        max_length = coerce_int(max_chars if max_chars is not None else self.settings.max_context_chars, default=self.settings.max_context_chars, minimum=256)

        parts: List[str] = []
        if use_summary and self.summary:
            parts.append(f"[Summary]\n{self.summary}")
        if use_history and self.history:
            formatted = self._format_history(self.get_history_messages(window=limit))
            if formatted:
                parts.append(f"[Recent History]\n{formatted}")
        if self.settings.include_slots and self.slot_values:
            parts.append("[Slots]\n" + "\n".join(f"- {key}: {compact_text(value, max_length=120)}" for key, value in self.slot_values.items()))
        if self.settings.include_unresolved and self.unresolved_issues:
            parts.append("[Pending Issues]\n" + "\n".join(f"- {issue.get('description')}" + (f" ({issue.get('slot')})" if issue.get("slot") else "") for issue in self.unresolved_issues))
        if self.settings.include_intents and self.intent_history:
            recent_intents = self.intent_history[-5:]
            parts.append("[Recent Intents]\n" + "\n".join(f"- {item.get('name')} ({coerce_float(item.get('confidence'), default=0.0):.2f})" for item in recent_intents))
        if self.settings.include_preferences:
            prefs = self.environment_state.get("user_preferences") or {}
            if prefs:
                parts.append("[User Preferences]\n" + "\n".join(f"- {key}: {compact_text(value, max_length=120)}" for key, value in prefs.items()))
        if include_entities:
            entity_lines = self._entity_context_lines()
            if entity_lines:
                parts.append("[Recent Entities]\n" + "\n".join(entity_lines))
        if include_memory and self.memory is not None and query:
            memory_context = self.memory.recall_text(MemoryQuery(text=query, top_k=3, min_score=0.15), include_scores=True)
            if memory_context:
                parts.append("[Relevant Memory]\n" + memory_context)
        if self.settings.include_environment:
            parts.append("[Environment]\n" + stable_json_dumps(self.environment_state, indent=2))

        return compact_text("\n\n".join(parts), max_length=max_length)

    def build_nlg_context(self, frame: Optional[LinguisticFrame] = None, *, grammar_result: Optional[Any] = None,
                          nlu_result: Optional[Any] = None, query: Optional[str] = None) -> Dict[str, Any]:
        """Build structured context that an agent can pass to NLG or telemetry."""
        context_text = self.get_relevant_context(query=query or (frame.intent if frame else None))
        return prune_none(
            {
                "context_id": self.context_id,
                "session_id": self.environment_state.get("session_id"),
                "context_text": context_text,
                "summary": self.summary,
                "history": self.get_history_messages(window=self.settings.default_history_window),
                "slots": self.get_slots(),
                "missing_slots": self.get_missing_slots(),
                "unresolved_issues": list(self.unresolved_issues),
                "last_intent": self.environment_state.get("last_intent"),
                "current_topic": self.environment_state.get("current_topic"),
                "frame": frame_to_dict(frame),
                "grammar": self._grammar_summary(grammar_result) if grammar_result is not None else None,
                "nlu": json_safe(nlu_result) if nlu_result is not None else None,
                "stats": self.stats().to_dict(),
            },
            drop_empty=True,
        )

    def prepare_for_nlg(self, frame: Optional[LinguisticFrame] = None, **kwargs: Any) -> Dict[str, Any]:
        """Alias for pipeline readability before NLGEngine.generate()."""
        return self.build_nlg_context(frame=frame, **kwargs)

    # ------------------------------------------------------------------
    # History views
    # ------------------------------------------------------------------
    def get_history_messages(self, window: Optional[int] = None) -> List[Dict[str, Any]]:
        if window is not None and window > 0:
            return [dict(item) for item in self.history[-int(window):]]
        return [dict(item) for item in self.history]

    def get_history_turns(self, window: Optional[int] = None) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        turns: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        pending_user: Optional[Dict[str, Any]] = None
        for message in self.history:
            role = message.get("role")
            if role == DialogueRole.USER.value:
                pending_user = message
            elif role in {DialogueRole.AGENT.value, DialogueRole.ASSISTANT.value} and pending_user is not None:
                turns.append((pending_user, message))
                pending_user = None
        if window is not None and window > 0:
            return turns[-int(window):]
        return turns

    def _format_history(self, messages: Sequence[Mapping[str, Any]]) -> str:
        lines = []
        for message in messages:
            role = ensure_text(message.get("role", "unknown")).capitalize()
            if role == "Agent":
                role = "Assistant"
            content = compact_text(message.get("content", ""), max_length=self.settings.text_preview_length)
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _entity_context_lines(self) -> List[str]:
        entities: Counter[str] = Counter()
        for trace in self.intent_records[-10:]:
            if trace.frame and trace.frame.entities:
                normalized = normalize_entities(trace.frame.entities)
                for entity in normalized:
                    label = entity.label
                    text = ensure_text(entity.normalized or entity.text or entity.value)
                    if text:
                        entities[f"{label}: {text}"] += 1
        return [f"- {item}" for item, _count in entities.most_common(8)]

    # ------------------------------------------------------------------
    # Environment, slots, unresolved issues, intents
    # ------------------------------------------------------------------
    def update_environment_state(self, key: str, value: Any) -> None:
        name = normalize_identifier_component(key, default="state", lowercase=True)
        self.environment_state[name] = json_safe(value)
        self._record_event("update_environment_state", key=name, value=value)
        self._autosave_if_due()

    def get_environment_state(self, key: Optional[str] = None) -> Any:
        if key:
            return self.environment_state.get(normalize_identifier_component(key, default=ensure_text(key), lowercase=True))
        return dict(self.environment_state)

    def update_slot(self, slot_name: str, value: Any, *, confidence: float = 1.0,
                    source: Optional[str] = None, frame: Optional[LinguisticFrame] = None,
                    metadata: Optional[Mapping[str, Any]] = None) -> SlotValue:
        slot = normalize_identifier_component(slot_name, default="slot", lowercase=True)
        if not self._validate_slot(slot, value):
            issue = ContextIssue(
                code=LanguageErrorCode.CONTEXT_SLOT_MISSING,
                message="Slot value failed configured validation.",
                severity=Severity.ERROR if self.settings.strict_slots else Severity.WARNING,
                module="DialogueContext",
                recoverable=not self.settings.strict_slots,
                frame=frame,
                details={"slot": slot, "value": json_safe(value), "rules": self.settings.slot_validation_rules.get(slot)},
            )
            self._add_issue(issue)
            if self.settings.strict_slots:
                raise ContextError(issue)
        record = SlotValue(
            name=slot,
            value=value,
            source=source,
            confidence=clamp_float(confidence, 0.0, 1.0),
            frame_intent=frame.intent if frame else None,
            metadata=dict(metadata or {}),
        )
        self.slot_records[slot] = record
        self.slot_values[slot] = value
        self.unresolved_issue_records = [issue for issue in self.unresolved_issue_records if issue.slot != slot]
        self.unresolved_issues = [issue.to_dict() for issue in self.unresolved_issue_records]
        if self.memory is not None:
            self.memory.remember_slot(
                key=slot,
                value=value,
                text=f"{slot}: {value}",
                frame=frame,
                confidence=record.confidence,
                source=source or "DialogueContext.update_slot",
            )
        self._record_event("update_slot", slot=record.to_dict())
        self._autosave_if_due()
        return record

    def update_slots_from_entities(self, entities: Any, *, frame: Optional[LinguisticFrame] = None) -> Dict[str, Any]:
        normalized = normalize_entities(entities)
        updated: Dict[str, Any] = {}
        for entity in normalized:
            key = normalize_identifier_component(entity.label, default="entity", lowercase=True)
            value = entity.value if entity.value is not None else entity.normalized or entity.text
            if value is not None and ensure_text(value).strip():
                self.update_slot(key, value, confidence=entity.confidence or 1.0, source="nlu_entity", frame=frame, metadata={"entity": entity.to_dict()})
                updated[key] = value
        return updated

    def _validate_slot(self, slot: str, value: Any) -> bool:
        rules = self.settings.slot_validation_rules.get(slot)
        if not isinstance(rules, Mapping):
            return True
        expected_type = ensure_text(rules.get("type", "")).lower()
        allowed_values = ensure_list(rules.get("allowed_values", []))
        if allowed_values and value not in allowed_values:
            return False
        if not expected_type:
            return True
        if expected_type == "string":
            return isinstance(value, str) and bool(value.strip())
        if expected_type in {"number", "float"}:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected_type in {"integer", "int"}:
            return isinstance(value, int) and not isinstance(value, bool)
        if expected_type in {"boolean", "bool"}:
            return isinstance(value, bool)
        if expected_type == "date_range":
            if isinstance(value, Mapping):
                return bool(value.get("start") and value.get("end"))
            if isinstance(value, (list, tuple)):
                return len(value) >= 2
            return bool(ensure_text(value).strip())
        return True

    def add_unresolved(self, issue: str, slot: Optional[str] = None, *, severity: str = "warning",
                       frame: Optional[LinguisticFrame] = None, metadata: Optional[Mapping[str, Any]] = None) -> UnresolvedIssueRecord:
        description = ensure_text(issue).strip()
        if not description:
            raise ContextError(
                ContextIssue(
                    code=LanguageErrorCode.CONTEXT_FOLLOWUP_AMBIGUOUS,
                    message="Unresolved issue description cannot be empty.",
                    severity=Severity.ERROR,
                    module="DialogueContext",
                    recoverable=True,
                )
            )
        existing = next((item for item in self.unresolved_issue_records if item.description == description and item.slot == slot), None)
        if existing:
            existing.attempts += 1
            existing.metadata.update(dict(metadata or {}))
            self.unresolved_issues = [record.to_dict() for record in self.unresolved_issue_records]
            return existing
        record = UnresolvedIssueRecord(
            description=description,
            slot=normalize_identifier_component(slot, default="slot", lowercase=True) if slot else None,
            turn_number=len(self.history),
            severity=severity,
            frame=frame,
            metadata=dict(metadata or {}),
        )
        self.unresolved_issue_records.append(record)
        self.unresolved_issues = [item.to_dict() for item in self.unresolved_issue_records]
        if self.memory is not None:
            self.memory.remember_issue(
                ContextIssue(
                    code=LanguageErrorCode.CONTEXT_FOLLOWUP_AMBIGUOUS,
                    message=description,
                    severity=Severity.WARNING,
                    module="DialogueContext",
                    recoverable=True,
                    frame=frame,
                    details=record.to_dict(),
                ),
                source="DialogueContext.add_unresolved",
            )
        self.environment_state["conversation_phase"] = ConversationPhase.CLARIFYING.value
        self._record_event("add_unresolved", issue=record.to_dict())
        self._autosave_if_due()
        return record

    def resolve_unresolved(self, description: Optional[str] = None, *, slot: Optional[str] = None) -> int:
        before = len(self.unresolved_issue_records)
        slot_norm = normalize_identifier_component(slot, default="slot", lowercase=True) if slot else None
        self.unresolved_issue_records = [
            item for item in self.unresolved_issue_records
            if not ((description is None or item.description == description) and (slot_norm is None or item.slot == slot_norm))
        ]
        self.unresolved_issues = [item.to_dict() for item in self.unresolved_issue_records]
        resolved = before - len(self.unresolved_issue_records)
        if resolved:
            self._record_event("resolve_unresolved", description=description, slot=slot_norm, resolved=resolved)
        return resolved

    def register_intent(
        self,
        intent: str,
        confidence: float,
        *,
        frame: Optional[LinguisticFrame] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> IntentTrace:
        normalized = normalize_intent(intent)
        trace = IntentTrace(
            name=normalized,
            confidence=clamp_float(confidence, 0.0, 1.0),
            turn=len(self.history),
            frame=frame,
            metadata=dict(metadata or {}),
        )
        self.intent_records.append(trace)
        self.intent_history.append(trace.to_dict())
        self.environment_state["last_intent"] = normalized
        if frame is not None:
            self.environment_state["last_frame"] = frame_to_dict(frame)
        if self.memory is not None:
            self.memory.remember_intent(normalized, confidence=trace.confidence, frame=frame, source="DialogueContext.register_intent")
        self._record_event("register_intent", intent=trace.to_dict())
        self._autosave_if_due()
        return trace

    @property
    def required_slots_filled(self) -> bool:
        return all(slot in self.slot_values for slot in self.settings.required_slots)

    def get_missing_slots(self) -> List[str]:
        return [slot for slot in self.settings.required_slots if slot not in self.slot_values]

    @property
    def required_slots(self) -> Tuple[str, ...]:
        return self.settings.required_slots

    def get_slot(self, slot_name: str, default: Any = None) -> Any:
        return self.slot_values.get(normalize_identifier_component(slot_name, default="slot", lowercase=True), default)

    def get_slots(self) -> Dict[str, Any]:
        return dict(self.slot_values)

    def update_user_profile(self, preferences: Mapping[str, Any]) -> None:
        incoming = ensure_mapping(preferences, field_name="preferences", allow_none=True)
        existing = ensure_mapping(self.environment_state.get("user_preferences", {}), field_name="user_preferences", allow_none=True)
        merged = merge_mappings(existing, incoming)
        self.environment_state["user_preferences"] = merged
        if self.memory is not None:
            for key, value in incoming.items():
                self.memory.remember_preference(key=ensure_text(key), value=value, source="DialogueContext.update_user_profile")
        self._record_event("update_user_profile", preferences=sanitize_for_logging(incoming))
        self._autosave_if_due()

    def get_personalization_context(self) -> str:
        prefs = ensure_mapping(self.environment_state.get("user_preferences", {}), field_name="user_preferences", allow_none=True)
        return "\n".join(f"{key}: {compact_text(value, max_length=120)}" for key, value in prefs.items())

    # ------------------------------------------------------------------
    # Follow-up and topic handling
    # ------------------------------------------------------------------
    def is_follow_up(self, current_utterance: str) -> bool:
        if not self.settings.enable_follow_up_detection:
            return False
        text = ensure_text(current_utterance).strip()
        if not text:
            return False
        for pattern in self._compiled_follow_up_patterns:
            if pattern.search(text):
                self._record_event("follow_up_detected", text_preview=compact_text(text, max_length=120), pattern=pattern.pattern)
                return True
        short_text = len(text.split()) <= 5
        pronounish = bool(re.search(r"\b(it|that|this|those|these|they|them|he|she|there)\b", text, flags=re.IGNORECASE))
        return short_text and pronounish and bool(self.history)

    def register_similarity_function(self, similarity_fn: SimilarityFn) -> None:
        if not callable(similarity_fn):
            raise PipelineContractError("similarity_fn must be callable", expected="callable", received=type(similarity_fn).__name__)
        self._similarity_fn = similarity_fn

    def detect_topic_shift(self, current_topic: str) -> bool:
        if not self.settings.enable_topic_detection:
            return False
        current = normalize_whitespace(current_topic)
        if len(current) < self.settings.topic_min_text_length:
            return False
        recent_user_texts = [
            ensure_text(message.get("content", ""))
            for message in self.history[-self.settings.topic_lookback_window * 2:]
            if message.get("role") == DialogueRole.USER.value
        ]
        if not recent_user_texts:
            return False
        scores = []
        for previous in recent_user_texts:
            score = self._similarity_fn(current, previous) if self._similarity_fn else lexical_overlap(current, previous)
            scores.append(score)
        max_score = max(scores) if scores else 0.0
        shifted = max_score < self.settings.topic_similarity_threshold
        if shifted:
            issue = ContextIssue(
                code=LanguageErrorCode.CONTEXT_TOPIC_SHIFT_AMBIGUOUS,
                message="Potential topic shift detected.",
                severity=Severity.INFO,
                module="DialogueContext",
                recoverable=True,
                details={"current_topic": current, "max_similarity": max_score},
            )
            self.diagnostics.add(issue)
        return shifted

    def update_topic(self, topic: str, *, force: bool = False) -> bool:
        normalized = compact_text(topic, max_length=160)
        if not force and self.detect_topic_shift(normalized):
            self.environment_state["previous_topic"] = self.environment_state.get("current_topic")
        self.environment_state["current_topic"] = normalized or "general"
        self._record_event("update_topic", topic=normalized)
        return True

    # ------------------------------------------------------------------
    # Persistence and state transfer
    # ------------------------------------------------------------------
    def serialize(self) -> Dict[str, Any]:
        return self.snapshot().to_dict()

    def deserialize(self, data: Mapping[str, Any]) -> None:
        payload = ensure_mapping(data, field_name="dialogue_context_state", allow_none=False)
        self.context_id = ensure_text(payload.get("context_id", self.context_id))
        self.summary = ensure_text(payload.get("summary", self.summary))
        self.environment_state = ensure_mapping(payload.get("environment_state", self.environment_state), field_name="environment_state", allow_none=True)
        self.history = []
        self._messages = []
        for raw in ensure_list(payload.get("history", [])):
            if isinstance(raw, Mapping):
                message = DialogueMessage(
                    role=self._normalize_role(raw.get("role", "unknown")),
                    content=ensure_text(raw.get("content", "")),
                    timestamp=ensure_text(raw.get("timestamp", utc_timestamp())),
                    message_id=ensure_text(raw.get("message_id", generate_language_id("dlg_msg"))),
                    metadata=ensure_mapping(raw.get("metadata", {}), field_name="message.metadata", allow_none=True),
                )
                self._messages.append(message)
                self.history.append(message.to_history_dict())
        self.slot_records = {}
        self.slot_values = {}
        for key, raw_slot in ensure_mapping(payload.get("slots", {}), field_name="slots", allow_none=True).items():
            if isinstance(raw_slot, Mapping):
                record = SlotValue(
                    name=ensure_text(raw_slot.get("name", key)),
                    value=raw_slot.get("value"),
                    updated_at=ensure_text(raw_slot.get("updated_at", utc_timestamp())),
                    source=raw_slot.get("source"),
                    confidence=clamp_float(raw_slot.get("confidence", 1.0)),
                    frame_intent=raw_slot.get("frame_intent"),
                    metadata=ensure_mapping(raw_slot.get("metadata", {}), field_name="slot.metadata", allow_none=True),
                )
                self.slot_records[record.name] = record
                self.slot_values[record.name] = record.value
        self.unresolved_issue_records = []
        for raw_issue in ensure_list(payload.get("unresolved_issues", [])):
            if isinstance(raw_issue, Mapping):
                self.unresolved_issue_records.append(
                    UnresolvedIssueRecord(
                        description=ensure_text(raw_issue.get("description", "")),
                        slot=raw_issue.get("slot"),
                        turn_number=coerce_int(raw_issue.get("turn_number", 0), default=0, minimum=0),
                        created_at=ensure_text(raw_issue.get("created_at", utc_timestamp())),
                        severity=ensure_text(raw_issue.get("severity", "warning")),
                        attempts=coerce_int(raw_issue.get("attempts", 1), default=1, minimum=1),
                        metadata=ensure_mapping(raw_issue.get("metadata", {}), field_name="issue.metadata", allow_none=True),
                    )
                )
        self.unresolved_issues = [issue.to_dict() for issue in self.unresolved_issue_records]
        self.intent_records = []
        for raw_intent in ensure_list(payload.get("intent_history", [])):
            if isinstance(raw_intent, Mapping):
                self.intent_records.append(
                    IntentTrace(
                        name=normalize_intent(raw_intent.get("name")),
                        confidence=clamp_float(raw_intent.get("confidence", 0.0)),
                        turn=coerce_int(raw_intent.get("turn", 0), default=0, minimum=0),
                        timestamp=ensure_text(raw_intent.get("timestamp", utc_timestamp())),
                        metadata=ensure_mapping(raw_intent.get("metadata", {}), field_name="intent.metadata", allow_none=True),
                    )
                )
        self.intent_history = [intent.to_dict() for intent in self.intent_records]
        self._record_event("deserialize", message_count=len(self.history))

    def save_state(self, file_path: Optional[Union[str, Path]] = None) -> Path:
        path = self._resolve_state_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.serialize()
        if self.settings.redact_persistence:
            payload = sanitize_for_logging(payload)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            save_json_file(tmp_path, payload, pretty=self.settings.persistence_pretty)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        self._last_save_at = epoch_seconds()
        self._record_event("save_state", path=str(path))
        return path

    @classmethod
    def load_state(cls, file_path: Union[str, Path]) -> "DialogueContext":
        path = resolve_path(file_path, must_exist=True, field_name="dialogue_context_state")
        payload = load_json_file(path)
        instance = cls()
        instance.deserialize(payload)
        instance._record_event("load_state", path=str(path))
        return instance

    def _resolve_state_path(self, file_path: Optional[Union[str, Path]]) -> Path:
        if file_path is not None:
            return resolve_path(file_path, field_name="file_path")
        if self.settings.default_save_path is None:
            session_id = normalize_identifier_component(self.environment_state.get("session_id") or self.context_id, default="session", lowercase=True)
            return resolve_path(f"src/agents/language/sessions/{session_id}.json", field_name="dialogue_context.persistence.default_save_path")
        if self.settings.default_save_path.suffix:
            return self.settings.default_save_path
        session_id = normalize_identifier_component(self.environment_state.get("session_id") or self.context_id, default="session", lowercase=True)
        return self.settings.default_save_path / f"{session_id}.json"

    def _autosave_if_due(self) -> None:
        if not self.settings.enable_persistence or self.settings.auto_save_interval <= 0:
            return
        now = epoch_seconds()
        if now - self._last_save_at >= self.settings.auto_save_interval:
            self.save_state()

    # ------------------------------------------------------------------
    # Timeouts, cleanup, stats
    # ------------------------------------------------------------------
    def get_time_since_last_interaction(self) -> float:
        """Return minutes since the last message for backward compatibility."""
        if not self.history:
            return 0.0
        timestamp = ensure_text(self.history[-1].get("timestamp", utc_timestamp()))
        try:
            last_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=timezone.utc)
            return max(0.0, (datetime.now(timezone.utc) - last_time.astimezone(timezone.utc)).total_seconds() / 60.0)
        except ValueError:
            return 0.0

    def is_stale(self) -> bool:
        if self.settings.session_timeout_seconds <= 0:
            return False
        return self.get_time_since_last_interaction() * 60.0 > self.settings.session_timeout_seconds

    def clear(self, *, clear_memory: bool = False) -> None:
        self.history = []
        self._messages = []
        self._turns = []
        self.summary = self.settings.default_initial_summary or ""
        self.environment_state = self._initial_environment_state()
        self.slot_records = {}
        self.slot_values = {}
        self.unresolved_issue_records = []
        self.unresolved_issues = []
        self.intent_records = []
        self.intent_history = []
        self.diagnostics = LanguageDiagnostics()
        if clear_memory and self.memory is not None:
            self.memory.clear(clear_disk=False)
        self._initialize_history()
        self._record_event("clear", clear_memory=clear_memory)

    def snapshot(self) -> DialogueContextSnapshot:
        return DialogueContextSnapshot(
            context_id=self.context_id,
            created_at=self.created_at,
            summary=self.summary,
            history=tuple(self._messages),
            turns=tuple(self._turns),
            slots=dict(self.slot_records),
            unresolved_issues=tuple(self.unresolved_issue_records),
            intent_history=tuple(self.intent_records),
            environment_state=dict(self.environment_state),
            diagnostics=tuple(self.diagnostics.issues),
            metadata={"settings_version": self.settings.version},
        )

    def stats(self) -> DialogueContextStats:
        return DialogueContextStats(
            context_id=self.context_id,
            session_id=self.environment_state.get("session_id"),
            message_count=len(self.history),
            turn_count=len(self.get_history_turns()),
            slot_count=len(self.slot_values),
            unresolved_count=len(self.unresolved_issues),
            intent_count=len(self.intent_history),
            summary_length=len(self.summary or ""),
            diagnostics_count=len(self.diagnostics.issues),
            last_interaction_seconds=self.get_time_since_last_interaction() * 60.0,
            memory_enabled=self.memory is not None,
            cache_enabled=self.cache is not None,
        )

    def diagnostics_result(self) -> LanguageResult[Dict[str, Any]]:
        return success_result(data=self.stats().to_dict(), metadata={"diagnostics": self.diagnostics.to_list()})

    def recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        count = coerce_int(limit, default=20, minimum=1)
        return list(self._events)[-count:]

    def _grammar_summary(self, grammar_result: Any) -> Dict[str, Any]:
        if grammar_result is None:
            return {}
        if hasattr(grammar_result, "to_dict"):
            return json_safe(grammar_result.to_dict())
        if hasattr(grammar_result, "__dict__"):
            return json_safe(vars(grammar_result))
        return {"value": json_safe(grammar_result)}

    def to_dict(self) -> Dict[str, Any]:
        return self.snapshot().to_dict()

    def __len__(self) -> int:
        return len(self.history)

    def __repr__(self) -> str:
        return (
            f"<DialogueContext context_id='{self.context_id}' "
            f"messages={len(self.history)} slots={len(self.slot_values)} "
            f"unresolved={len(self.unresolved_issues)}>"
        )


if __name__ == "__main__":
    print("\n=== Running Dialogue Context ===\n")
    printer.status("TEST", "Dialogue Context initialized", "info")

    context = DialogueContext()
    context.update_environment_state("session_id", "dialogue-context-test")
    context.update_user_profile({"tone": "technical", "verbosity": "detailed"})

    frame = LinguisticFrame(
        intent="ask_meaning",
        entities={"topic": "life", "domain": "philosophy"},
        sentiment=0.1,
        modality="interrogative",
        confidence=0.92,
        act_type=SpeechActType.DIRECTIVE,
    )

    context.record_user_turn(
        "What is life about?",
        frame=frame,
        metadata={"pipeline_stage": "after_nlu"},
    )
    context.update_slot("topic", "life", confidence=0.95, frame=frame)
    context.register_intent("ask_meaning", 0.92, frame=frame)
    context.add_unresolved("needs_philosophical_scope", slot="scope", frame=frame)

    nlg_context = context.prepare_for_nlg(frame=frame, query="life meaning")
    response = "The meaning of life differs from person to person, but we can discuss it through values, purpose, relationships, or philosophy."
    context.record_agent_turn(response, frame=frame, metadata={"pipeline_stage": "after_nlg"})

    follow_up = context.is_follow_up("What about purpose?")
    topic_shift = context.detect_topic_shift("quantum mechanics")
    prompt_context = context.get_context_for_prompt(history_messages_window=6)
    relevant_context = context.get_relevant_context(query="life purpose", include_entities=True)
    stats = context.stats().to_dict()

    save_path = context.save_state("/tmp/dialogue_context_test_state.json")
    loaded = DialogueContext.load_state(save_path)

    printer.pretty("NLG_CONTEXT", nlg_context, "success")
    printer.pretty("PROMPT_CONTEXT", {"text": prompt_context}, "success")
    printer.pretty("RELEVANT_CONTEXT", {"text": relevant_context}, "success")
    printer.pretty("FOLLOW_UP", {"is_follow_up": follow_up}, "success")
    printer.pretty("TOPIC_SHIFT", {"topic_shift": topic_shift}, "success")
    printer.pretty("HISTORY", context.get_history_messages(), "success")
    printer.pretty("TURNS", context.get_history_turns(), "success")
    printer.pretty("SLOTS", context.get_slots(), "success")
    printer.pretty("MISSING_SLOTS", context.get_missing_slots(), "success")
    printer.pretty("STATS", stats, "success")
    printer.pretty("LOADED", loaded.stats().to_dict(), "success")

    print("\n=== Test ran successfully ===\n")
