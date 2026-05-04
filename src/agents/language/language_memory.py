"""
Production-grade memory module for the language subsystem.

Core Function:
Maintains durable, queryable, language-aware memory for the language agent.
It stores conversational turns, facts, preferences, slots, summaries, entities,
intent traces, unresolved issues, and semantic notes with confidence, salience,
TTL, source metadata, tags, scopes, and optional LinguisticFrame context.

Responsibilities:
- Store short-term and long-term language memories with stable typed records.
- Preserve multi-turn context without forcing dialogue_context.py to own all
  memory policy, persistence, ranking, and retrieval concerns.
- Retrieve relevant memories using recency, salience, confidence, lexical
  overlap, tags, scope, kind, key, entities, and intent-aware scoring.
- Support slots, preferences, facts, summaries, entity traces, intent history,
  user/system/assistant turns, and unresolved language issues.
- Persist memory state atomically using language_config.yaml settings.
- Reuse language_helpers.py for normalization, hashing, JSON safety, IDs,
  path/config helpers, redaction, text validation, scoring primitives, frame
  helpers, and result payloads instead of duplicating helper logic.
- Reuse language_error.py for structured diagnostics and recoverable failures.

Why it matters:
A language agent should not be stateless or reactive-only. A dedicated memory
layer lets orthography, NLP, NLU, dialogue context, and NLG share a stable
record of what has been said, inferred, resolved, corrected, preferred, and
remembered without duplicating storage logic across modules.
"""

from __future__ import annotations

import gzip
import math
import os
import pickle
import tempfile
import time as time_module

from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.linguistic_frame import LinguisticFrame, SpeechActType
from .utils.language_error import *
from .utils.language_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Language Memory")
printer = PrettyPrinter()

E_mem = TypeVar("E_mem", bound=Enum)

MemoryId = str
MemoryKey = str
MemoryTag = str
MemoryNamespace = str
MemoryPredicate = Callable[["MemoryRecord"], bool]


class MemoryKind(str, Enum):
    """Language-memory record categories."""

    TURN = "turn"
    FACT = "fact"
    PREFERENCE = "preference"
    SLOT = "slot"
    SUMMARY = "summary"
    ENTITY = "entity"
    INTENT = "intent"
    ISSUE = "issue"
    TOPIC = "topic"
    NOTE = "note"
    SYSTEM = "system"


class MemoryScope(str, Enum):
    """Scope controls how broadly a memory should be reused."""

    SESSION = "session"
    USER = "user"
    AGENT = "agent"
    GLOBAL = "global"
    TASK = "task"
    EPHEMERAL = "ephemeral"


class MemoryRole(str, Enum):
    """Normalized speaker/source roles for turn memories."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    AGENT = "agent"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MemoryQuery:
    """Structured retrieval query for language memory."""

    text: Optional[str] = None
    key: Optional[str] = None
    kinds: Tuple[MemoryKind, ...] = ()
    scopes: Tuple[MemoryScope, ...] = ()
    tags: Tuple[str, ...] = ()
    entities: Tuple[str, ...] = ()
    intent: Optional[str] = None
    source: Optional[str] = None
    since: Optional[float] = None
    until: Optional[float] = None
    top_k: int = 10
    min_score: float = 0.0
    min_confidence: Optional[float] = None
    min_salience: Optional[float] = None
    include_expired: bool = False
    include_values: bool = True
    metadata_filter: Optional[Callable[[Dict[str, Any]], bool]] = None

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(
            {
                "text": self.text,
                "key": self.key,
                "kinds": [kind.value for kind in self.kinds],
                "scopes": [scope.value for scope in self.scopes],
                "tags": list(self.tags),
                "entities": list(self.entities),
                "intent": self.intent,
                "source": self.source,
                "since": self.since,
                "until": self.until,
                "top_k": self.top_k,
                "min_score": self.min_score,
                "min_confidence": self.min_confidence,
                "min_salience": self.min_salience,
                "include_expired": self.include_expired,
                "include_values": self.include_values,
                "has_metadata_filter": self.metadata_filter is not None,
            },
            drop_empty=True,
        )


@dataclass
class MemoryRecord:
    """Single durable language memory record."""

    memory_id: MemoryId
    kind: MemoryKind
    scope: MemoryScope
    key: MemoryKey
    value: Any
    text: str
    created_at: float
    updated_at: float
    accessed_at: float
    ttl_seconds: Optional[float] = None
    role: MemoryRole = MemoryRole.UNKNOWN
    source: Optional[str] = None
    confidence: float = 1.0
    salience: float = 0.5
    priority: int = 0
    tags: Tuple[str, ...] = ()
    entities: Tuple[str, ...] = ()
    frame: Optional[LinguisticFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    hits: int = 0
    superseded_by: Optional[MemoryId] = None
    deleted: bool = False

    @property
    def age_seconds(self) -> float:
        return max(0.0, time_module.time() - float(self.created_at))

    @property
    def idle_seconds(self) -> float:
        return max(0.0, time_module.time() - float(self.accessed_at))

    @property
    def expires_at(self) -> Optional[float]:
        if self.ttl_seconds is None or self.ttl_seconds <= 0:
            return None
        return float(self.updated_at) + float(self.ttl_seconds)

    @property
    def intent(self) -> Optional[str]:
        return self.frame.intent if self.frame else None

    @property
    def active(self) -> bool:
        return not self.deleted and self.superseded_by is None

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.ttl_seconds is None or self.ttl_seconds <= 0:
            return False
        current = time_module.time() if now is None else float(now)
        return current - float(self.updated_at) > float(self.ttl_seconds)

    def touch(self, *, timestamp: Optional[float] = None) -> None:
        self.accessed_at = time_module.time() if timestamp is None else float(timestamp)
        self.hits += 1

    def refresh(self, *, timestamp: Optional[float] = None) -> None:
        now = time_module.time() if timestamp is None else float(timestamp)
        self.updated_at = now
        self.accessed_at = now

    def decay_salience(self, *, half_life_seconds: Optional[float]) -> float:
        if not half_life_seconds or half_life_seconds <= 0:
            return self.salience
        return clamp_float(self.salience * math.pow(0.5, self.age_seconds / half_life_seconds), 0.0, 1.0)

    def to_dict(self, *, include_value: bool = True) -> Dict[str, Any]:
        payload = {
            "memory_id": self.memory_id,
            "kind": self.kind.value,
            "scope": self.scope.value,
            "key": self.key,
            "value": json_safe(self.value) if include_value else None,
            "text": self.text,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "accessed_at": self.accessed_at,
            "ttl_seconds": self.ttl_seconds,
            "expires_at": self.expires_at,
            "age_seconds": round(self.age_seconds, 6),
            "idle_seconds": round(self.idle_seconds, 6),
            "role": self.role.value,
            "source": self.source,
            "confidence": self.confidence,
            "salience": self.salience,
            "priority": self.priority,
            "tags": list(self.tags),
            "entities": list(self.entities),
            "intent": self.intent,
            "frame": frame_to_dict(self.frame),
            "metadata": json_safe(self.metadata),
            "hits": self.hits,
            "superseded_by": self.superseded_by,
            "deleted": self.deleted,
            "active": self.active,
        }
        return prune_none(payload, drop_empty=True)


@dataclass(frozen=True)
class MemoryMatch:
    """Ranked memory retrieval result."""

    record: MemoryRecord
    score: float
    reasons: Dict[str, float] = field(default_factory=dict)

    def to_dict(self, *, include_value: bool = True) -> Dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "reasons": {key: round(value, 6) for key, value in self.reasons.items()},
            "record": self.record.to_dict(include_value=include_value),
        }


@dataclass(frozen=True)
class MemorySnapshot:
    """Serializable memory state snapshot."""

    records: Tuple[MemoryRecord, ...]
    created_at: str
    stats: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_values: bool = True) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "count": len(self.records),
            "records": [record.to_dict(include_value=include_values) for record in self.records],
            "stats": json_safe(self.stats),
            "metadata": json_safe(self.metadata),
        }


@dataclass
class LanguageMemoryStats:
    """Operational counters for memory observability."""

    started_at: float = field(default_factory=time_module.time)
    writes: int = 0
    reads: int = 0
    hits: int = 0
    misses: int = 0
    updates: int = 0
    deletes: int = 0
    recalls: int = 0
    expirations: int = 0
    evictions: int = 0
    saves: int = 0
    loads: int = 0
    save_failures: int = 0
    load_failures: int = 0

    @property
    def uptime_seconds(self) -> float:
        return max(0.0, time_module.time() - self.started_at)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return 0.0 if total == 0 else self.hits / total

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["uptime_seconds"] = round(self.uptime_seconds, 6)
        payload["hit_rate"] = round(self.hit_rate, 6)
        return payload


@dataclass(frozen=True)
class LanguageMemoryConfig:
    """Runtime configuration loaded from language_config.yaml."""

    version: str = "1.0"
    max_records: int = 5000
    max_turns: int = 500
    max_records_per_scope: int = 2500
    default_scope: MemoryScope = MemoryScope.SESSION
    default_ttl_seconds: Optional[float] = None
    turn_ttl_seconds: Optional[float] = None
    fact_ttl_seconds: Optional[float] = None
    preference_ttl_seconds: Optional[float] = None
    slot_ttl_seconds: Optional[float] = None
    summary_ttl_seconds: Optional[float] = None
    issue_ttl_seconds: Optional[float] = 604800.0
    memory_path: Optional[Path] = None
    metadata_export_path: Optional[Path] = None
    autosave: bool = True
    load_on_init: bool = True
    save_interval_seconds: float = 60.0
    prune_expired_on_save: bool = True
    enable_compression: bool = True
    enable_encryption: bool = False
    serialization_protocol: int = pickle.HIGHEST_PROTOCOL
    strict_persistence: bool = False
    normalize_keys: bool = True
    key_namespace: str = "language_memory"
    max_text_length: int = 4000
    max_tags: int = 32
    max_entities: int = 64
    retain_last_turns: int = 40
    salience_decay_half_life_seconds: Optional[float] = 604800.0
    recall_top_k: int = 10
    recall_min_score: float = 0.05
    recency_weight: float = 0.20
    salience_weight: float = 0.25
    confidence_weight: float = 0.15
    lexical_weight: float = 0.25
    tag_weight: float = 0.10
    kind_weight: float = 0.10
    scope_weight: float = 0.05
    key_weight: float = 0.20
    entity_weight: float = 0.10
    intent_weight: float = 0.10
    allowed_roles: Tuple[str, ...] = ("user", "assistant", "system", "tool", "agent", "unknown")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "LanguageMemoryConfig":
        section = dict(config or {})
        default_scope = _coerce_scope(section.get("default_scope", MemoryScope.SESSION.value))
        memory_path_value = first_non_none(section.get("memory_path"), section.get("path"))
        metadata_export_value = section.get("metadata_export_path")
        return cls(
            version=str(section.get("version", "1.0")),
            max_records=coerce_int(section.get("max_records"), default=5000, minimum=1),
            max_turns=coerce_int(section.get("max_turns"), default=500, minimum=1),
            max_records_per_scope=coerce_int(section.get("max_records_per_scope"), default=2500, minimum=1),
            default_scope=default_scope,
            default_ttl_seconds=_optional_seconds(section.get("default_ttl_seconds"), default=None),
            turn_ttl_seconds=_optional_seconds(section.get("turn_ttl_seconds"), default=None),
            fact_ttl_seconds=_optional_seconds(section.get("fact_ttl_seconds"), default=None),
            preference_ttl_seconds=_optional_seconds(section.get("preference_ttl_seconds"), default=None),
            slot_ttl_seconds=_optional_seconds(section.get("slot_ttl_seconds"), default=None),
            summary_ttl_seconds=_optional_seconds(section.get("summary_ttl_seconds"), default=None),
            issue_ttl_seconds=_optional_seconds(section.get("issue_ttl_seconds"), default=604800.0),
            memory_path=Path(memory_path_value) if memory_path_value else None,
            metadata_export_path=Path(metadata_export_value) if metadata_export_value else None,
            autosave=coerce_bool(section.get("autosave"), default=True),
            load_on_init=coerce_bool(section.get("load_on_init"), default=True),
            save_interval_seconds=coerce_float(section.get("save_interval_seconds"), default=60.0, minimum=0.0),
            prune_expired_on_save=coerce_bool(section.get("prune_expired_on_save"), default=True),
            enable_compression=coerce_bool(section.get("enable_compression"), default=True),
            enable_encryption=coerce_bool(section.get("enable_encryption"), default=False),
            serialization_protocol=coerce_int(section.get("serialization_protocol"), default=pickle.HIGHEST_PROTOCOL, minimum=0, maximum=pickle.HIGHEST_PROTOCOL),
            strict_persistence=coerce_bool(section.get("strict_persistence"), default=False),
            normalize_keys=coerce_bool(section.get("normalize_keys"), default=True),
            key_namespace=normalize_identifier_component(section.get("key_namespace", "language_memory"), default="language_memory"),
            max_text_length=coerce_int(section.get("max_text_length"), default=4000, minimum=64),
            max_tags=coerce_int(section.get("max_tags"), default=32, minimum=1),
            max_entities=coerce_int(section.get("max_entities"), default=64, minimum=1),
            retain_last_turns=coerce_int(section.get("retain_last_turns"), default=40, minimum=1),
            salience_decay_half_life_seconds=_optional_seconds(section.get("salience_decay_half_life_seconds"), default=604800.0),
            recall_top_k=coerce_int(section.get("recall_top_k"), default=10, minimum=1),
            recall_min_score=coerce_probability(section.get("recall_min_score"), default=0.05),
            recency_weight=coerce_float(section.get("recency_weight"), default=0.20, minimum=0.0),
            salience_weight=coerce_float(section.get("salience_weight"), default=0.25, minimum=0.0),
            confidence_weight=coerce_float(section.get("confidence_weight"), default=0.15, minimum=0.0),
            lexical_weight=coerce_float(section.get("lexical_weight"), default=0.25, minimum=0.0),
            tag_weight=coerce_float(section.get("tag_weight"), default=0.10, minimum=0.0),
            kind_weight=coerce_float(section.get("kind_weight"), default=0.10, minimum=0.0),
            scope_weight=coerce_float(section.get("scope_weight"), default=0.05, minimum=0.0),
            key_weight=coerce_float(section.get("key_weight"), default=0.20, minimum=0.0),
            entity_weight=coerce_float(section.get("entity_weight"), default=0.10, minimum=0.0),
            intent_weight=coerce_float(section.get("intent_weight"), default=0.10, minimum=0.0),
            allowed_roles=tuple(str(role).strip().lower() for role in ensure_list(section.get("allowed_roles", ["user", "assistant", "system", "tool", "agent", "unknown"])) if str(role).strip()),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["default_scope"] = self.default_scope.value
        payload["memory_path"] = str(self.memory_path) if self.memory_path else None
        payload["metadata_export_path"] = str(self.metadata_export_path) if self.metadata_export_path else None
        return prune_none(payload, drop_empty=True)


class LanguageMemory:
    """Shared language memory store for the language agent subsystem."""

    VERSION = "1.0"

    def __init__(self):
        self.config = load_global_config()
        self.memory_config = get_config_section("language_memory")
        self.settings = LanguageMemoryConfig.from_mapping(self.memory_config)
        self.VERSION = self.settings.version

        self.records: "OrderedDict[str, MemoryRecord]" = OrderedDict()
        self.by_kind: Dict[MemoryKind, Set[str]] = defaultdict(set)
        self.by_scope: Dict[MemoryScope, Set[str]] = defaultdict(set)
        self.by_key: Dict[str, Set[str]] = defaultdict(set)
        self.by_tag: Dict[str, Set[str]] = defaultdict(set)
        self.by_entity: Dict[str, Set[str]] = defaultdict(set)
        self.by_intent: Dict[str, Set[str]] = defaultdict(set)
        self.by_role: Dict[MemoryRole, Set[str]] = defaultdict(set)

        self.stats = LanguageMemoryStats()
        self.diagnostics = LanguageDiagnostics()
        self._last_save = 0.0

        self._validate_settings()
        self._prepare_storage()
        if self.settings.load_on_init and self.settings.memory_path is not None and self.settings.memory_path.exists():
            self._load_from_disk()

        logger.info("LanguageMemory initialized with max_records=%s, memory_path=%s", self.settings.max_records, self.settings.memory_path)

    # ------------------------------------------------------------------
    # Core write methods
    # ------------------------------------------------------------------
    def remember(
        self,
        kind: Union[str, MemoryKind],
        key: Optional[str],
        value: Any,
        *,
        text: Optional[str] = None,
        scope: Union[str, MemoryScope, None] = None,
        role: Union[str, MemoryRole, None] = None,
        source: Optional[str] = None,
        confidence: float = 1.0,
        salience: float = 0.5,
        priority: int = 0,
        tags: Optional[Iterable[str]] = None,
        entities: Optional[Iterable[Any]] = None,
        frame: Optional[LinguisticFrame] = None,
        ttl_seconds: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        replace_existing: bool = False,
    ) -> MemoryRecord:
        """Store a language memory record and return the created record."""

        memory_kind = _coerce_kind(kind)
        memory_scope = _coerce_scope(scope or self.settings.default_scope)
        memory_role = _coerce_role(role or MemoryRole.UNKNOWN)
        normalized_key = self._normalize_key(key or self._derive_key(memory_kind, value, text=text, frame=frame))
        clean_text = compact_text(text if text is not None else self._derive_text(value), max_length=self.settings.max_text_length)
        now = time_module.time()
        ttl = self._effective_ttl(memory_kind, ttl_seconds)
        tag_tuple = self._normalize_tags(tags)
        entity_tuple = self._normalize_entities(entities, frame=frame)
        metadata_payload = json_safe(dict(metadata or {}))

        memory_id = generate_language_id("mem", length=24)
        if replace_existing:
            self._supersede_existing(kind=memory_kind, scope=memory_scope, key=normalized_key, replacement_id=memory_id)

        record = MemoryRecord(
            memory_id=memory_id,
            kind=memory_kind,
            scope=memory_scope,
            key=normalized_key,
            value=value,
            text=clean_text,
            created_at=now,
            updated_at=now,
            accessed_at=now,
            ttl_seconds=ttl,
            role=memory_role,
            source=source,
            confidence=coerce_probability(confidence, default=1.0),
            salience=coerce_probability(salience, default=0.5),
            priority=coerce_int(priority, default=0),
            tags=tag_tuple,
            entities=entity_tuple,
            frame=frame,
            metadata=metadata_payload,
        )
        self.records[record.memory_id] = record
        self._index_record(record)
        self.stats.writes += 1
        self._enforce_limits()
        self._autosave_if_needed()
        return record

    def remember_turn(
        self,
        role: Union[str, MemoryRole],
        content: str,
        *,
        scope: Union[str, MemoryScope, None] = None,
        frame: Optional[LinguisticFrame] = None,
        tags: Optional[Iterable[str]] = None,
        entities: Optional[Iterable[Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        source: Optional[str] = "conversation",
        salience: float = 0.45,
        confidence: float = 1.0,
    ) -> MemoryRecord:
        memory_role = _coerce_role(role)
        return self.remember(
            MemoryKind.TURN,
            key=f"turn:{memory_role.value}:{stable_hash(content, length=24)}",
            value={"role": memory_role.value, "content": content},
            text=content,
            scope=scope,
            role=memory_role,
            source=source,
            frame=frame,
            tags=tags,
            entities=entities,
            ttl_seconds=self.settings.turn_ttl_seconds,
            metadata=metadata,
            salience=salience,
            confidence=confidence,
        )

    def remember_fact(self, key: str, value: Any, *, text: Optional[str] = None, scope: Union[str, MemoryScope, None] = None, **kwargs: Any) -> MemoryRecord:
        ttl_seconds = kwargs.pop("ttl_seconds", self.settings.fact_ttl_seconds)
        return self.remember(MemoryKind.FACT, key, value, text=text, scope=scope, ttl_seconds=ttl_seconds, replace_existing=True, **kwargs)

    def remember_preference(self, key: str, value: Any, *, text: Optional[str] = None, scope: Union[str, MemoryScope, None] = MemoryScope.USER, **kwargs: Any) -> MemoryRecord:
        ttl_seconds = kwargs.pop("ttl_seconds", self.settings.preference_ttl_seconds)
        return self.remember(MemoryKind.PREFERENCE, key, value, text=text, scope=scope, ttl_seconds=ttl_seconds, replace_existing=True, **kwargs)

    def remember_slot(self, key: str, value: Any, *, text: Optional[str] = None, scope: Union[str, MemoryScope, None] = None, **kwargs: Any) -> MemoryRecord:
        ttl_seconds = kwargs.pop("ttl_seconds", self.settings.slot_ttl_seconds)
        return self.remember(MemoryKind.SLOT, key, value, text=text, scope=scope, ttl_seconds=ttl_seconds, replace_existing=True, **kwargs)

    def remember_summary(self, key: str, summary: str, *, scope: Union[str, MemoryScope, None] = None, **kwargs: Any) -> MemoryRecord:
        ttl_seconds = kwargs.pop("ttl_seconds", self.settings.summary_ttl_seconds)
        return self.remember(MemoryKind.SUMMARY, key, summary, text=summary, scope=scope, ttl_seconds=ttl_seconds, replace_existing=True, **kwargs)

    def remember_entity(self, entity: Any, *, label: Optional[str] = None, scope: Union[str, MemoryScope, None] = None, **kwargs: Any) -> MemoryRecord:
        normalized = normalize_entity(entity, default_label=label or "entity")
        key = f"entity:{normalize_entity_label(normalized.label)}:{normalize_for_comparison(normalized.text or '')}"
        return self.remember(MemoryKind.ENTITY, key, normalized, text=normalized.text, scope=scope, entities=[normalized], **kwargs)

    def remember_intent(self, intent: str, *, confidence: float = 1.0, frame: Optional[LinguisticFrame] = None, scope: Union[str, MemoryScope, None] = None, **kwargs: Any) -> MemoryRecord:
        normalized_intent = normalize_intent(intent)
        return self.remember(
            MemoryKind.INTENT,
            key=f"intent:{normalized_intent}",
            value={"intent": normalized_intent, "confidence": coerce_probability(confidence, default=1.0)},
            text=normalized_intent,
            scope=scope,
            frame=frame,
            confidence=confidence,
            **kwargs,
        )

    def remember_issue(self, issue: Union[LanguageIssue, LanguageError, Mapping[str, Any]], *, scope: Union[str, MemoryScope, None] = None, **kwargs: Any) -> MemoryRecord:
        # Normalise issue to a dictionary
        if isinstance(issue, (LanguageIssue, LanguageError)):
            issue_payload = issue.to_dict()
        else:
            issue_payload = dict(issue)  # Mapping -> dict
    
        # Extract code and message safely
        if hasattr(issue, "code"):
            code = str(issue.code) # type: ignore
        else:
            code = str(issue_payload.get("code", "language.issue"))
    
        if hasattr(issue, "message"):
            message = str(issue.message) # type: ignore
        else:
            message = str(issue_payload.get("message", code))
    
        ttl_seconds = kwargs.pop("ttl_seconds", self.settings.issue_ttl_seconds)
        return self.remember(
            MemoryKind.ISSUE,
            key=f"issue:{code}:{stable_hash(issue_payload, length=16)}",
            value=issue_payload,
            text=message,
            scope=scope,
            ttl_seconds=ttl_seconds,
            salience=kwargs.pop("salience", 0.75),
            confidence=kwargs.pop("confidence", 1.0),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Read and retrieval methods
    # ------------------------------------------------------------------
    def get(self, memory_id: str, *, include_expired: bool = False, touch: bool = True) -> Optional[MemoryRecord]:
        record = self.records.get(str(memory_id))
        self.stats.reads += 1
        if record is None or record.deleted:
            self.stats.misses += 1
            return None
        if record.is_expired() and not include_expired:
            self.stats.misses += 1
            self._delete_record(record.memory_id, reason="expired")
            return None
        if touch:
            record.touch()
        self.stats.hits += 1
        return record

    def get_by_key(
        self,
        key: str,
        *,
        kind: Union[str, MemoryKind, None] = None,
        scope: Union[str, MemoryScope, None] = None,
        latest: bool = True,
        include_expired: bool = False,
    ) -> Optional[MemoryRecord]:
        normalized_key = self._normalize_key(key)
        candidate_ids = set(self.by_key.get(normalized_key, set()))
        if kind is not None:
            candidate_ids &= self.by_kind.get(_coerce_kind(kind), set())
        if scope is not None:
            candidate_ids &= self.by_scope.get(_coerce_scope(scope), set())
        candidates = [self.records[memory_id] for memory_id in candidate_ids if memory_id in self.records]
        candidates = [record for record in candidates if record.active and (include_expired or not record.is_expired())]
        if not candidates:
            self.stats.misses += 1
            return None
        candidates.sort(key=lambda record: record.updated_at, reverse=latest)
        selected = candidates[0]
        selected.touch()
        self.stats.hits += 1
        return selected

    def recall(self, query: Union[str, MemoryQuery, None] = None, **kwargs: Any) -> List[MemoryMatch]:
        """Recall ranked memories using text and structured filters."""

        memory_query = self._normalize_query(query, **kwargs)
        candidates = self._candidate_records(memory_query)
        matches: List[MemoryMatch] = []
        for record in candidates:
            score, reasons = self._score_record(record, memory_query)
            if score >= memory_query.min_score:
                matches.append(MemoryMatch(record=record, score=score, reasons=reasons))
        matches.sort(key=lambda match: (match.score, match.record.priority, match.record.updated_at), reverse=True)
        selected = matches[: memory_query.top_k]
        for match in selected:
            match.record.touch()
        self.stats.recalls += 1
        self.stats.hits += len(selected)
        if not selected:
            self.stats.misses += 1
        return selected

    def recall_text(self, query: Union[str, MemoryQuery], *, separator: str = "\n", include_scores: bool = False, **kwargs: Any) -> str:
        matches = self.recall(query, **kwargs)
        lines: List[str] = []
        for match in matches:
            prefix = f"[{match.score:.3f}] " if include_scores else ""
            lines.append(f"{prefix}{match.record.text}")
        return separator.join(lines)

    def recent(
        self,
        *,
        limit: Optional[int] = None,
        kind: Union[str, MemoryKind, None] = None,
        scope: Union[str, MemoryScope, None] = None,
        include_expired: bool = False,
    ) -> List[MemoryRecord]:
        query = MemoryQuery(
            kinds=(_coerce_kind(kind),) if kind is not None else (),
            scopes=(_coerce_scope(scope),) if scope is not None else (),
            top_k=limit or self.settings.recall_top_k,
            include_expired=include_expired,
            min_score=0.0,
        )
        records = self._candidate_records(query)
        records.sort(key=lambda record: record.updated_at, reverse=True)
        return records[: query.top_k]

    def history(self, *, limit: Optional[int] = None, scope: Union[str, MemoryScope, None] = None) -> List[Dict[str, str]]:
        turns = self.recent(limit=limit or self.settings.retain_last_turns, kind=MemoryKind.TURN, scope=scope)
        turns.sort(key=lambda record: record.created_at)
        result: List[Dict[str, str]] = []
        for record in turns:
            if isinstance(record.value, Mapping):
                role = str(record.value.get("role", record.role.value))
                content = str(record.value.get("content", record.text))
            else:
                role = record.role.value
                content = record.text
            result.append({"role": role, "content": content})
        return result

    def get_slot(self, key: str, *, scope: Union[str, MemoryScope, None] = None, default: Any = None) -> Any:
        record = self.get_by_key(key, kind=MemoryKind.SLOT, scope=scope)
        return default if record is None else record.value

    def get_slots(self, *, scope: Union[str, MemoryScope, None] = None) -> Dict[str, Any]:
        records = self.recent(limit=self.settings.max_records, kind=MemoryKind.SLOT, scope=scope)
        return {record.key: record.value for record in records if record.active}

    def get_preferences(self, *, scope: Union[str, MemoryScope, None] = MemoryScope.USER) -> Dict[str, Any]:
        records = self.recent(limit=self.settings.max_records, kind=MemoryKind.PREFERENCE, scope=scope)
        return {record.key: record.value for record in records if record.active}

    def get_facts(self, *, scope: Union[str, MemoryScope, None] = None) -> Dict[str, Any]:
        records = self.recent(limit=self.settings.max_records, kind=MemoryKind.FACT, scope=scope)
        return {record.key: record.value for record in records if record.active}

    # ------------------------------------------------------------------
    # Update/delete lifecycle
    # ------------------------------------------------------------------
    def update(self, memory_id: str, **updates: Any) -> MemoryRecord:
        record = self.get(memory_id, include_expired=True, touch=False)
        if record is None:
            raise ContextError(
                ContextIssue(
                    code="MEMORY.RECORD.NOT_FOUND",
                    message="Cannot update a missing memory record.",
                    severity=Severity.ERROR,
                    module="LanguageMemory",
                    details={"memory_id": memory_id},
                ),
                recoverable=True,
            )
        self._unindex_record(record)
        for field_name, value in updates.items():
            if not hasattr(record, field_name):
                raise PipelineContractError(f"Unsupported memory update field: {field_name}", expected="MemoryRecord field", received=field_name)
            if field_name in {"kind", "scope", "role"}:
                value = _coerce_kind(value) if field_name == "kind" else _coerce_scope(value) if field_name == "scope" else _coerce_role(value)
            elif field_name == "tags":
                value = self._normalize_tags(value)
            elif field_name == "entities":
                value = self._normalize_entities(value)
            elif field_name in {"confidence", "salience"}:
                value = coerce_probability(value, default=getattr(record, field_name))
            setattr(record, field_name, value)
        record.refresh()
        self._index_record(record)
        self.stats.updates += 1
        self._autosave_if_needed()
        return record

    def reinforce(self, memory_id: str, *, salience_delta: float = 0.05, confidence_delta: float = 0.0) -> Optional[MemoryRecord]:
        record = self.get(memory_id)
        if record is None:
            return None
        record.salience = clamp_float(record.salience + salience_delta, 0.0, 1.0)
        record.confidence = clamp_float(record.confidence + confidence_delta, 0.0, 1.0)
        record.refresh()
        self.stats.updates += 1
        return record

    def forget(self, memory_id: str, *, hard: bool = False) -> bool:
        return self._delete_record(str(memory_id), reason="forget", hard=hard)

    def forget_by_key(self, key: str, *, kind: Union[str, MemoryKind, None] = None, scope: Union[str, MemoryScope, None] = None, hard: bool = False) -> int:
        normalized_key = self._normalize_key(key)
        candidate_ids = set(self.by_key.get(normalized_key, set()))
        if kind is not None:
            candidate_ids &= self.by_kind.get(_coerce_kind(kind), set())
        if scope is not None:
            candidate_ids &= self.by_scope.get(_coerce_scope(scope), set())
        removed = 0
        for memory_id in list(candidate_ids):
            if self._delete_record(memory_id, reason="forget_by_key", hard=hard):
                removed += 1
        return removed

    def clean_expired(self) -> int:
        now = time_module.time()
        expired_ids = [memory_id for memory_id, record in self.records.items() if record.is_expired(now)]
        for memory_id in expired_ids:
            self._delete_record(memory_id, reason="expired", hard=True)
        self.stats.expirations += len(expired_ids)
        if expired_ids:
            logger.info("Cleaned %s expired language memories", len(expired_ids))
        return len(expired_ids)

    def clear(self, *, clear_disk: bool = False) -> None:
        self.records.clear()
        self._clear_indexes()
        self.stats = LanguageMemoryStats()
        self.diagnostics = LanguageDiagnostics()
        if clear_disk and self.settings.memory_path is not None and self.settings.memory_path.exists():
            self.settings.memory_path.unlink()
            logger.info("Removed language memory file at %s", self.settings.memory_path)

    # ------------------------------------------------------------------
    # Context/snapshot helpers
    # ------------------------------------------------------------------
    def build_context_window(
        self,
        *,
        query: Optional[str] = None,
        max_records: Optional[int] = None,
        max_chars: Optional[int] = None,
        include_kinds: Optional[Iterable[Union[str, MemoryKind]]] = None,
        scope: Union[str, MemoryScope, None] = None,
        include_scores: bool = False,
    ) -> str:
        kinds = tuple(_coerce_kind(kind) for kind in include_kinds) if include_kinds else ()
        scores: Dict[str, float] = {}
        if query:
            matches = self.recall(query, kinds=kinds, scopes=(_coerce_scope(scope),) if scope else (), top_k=max_records or self.settings.recall_top_k)
            records = [match.record for match in matches]
            scores = {match.record.memory_id: match.score for match in matches}
        else:
            records = self.recent(limit=max_records or self.settings.retain_last_turns, scope=scope)
            if kinds:
                allowed = set(kinds)
                records = [record for record in records if record.kind in allowed]
        records.sort(key=lambda record: record.created_at)
        lines = [
            f"[{scores[record.memory_id]:.3f}] {self._format_record_for_context(record)}"
            if include_scores and record.memory_id in scores
            else self._format_record_for_context(record)
            for record in records
        ]
        text = "\n".join(line for line in lines if line)
        return truncate_text(text, max_chars or self.settings.max_text_length)

    def snapshot(self, *, include_expired: bool = False) -> MemorySnapshot:
        records = tuple(record for record in self.records.values() if include_expired or not record.is_expired())
        return MemorySnapshot(records=records, created_at=utc_timestamp(), stats=self.stats_snapshot(), metadata={"version": self.VERSION})

    def stats_snapshot(self) -> Dict[str, Any]:
        counts = {
            "total": len(self.records),
            "active": sum(1 for record in self.records.values() if record.active and not record.is_expired()),
            "expired": sum(1 for record in self.records.values() if record.is_expired()),
            "deleted": sum(1 for record in self.records.values() if record.deleted),
            "by_kind": {kind.value: len(ids) for kind, ids in self.by_kind.items()},
            "by_scope": {scope.value: len(ids) for scope, ids in self.by_scope.items()},
        }
        return {"version": self.VERSION, "counts": counts, "stats": self.stats.to_dict(), "settings": self.settings.to_dict()}

    def diagnostics_result(self) -> LanguageResult[Dict[str, Any]]:
        return LanguageResult(data=self.stats_snapshot(), issues=list(self.diagnostics.issues), metadata={"module": "LanguageMemory"})

    def export_metadata(self) -> Dict[str, Any]:
        payload = self.snapshot(include_expired=False).to_dict(include_values=False)
        if self.settings.metadata_export_path is not None:
            write_text_file(self.settings.metadata_export_path, json_dumps(payload, indent=2, redact=True))
        return payload

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_to_disk(self, *, force: bool = False) -> None:
        if self.settings.memory_path is None:
            return
        if not force and self.settings.autosave and time_module.time() - self._last_save < self.settings.save_interval_seconds:
            return
        if self.settings.prune_expired_on_save:
            self.clean_expired()

        payload = {
            "version": self.VERSION,
            "saved_at": time_module.time(),
            "records": self.records,
            "stats": self.stats,
        }
        target = self.settings.memory_path
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile("wb", dir=str(target.parent), delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                if self.settings.enable_compression:
                    with gzip.GzipFile(fileobj=temp_file, mode="wb") as gzip_file:
                        pickle.dump(payload, gzip_file, protocol=self.settings.serialization_protocol)
                else:
                    pickle.dump(payload, temp_file, protocol=self.settings.serialization_protocol)
            os.replace(temp_path, target)
            self._last_save = time_module.time()
            self.stats.saves += 1
            logger.debug("Language memory saved to %s", target)
        except Exception as exc:
            self.stats.save_failures += 1
            issue = ContextIssue(
                code="MEMORY.PERSISTENCE.SAVE_FAILED",
                message="Failed to save language memory to disk.",
                severity=Severity.ERROR,
                module="LanguageMemory",
                details={"memory_path": str(target), "exception_type": type(exc).__name__, "exception_message": str(exc)},
            )
            self.diagnostics.add(issue)
            logger.error("Failed to save language memory to %s: %s", target, exc)
            if self.settings.strict_persistence:
                raise ContextError(issue, recoverable=False, cause=exc) from exc

    def _load_from_disk(self) -> None:
        path = self.settings.memory_path
        if path is None or not path.exists():
            return
        try:
            payload = self._read_disk_payload(path)
            records = payload.get("records", OrderedDict()) if isinstance(payload, Mapping) else OrderedDict()
            if not isinstance(records, Mapping):
                raise ValueError("Persisted memory payload does not contain a mapping of records.")
            self.records = OrderedDict()
            for memory_id, raw_record in records.items():
                record = self._coerce_record(str(memory_id), raw_record)
                if record is not None:
                    self.records[record.memory_id] = record
            stats = payload.get("stats") if isinstance(payload, Mapping) else None
            if isinstance(stats, LanguageMemoryStats):
                self.stats = stats
            self._rebuild_indexes()
            self.clean_expired()
            self.stats.loads += 1
            logger.info("Loaded language memory from %s with %s records", path, len(self.records))
        except Exception as exc:
            self.stats.load_failures += 1
            issue = ContextIssue(
                code="MEMORY.PERSISTENCE.LOAD_FAILED",
                message="Failed to load language memory from disk.",
                severity=Severity.ERROR,
                module="LanguageMemory",
                details={"memory_path": str(path), "exception_type": type(exc).__name__, "exception_message": str(exc)},
            )
            self.diagnostics.add(issue)
            logger.error("Failed to load language memory from %s: %s", path, exc)
            self.records = OrderedDict()
            self._clear_indexes()
            if self.settings.strict_persistence:
                raise ContextError(issue, recoverable=False, cause=exc) from exc

    def _read_disk_payload(self, path: Path) -> Any:
        with open(path, "rb") as raw_file:
            prefix = raw_file.read(2)
            raw_file.seek(0)
            if prefix == b"\x1f\x8b" or self.settings.enable_compression:
                with gzip.GzipFile(fileobj=raw_file, mode="rb") as gzip_file:
                    return pickle.load(gzip_file)
            return pickle.load(raw_file)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_settings(self) -> None:
        if self.settings.enable_encryption:
            raise ConfigurationLanguageError(
                ConfigurationIssue(
                    code=LanguageErrorCode.CONFIG_VALUE_INVALID,
                    message="language_memory.enable_encryption is enabled, but encryption is not configured in this memory implementation.",
                    module="LanguageMemory",
                    details={"setting": "language_memory.enable_encryption", "value": True},
                ),
                recoverable=False,
            )

    def _prepare_storage(self) -> None:
        if self.settings.memory_path is not None:
            self.settings.memory_path.parent.mkdir(parents=True, exist_ok=True)
        if self.settings.metadata_export_path is not None:
            self.settings.metadata_export_path.parent.mkdir(parents=True, exist_ok=True)

    def _normalize_key(self, key: str) -> str:
        raw = require_non_empty_string(key, "memory key")
        if not self.settings.normalize_keys:
            return raw
        prefix = f"{self.settings.key_namespace}:"
        if raw.startswith(prefix):
            return raw
        return f"{prefix}{normalize_for_comparison(raw)}"

    def _normalize_tags(self, tags: Optional[Iterable[str]]) -> Tuple[str, ...]:
        values = [normalize_identifier_component(tag, default="tag") for tag in ensure_list(tags)]
        return tuple(dedupe_preserve_order(values))[: self.settings.max_tags]

    def _normalize_entities(self, entities: Optional[Iterable[Any]], *, frame: Optional[LinguisticFrame] = None) -> Tuple[str, ...]:
        values: List[str] = []
        for entity in ensure_list(entities):
            if isinstance(entity, Mapping):
                text = first_non_none(entity.get("text"), entity.get("value"), entity.get("name"), default="")
            else:
                text = str(entity)
            normalized = normalize_for_comparison(text)
            if normalized:
                values.append(normalized)
        if frame and isinstance(frame.entities, Mapping):
            for key, value in frame.entities.items():
                values.append(normalize_for_comparison(key))
                if isinstance(value, str):
                    values.append(normalize_for_comparison(value))
        return tuple(dedupe_preserve_order(values))[: self.settings.max_entities]

    def _derive_key(self, kind: MemoryKind, value: Any, *, text: Optional[str], frame: Optional[LinguisticFrame]) -> str:
        basis = {
            "kind": kind.value,
            "text": text,
            "value": json_safe(value),
            "intent": frame.intent if frame else None,
        }
        return f"{kind.value}:{stable_hash(basis, length=24)}"

    def _derive_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            for key in ("text", "content", "summary", "message", "value"):
                if key in value and value[key] is not None:
                    return str(value[key])
        return compact_text(json_dumps(value, redact=True), max_length=self.settings.max_text_length)

    def _effective_ttl(self, kind: MemoryKind, explicit_ttl: Optional[float]) -> Optional[float]:
        if explicit_ttl is not None:
            return _optional_seconds(explicit_ttl, default=None)
        by_kind = {
            MemoryKind.TURN: self.settings.turn_ttl_seconds,
            MemoryKind.FACT: self.settings.fact_ttl_seconds,
            MemoryKind.PREFERENCE: self.settings.preference_ttl_seconds,
            MemoryKind.SLOT: self.settings.slot_ttl_seconds,
            MemoryKind.SUMMARY: self.settings.summary_ttl_seconds,
            MemoryKind.ISSUE: self.settings.issue_ttl_seconds,
        }
        return by_kind.get(kind, self.settings.default_ttl_seconds)

    def _normalize_query(self, query: Union[str, MemoryQuery, None], **kwargs: Any) -> MemoryQuery:
        if isinstance(query, MemoryQuery):
            return query
        if isinstance(query, str):
            kwargs.setdefault("text", query)
        kinds = tuple(_coerce_kind(kind) for kind in ensure_list(kwargs.pop("kinds", kwargs.pop("kind", None))))
        scopes = tuple(_coerce_scope(scope) for scope in ensure_list(kwargs.pop("scopes", kwargs.pop("scope", None))))
        return MemoryQuery(
            text=kwargs.pop("text", None),
            key=kwargs.pop("key", None),
            kinds=kinds,
            scopes=scopes,
            tags=tuple(normalize_identifier_component(tag, default="tag") for tag in ensure_list(kwargs.pop("tags", None))),
            entities=tuple(normalize_for_comparison(entity) for entity in ensure_list(kwargs.pop("entities", None))),
            intent=normalize_intent(kwargs.pop("intent")) if kwargs.get("intent") else None,
            source=kwargs.pop("source", None),
            since=kwargs.pop("since", None),
            until=kwargs.pop("until", None),
            top_k=coerce_int(kwargs.pop("top_k", self.settings.recall_top_k), default=self.settings.recall_top_k, minimum=1),
            min_score=coerce_probability(kwargs.pop("min_score", self.settings.recall_min_score), default=self.settings.recall_min_score),
            min_confidence=kwargs.pop("min_confidence", None),
            min_salience=kwargs.pop("min_salience", None),
            include_expired=coerce_bool(kwargs.pop("include_expired", False), default=False),
            include_values=coerce_bool(kwargs.pop("include_values", True), default=True),
            metadata_filter=kwargs.pop("metadata_filter", None),
        )

    def _candidate_records(self, query: MemoryQuery) -> List[MemoryRecord]:
        candidate_ids: Optional[Set[str]] = None
        filters: List[Set[str]] = []
        if query.kinds:
            filters.append(set().union(*(self.by_kind.get(kind, set()) for kind in query.kinds)))
        if query.scopes:
            filters.append(set().union(*(self.by_scope.get(scope, set()) for scope in query.scopes)))
        if query.key:
            filters.append(set(self.by_key.get(self._normalize_key(query.key), set())))
        if query.tags:
            filters.append(set().union(*(self.by_tag.get(tag, set()) for tag in query.tags)))
        if query.entities:
            filters.append(set().union(*(self.by_entity.get(entity, set()) for entity in query.entities)))
        if query.intent:
            filters.append(set(self.by_intent.get(query.intent, set())))
        for item in filters:
            candidate_ids = item if candidate_ids is None else candidate_ids & item
        ids = candidate_ids if candidate_ids is not None else set(self.records.keys())
        records: List[MemoryRecord] = []
        for memory_id in ids:
            record = self.records.get(memory_id)
            if record is None or not record.active:
                continue
            if record.is_expired() and not query.include_expired:
                continue
            if query.since is not None and record.created_at < query.since:
                continue
            if query.until is not None and record.created_at > query.until:
                continue
            if query.source is not None and record.source != query.source:
                continue
            if query.min_confidence is not None and record.confidence < query.min_confidence:
                continue
            if query.min_salience is not None and record.salience < query.min_salience:
                continue
            if query.metadata_filter is not None and not query.metadata_filter(record.metadata):
                continue
            records.append(record)
        return records

    def _score_record(self, record: MemoryRecord, query: MemoryQuery) -> Tuple[float, Dict[str, float]]:
        reasons: Dict[str, float] = {}
        reasons["recency"] = self._recency_score(record) * self.settings.recency_weight
        reasons["salience"] = record.decay_salience(half_life_seconds=self.settings.salience_decay_half_life_seconds) * self.settings.salience_weight
        reasons["confidence"] = record.confidence * self.settings.confidence_weight
        if query.text:
            reasons["lexical"] = lexical_overlap(query.text, record.text) * self.settings.lexical_weight
        if query.tags:
            overlap = len(set(query.tags) & set(record.tags)) / max(1, len(set(query.tags)))
            reasons["tags"] = overlap * self.settings.tag_weight
        if query.kinds:
            reasons["kind"] = (1.0 if record.kind in query.kinds else 0.0) * self.settings.kind_weight
        if query.scopes:
            reasons["scope"] = (1.0 if record.scope in query.scopes else 0.0) * self.settings.scope_weight
        if query.key:
            reasons["key"] = (1.0 if record.key == self._normalize_key(query.key) else 0.0) * self.settings.key_weight
        if query.entities:
            overlap = len(set(query.entities) & set(record.entities)) / max(1, len(set(query.entities)))
            reasons["entities"] = overlap * self.settings.entity_weight
        if query.intent:
            reasons["intent"] = (1.0 if record.intent == query.intent else 0.0) * self.settings.intent_weight
        return sum(reasons.values()), reasons

    def _recency_score(self, record: MemoryRecord) -> float:
        if not self.settings.salience_decay_half_life_seconds:
            return 1.0
        return clamp_float(math.pow(0.5, record.age_seconds / self.settings.salience_decay_half_life_seconds), 0.0, 1.0)

    def _supersede_existing(self, *, kind: MemoryKind, scope: MemoryScope, key: str, replacement_id: str) -> None:
        existing_ids = set(self.by_kind.get(kind, set())) & set(self.by_scope.get(scope, set())) & set(self.by_key.get(key, set()))
        for memory_id in existing_ids:
            record = self.records.get(memory_id)
            if record and not record.deleted:
                record.superseded_by = replacement_id
                record.refresh()

    def _enforce_limits(self) -> None:
        self._enforce_turn_limit()
        self._enforce_total_limit()
        self._enforce_scope_limit()

    def _enforce_turn_limit(self) -> None:
        turn_records = [self.records[memory_id] for memory_id in self.by_kind.get(MemoryKind.TURN, set()) if memory_id in self.records]
        active_turns = [record for record in turn_records if not record.deleted]
        if len(active_turns) <= self.settings.max_turns:
            return
        active_turns.sort(key=lambda record: record.created_at)
        for record in active_turns[: len(active_turns) - self.settings.max_turns]:
            self._delete_record(record.memory_id, reason="turn_limit", hard=True)
            self.stats.evictions += 1

    def _enforce_total_limit(self) -> None:
        active_records = [record for record in self.records.values() if not record.deleted]
        if len(active_records) <= self.settings.max_records:
            return
        active_records.sort(key=lambda record: (record.priority, record.salience, record.accessed_at, record.created_at))
        for record in active_records[: len(active_records) - self.settings.max_records]:
            self._delete_record(record.memory_id, reason="max_records", hard=True)
            self.stats.evictions += 1

    def _enforce_scope_limit(self) -> None:
        for scope, ids in list(self.by_scope.items()):
            records = [self.records[memory_id] for memory_id in ids if memory_id in self.records and not self.records[memory_id].deleted]
            if len(records) <= self.settings.max_records_per_scope:
                continue
            records.sort(key=lambda record: (record.priority, record.salience, record.accessed_at, record.created_at))
            for record in records[: len(records) - self.settings.max_records_per_scope]:
                self._delete_record(record.memory_id, reason="scope_limit", hard=True)
                self.stats.evictions += 1

    def _format_record_for_context(self, record: MemoryRecord) -> str:
        if record.kind == MemoryKind.TURN:
            return f"{record.role.value}: {record.text}"
        label = record.kind.value
        return f"[{label}] {record.text}"

    def _coerce_record(self, memory_id: str, raw_record: Any) -> Optional[MemoryRecord]:
        if isinstance(raw_record, MemoryRecord):
            return raw_record
        if not isinstance(raw_record, Mapping):
            return None
        now = time_module.time()
        frame_payload = raw_record.get("frame")
        frame = frame_from_mapping(frame_payload) if isinstance(frame_payload, Mapping) else None
        return MemoryRecord(
            memory_id=str(raw_record.get("memory_id", memory_id)),
            kind=_coerce_kind(raw_record.get("kind", MemoryKind.NOTE.value)),
            scope=_coerce_scope(raw_record.get("scope", self.settings.default_scope.value)),
            key=str(raw_record.get("key", memory_id)),
            value=raw_record.get("value"),
            text=str(raw_record.get("text", "")),
            created_at=float(raw_record.get("created_at", now)),
            updated_at=float(raw_record.get("updated_at", now)),
            accessed_at=float(raw_record.get("accessed_at", now)),
            ttl_seconds=_optional_seconds(raw_record.get("ttl_seconds"), default=None),
            role=_coerce_role(raw_record.get("role", MemoryRole.UNKNOWN.value)),
            source=raw_record.get("source"),
            confidence=coerce_probability(raw_record.get("confidence"), default=1.0),
            salience=coerce_probability(raw_record.get("salience"), default=0.5),
            priority=coerce_int(raw_record.get("priority"), default=0),
            tags=tuple(str(tag) for tag in ensure_list(raw_record.get("tags"))),
            entities=tuple(str(entity) for entity in ensure_list(raw_record.get("entities"))),
            frame=frame,
            metadata=dict(json_safe(raw_record.get("metadata", {}))),
            hits=coerce_int(raw_record.get("hits"), default=0, minimum=0),
            superseded_by=raw_record.get("superseded_by"),
            deleted=coerce_bool(raw_record.get("deleted"), default=False),
        )

    def _index_record(self, record: MemoryRecord) -> None:
        self.by_kind[record.kind].add(record.memory_id)
        self.by_scope[record.scope].add(record.memory_id)
        self.by_key[record.key].add(record.memory_id)
        self.by_role[record.role].add(record.memory_id)
        for tag in record.tags:
            self.by_tag[tag].add(record.memory_id)
        for entity in record.entities:
            self.by_entity[entity].add(record.memory_id)
        if record.intent:
            self.by_intent[record.intent].add(record.memory_id)

    def _unindex_record(self, record: MemoryRecord) -> None:
        self.by_kind[record.kind].discard(record.memory_id)
        self.by_scope[record.scope].discard(record.memory_id)
        self.by_key[record.key].discard(record.memory_id)
        self.by_role[record.role].discard(record.memory_id)
        for tag in record.tags:
            self.by_tag[tag].discard(record.memory_id)
        for entity in record.entities:
            self.by_entity[entity].discard(record.memory_id)
        if record.intent:
            self.by_intent[record.intent].discard(record.memory_id)

    def _rebuild_indexes(self) -> None:
        self._clear_indexes()
        for record in self.records.values():
            self._index_record(record)

    def _clear_indexes(self) -> None:
        self.by_kind.clear()
        self.by_scope.clear()
        self.by_key.clear()
        self.by_tag.clear()
        self.by_entity.clear()
        self.by_intent.clear()
        self.by_role.clear()

    def _delete_record(self, memory_id: str, *, reason: str, hard: bool = False) -> bool:
        record = self.records.get(memory_id)
        if record is None:
            return False
        self._unindex_record(record)
        if hard:
            self.records.pop(memory_id, None)
        else:
            record.deleted = True
            record.metadata = merge_mappings(record.metadata, {"deleted_reason": reason, "deleted_at": utc_timestamp()})
            self.records[memory_id] = record
        self.stats.deletes += 1
        self._autosave_if_needed()
        return True

    def _autosave_if_needed(self) -> None:
        if not self.settings.autosave:
            return
        if self.settings.memory_path is None:
            return
        if time_module.time() - self._last_save >= self.settings.save_interval_seconds:
            self.save_to_disk(force=True)

    def __len__(self) -> int:
        return len([record for record in self.records.values() if not record.deleted])

    def __contains__(self, memory_id: str) -> bool:
        record = self.records.get(memory_id)
        return record is not None and not record.deleted

    def __iter__(self) -> Iterator[MemoryRecord]:
        return (record for record in self.records.values() if not record.deleted)

    def __repr__(self) -> str:
        return f"LanguageMemory(version={self.VERSION!r}, records={len(self)}, path={self.settings.memory_path!s})"


# ---------------------------------------------------------------------------
# Local coercion helpers for memory-specific enums
# ---------------------------------------------------------------------------
def _coerce_memory_enum(value: Union[str, E_mem, None], enum_cls: Type[E_mem], default: E_mem) -> E_mem:
    """Generic enum coercion for memory-specific enums."""
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    raw_value = str(value).strip().lower()
    # Handle aliases if needed (e.g., for MemoryRole)
    aliases: Dict[str, str] = {}
    if enum_cls == MemoryRole:
        aliases = {"assistant": "assistant", "ai": "assistant", "bot": "assistant", "agent": "agent", "user": "user", "human": "user", "system": "system", "tool": "tool"}
        raw_value = aliases.get(raw_value, raw_value)
    try:
        return enum_cls(raw_value)
    except ValueError:
        logger.warning("Unknown %s value: %r. Using default %s.", enum_cls.__name__, raw_value, default.value)
        return default

def _coerce_kind(value: Union[str, MemoryKind]) -> MemoryKind:
    return _coerce_memory_enum(value, MemoryKind, MemoryKind.NOTE)

def _coerce_scope(value: Union[str, MemoryScope]) -> MemoryScope:
    return _coerce_memory_enum(value, MemoryScope, MemoryScope.SESSION)

def _coerce_role(value: Union[str, MemoryRole]) -> MemoryRole:
    return _coerce_memory_enum(value, MemoryRole, MemoryRole.UNKNOWN)


def _optional_seconds(value: Any, *, default: Optional[float]) -> Optional[float]:
    if value is None:
        return default
    number = coerce_float(value, default=-1.0)
    if number <= 0:
        return None
    return number


if __name__ == "__main__":
    print("\n=== Running Language Memory ===\n")
    printer.status("TEST", "Language Memory initialized", "info")

    memory = LanguageMemory()
    memory.clear(clear_disk=False)

    frame = make_linguistic_frame(
        intent="remember_user_preference",
        entities={"preference": "detailed technical answers", "topic": "language subsystem"},
        sentiment=0.2,
        modality="declarative",
        confidence=0.93,
        act_type=SpeechActType.ASSERTIVE,
        propositional_content="The user prefers detailed production-ready language modules.",
    )

    user_turn = memory.remember_turn(
        "user",
        "Please keep the language subsystem production-ready and avoid duplicating helper logic.",
        frame=frame,
        tags=["requirements", "language", "production"],
        salience=0.82,
    )
    assistant_turn = memory.remember_turn(
        "assistant",
        "I will keep memory-specific logic in language_memory and reuse shared helpers.",
        tags=["commitment", "language"],
        salience=0.64,
    )
    preference = memory.remember_preference(
        "response_detail_level",
        "thorough and detailed",
        text="The user prefers thorough, detailed, production-ready code reviews and modules.",
        tags=["user_preference", "style"],
        salience=0.9,
        frame=frame,
    )
    slot = memory.remember_slot(
        "current_module",
        "language_memory",
        text="The active module under construction is language_memory.",
        tags=["module", "state"],
    )
    fact = memory.remember_fact(
        "integration_rule",
        "Use language_helpers and language_error directly; do not wrap local imports in fallback try/except blocks.",
        tags=["integration", "rules"],
        salience=0.88,
    )
    summary = memory.remember_summary(
        "session_summary",
        "The session is improving the language subsystem with production-ready support modules.",
        tags=["summary"],
    )

    matches = memory.recall("production-ready language memory helper integration", top_k=5)
    context_window = memory.build_context_window(query="language subsystem requirements", max_records=5, include_scores=False)
    history = memory.history(limit=10)
    slots = memory.get_slots()
    preferences = memory.get_preferences()
    stats = memory.stats_snapshot()

    if not matches:
        raise AssertionError("Memory recall failed.")
    if memory.get(preference.memory_id) is None:
        raise AssertionError("Preference lookup failed.")
    if memory.get_slot("current_module") != "language_memory":
        raise AssertionError("Slot lookup failed.")
    if len(history) != 2:
        raise AssertionError("Turn history reconstruction failed.")
    if "language subsystem" not in context_window.lower():
        raise AssertionError("Context window generation failed.")

    printer.pretty("User turn", user_turn.to_dict(), "success")
    printer.pretty("Assistant turn", assistant_turn.to_dict(), "success")
    printer.pretty("Preference", preference.to_dict(), "success")
    printer.pretty("Slot", slot.to_dict(), "success")
    printer.pretty("Fact", fact.to_dict(), "success")
    printer.pretty("Summary", summary.to_dict(), "success")
    printer.pretty("Matches", [match.to_dict() for match in matches], "success")
    printer.pretty("Context window", context_window, "success")
    printer.pretty("History", history, "success")
    printer.pretty("Slots", slots, "success")
    printer.pretty("Preferences", preferences, "success")
    printer.pretty("Stats", stats, "success")

    memory.clean_expired()
    if memory.settings.memory_path is not None:
        memory.save_to_disk(force=True)
        printer.status("TEST", f"Language Memory persisted to {memory.settings.memory_path}", "success")

    print("\n=== Test ran successfully ===\n")
