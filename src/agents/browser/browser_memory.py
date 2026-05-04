from __future__ import annotations

"""
Production-grade browser memory backend.

This module provides a thread-safe, namespaced memory layer for the browser
agent subsystem. It is intentionally scoped to browser automation and browsing
telemetry rather than general application persistence.

The memory layer is designed to store and retrieve:

- page snapshots and extracted page content;
- search result candidates and query/result relationships;
- browser action outcomes from navigation, clicking, typing, scrolling,
  clipboard operations, extraction, and workflow execution;
- lightweight task context used by planning, retry, learning, and reasoning
  layers;
- structured metadata such as URLs, domains, tags, action names, scores,
  source labels, expiry information, and access statistics.

Design principles
-----------------
1. Browser-focused: the API speaks in browsing concepts such as URLs, actions,
   page snapshots, search results, workflow executions, and selectors.
2. Expandable: public methods are granular and composable so later browser
   modules can integrate without duplicating indexing, expiry, serialization, or
   result-building logic.
3. Safe by default: exported data is redacted by default, values are JSON-safe,
   stale/expired records are pruned automatically, and persistence writes can be
   atomic.
4. Deterministic contracts: entries, query objects, stats, snapshots, and result
   payloads have stable shapes for tests, logs, metrics, and downstream agents.
5. Integration-first: error handling and helper utilities are reused from the
   browser subsystem instead of reimplementing validation, serialization,
   redaction, hashing, URL normalization, and result payload construction.

Local imports are intentionally direct. They are not wrapped in try/except so
packaging or path problems fail clearly during development and deployment.
"""

import copy
import os

from collections import OrderedDict, defaultdict, deque
from collections.abc import Mapping as ABCMapping, Sequence as ABCSequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.browser_errors import *
from .utils.Browser_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Browser Memory")
printer = PrettyPrinter


# ---------------------------------------------------------------------------
# Constants and defaults
# ---------------------------------------------------------------------------
DEFAULT_MEMORY_NAMESPACE = "default"
DEFAULT_MEMORY_KIND = "generic"
MEMORY_SCHEMA_VERSION = "1.0"
ENTRY_REF_SEPARATOR = "\u241f"

SUPPORTED_MERGE_STRATEGIES = {
    "replace",
    "merge",
    "append",
    "extend",
    "extend_unique",
    "keep_existing",
}

SORTABLE_FIELDS = {
    "created_at",
    "updated_at",
    "last_accessed_at",
    "access_count",
    "score",
    "priority",
    "key",
    "kind",
    "namespace",
}

# These are fallback defaults only. The canonical runtime configuration belongs
# in browser_config.yaml under the browser_memory section. See the generated
# browser_config entry provided with this module.
DEFAULT_BROWSER_MEMORY_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "default_namespace": DEFAULT_MEMORY_NAMESPACE,
    "max_entries": 5000,
    "max_entries_per_namespace": 1000,
    "default_ttl_seconds": None,
    "auto_prune": True,
    "prune_interval_seconds": 60,
    "copy_values_on_write": True,
    "copy_values_on_read": True,
    "redact_on_export": True,
    "include_values_in_snapshots": True,
    "max_value_chars": 20000,
    "query": {
        "default_limit": 50,
        "max_limit": 500,
        "default_sort_by": "updated_at",
        "default_descending": True,
    },
    "scoring": {
        "default_score": 1.0,
        "min_score": 0.0,
        "max_score": 1.0,
    },
    "event_log": {
        "enabled": True,
        "max_events": 500,
    },
    "persistence": {
        "enabled": False,
        "path": "storage/browser_memory.json",
        "autosave": False,
        "autosave_interval_seconds": 30,
        "atomic_write": True,
        "load_on_startup": False,
    },
    "namespaces": {
        "default": {"max_entries": 1000, "ttl_seconds": None},
        "pages": {"max_entries": 1000, "ttl_seconds": 86400},
        "search": {"max_entries": 1500, "ttl_seconds": 43200},
        "actions": {"max_entries": 2000, "ttl_seconds": 21600},
        "workflows": {"max_entries": 500, "ttl_seconds": 604800},
        "security": {"max_entries": 500, "ttl_seconds": 604800},
        "scratch": {"max_entries": 500, "ttl_seconds": 3600},
    },
}


class MemoryKind(str, Enum):
    """Common browser memory entry categories.

    The enum is intentionally not restrictive. ``BrowserMemory`` accepts custom
    kind strings so future browser modules can add new categories without
    changing this file.
    """

    GENERIC = "generic"
    PAGE_SNAPSHOT = "page_snapshot"
    PAGE_CONTENT = "page_content"
    SEARCH_QUERY = "search_query"
    SEARCH_RESULT = "search_result"
    ACTION_RESULT = "action_result"
    WORKFLOW_RESULT = "workflow_result"
    TASK_RESULT = "task_result"
    ELEMENT = "element"
    SECURITY_EVENT = "security_event"
    ERROR = "error"
    NOTE = "note"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class MemoryEntry:
    """Structured record stored inside ``BrowserMemory``.

    ``value`` may hold arbitrary browser-related content, but every exported
    representation is passed through browser helper serialization/redaction.
    ``key`` is unique within a namespace, while ``id`` is unique across all
    namespaces for stable references.
    """

    key: str
    value: Any
    namespace: str = DEFAULT_MEMORY_NAMESPACE
    kind: str = DEFAULT_MEMORY_KIND
    id: str = ""
    tags: Tuple[str, ...] = ()
    url: Optional[str] = None
    domain: Optional[str] = None
    action: Optional[str] = None
    source: Optional[str] = None
    score: float = 1.0
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    expires_at: Optional[str] = None
    last_accessed_at: Optional[str] = None
    access_count: int = 0
    version: int = 1
    fingerprint: str = ""

    def __post_init__(self) -> None:
        self.namespace = normalize_memory_namespace(self.namespace)
        self.key = normalize_memory_key(self.key)
        self.kind = normalize_memory_kind(self.kind)
        self.tags = normalize_memory_tags(self.tags)
        self.action = normalize_action_name(self.action) if self.action else None
        self.source = normalize_whitespace(self.source) if self.source else None
        self.metadata = dict(self.metadata or {})
        self.score = coerce_float(self.score, default=1.0, minimum=0.0)
        self.priority = coerce_int(self.priority, default=0)
        if not self.id:
            self.id = new_correlation_id("mem")
        if self.url:
            self.url = normalize_url(self.url)
            if not self.domain:
                self.domain = safe_domain_from_url(self.url)
        if not self.fingerprint:
            self.fingerprint = self.build_fingerprint()

    @property
    def ref(self) -> str:
        return build_entry_ref(self.namespace, self.id)

    def build_fingerprint(self) -> str:
        return stable_hash(
            {
                "namespace": self.namespace,
                "key": self.key,
                "kind": self.kind,
                "url": self.url,
                "action": self.action,
                "value": safe_serialize(self.value),
                "metadata": safe_serialize(self.metadata),
            },
            length=20,
        )

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if not self.expires_at:
            return False
        expires_at = parse_memory_datetime(self.expires_at)
        if expires_at is None:
            return False
        current = now or datetime.now(timezone.utc)
        return expires_at <= current

    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed_at = utc_now_iso()

    def update(
        self,
        *,
        value: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
        tags: Optional[Iterable[Any]] = None,
        score: Optional[float] = None,
        priority: Optional[int] = None,
        expires_at: Optional[Union[str, datetime]] = None,
        ttl_seconds: Optional[Union[int, float]] = None,
        url: Optional[str] = None,
        action: Optional[str] = None,
        source: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> None:
        if value is not None:
            self.value = value
        if metadata:
            self.metadata = merge_dicts(self.metadata, dict(metadata), deep=True)
        if tags is not None:
            self.tags = normalize_memory_tags([*self.tags, *ensure_list(tags)])
        if score is not None:
            self.score = coerce_float(score, default=self.score, minimum=0.0)
        if priority is not None:
            self.priority = coerce_int(priority, default=self.priority)
        if kind is not None:
            self.kind = normalize_memory_kind(kind)
        if url is not None:
            self.url = normalize_url(url) if url else None
            self.domain = safe_domain_from_url(self.url) if self.url else None
        if action is not None:
            self.action = normalize_action_name(action) if action else None
        if source is not None:
            self.source = normalize_whitespace(source) if source else None
        if ttl_seconds is not None:
            self.expires_at = expiry_from_ttl(ttl_seconds)
        elif expires_at is not None:
            self.expires_at = memory_datetime_to_iso(expires_at)
        self.version += 1
        self.updated_at = utc_now_iso()
        self.fingerprint = self.build_fingerprint()

    def clone(self, *, deep: bool = True) -> "MemoryEntry":
        return copy.deepcopy(self) if deep else copy.copy(self)

    def to_dict(self, *, include_value: bool = True, redact: bool = True) -> Dict[str, Any]:
        payload = {
            "id": self.id,
            "ref": self.ref,
            "namespace": self.namespace,
            "key": self.key,
            "kind": self.kind,
            "tags": list(self.tags),
            "url": self.url,
            "domain": self.domain,
            "action": self.action,
            "source": self.source,
            "score": self.score,
            "priority": self.priority,
            "metadata": safe_serialize(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "version": self.version,
            "fingerprint": self.fingerprint,
            "expired": self.is_expired(),
        }
        if include_value:
            payload["value"] = safe_serialize(self.value)
        payload = prune_none(payload)
        return redact_mapping(payload) if redact else payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MemoryEntry":
        require_mapping(payload, "memory_entry", allow_empty=False)
        return cls(
            id=str(payload.get("id") or ""),
            namespace=str(payload.get("namespace") or DEFAULT_MEMORY_NAMESPACE),
            key=str(payload.get("key") or ""),
            value=payload.get("value"),
            kind=str(payload.get("kind") or DEFAULT_MEMORY_KIND),
            tags=tuple(payload.get("tags") or ()),
            url=payload.get("url"),
            domain=payload.get("domain"),
            action=payload.get("action"),
            source=payload.get("source"),
            score=coerce_float(payload.get("score"), default=1.0, minimum=0.0),
            priority=coerce_int(payload.get("priority"), default=0),
            metadata=dict(payload.get("metadata") or {}),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            updated_at=str(payload.get("updated_at") or utc_now_iso()),
            expires_at=payload.get("expires_at"),
            last_accessed_at=payload.get("last_accessed_at"),
            access_count=coerce_int(payload.get("access_count"), default=0, minimum=0),
            version=coerce_int(payload.get("version"), default=1, minimum=1),
            fingerprint=str(payload.get("fingerprint") or ""),
        )


@dataclass
class MemoryStats:
    """Statistics about the current state of the memory."""

    namespaces_count: int = 0
    entries_total: int = 0
    expired_count: int = 0
    events_total: int = 0
    hits: int = 0
    misses: int = 0
    writes: int = 0
    updates: int = 0
    deletes: int = 0
    evictions: int = 0
    prunes: int = 0
    queries: int = 0
    merges: int = 0
    saves: int = 0
    loads: int = 0
    namespace_sizes: Dict[str, int] = field(default_factory=dict)
    kind_counts: Dict[str, int] = field(default_factory=dict)
    tag_counts: Dict[str, int] = field(default_factory=dict)
    action_counts: Dict[str, int] = field(default_factory=dict)
    domain_counts: Dict[str, int] = field(default_factory=dict)
    current_bytes_estimate: int = 0
    oldest_created_at: Optional[str] = None
    newest_updated_at: Optional[str] = None
    generated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = prune_none(asdict(self))
        return redact_mapping(payload) if redact else payload


@dataclass
class MemoryQuery:
    """Structured query object for retrieving information from memory."""

    namespaces: Optional[Tuple[str, ...]] = None
    keys: Optional[Tuple[str, ...]] = None
    ids: Optional[Tuple[str, ...]] = None
    refs: Optional[Tuple[str, ...]] = None
    kinds: Optional[Tuple[str, ...]] = None
    tags: Optional[Tuple[str, ...]] = None
    urls: Optional[Tuple[str, ...]] = None
    domains: Optional[Tuple[str, ...]] = None
    actions: Optional[Tuple[str, ...]] = None
    sources: Optional[Tuple[str, ...]] = None
    text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    include_expired: bool = False
    include_value: bool = True
    match_any_tag: bool = True
    exact_key: bool = True
    min_score: Optional[float] = None
    created_after: Optional[str] = None
    created_before: Optional[str] = None
    updated_after: Optional[str] = None
    updated_before: Optional[str] = None
    sort_by: str = "updated_at"
    descending: bool = True
    offset: int = 0
    limit: int = 50

    def __post_init__(self) -> None:
        self.namespaces = normalize_optional_tuple(self.namespaces, normalize_memory_namespace)
        self.keys = normalize_optional_tuple(self.keys, normalize_memory_key)
        self.ids = normalize_optional_tuple(self.ids, lambda value: normalize_whitespace(value))
        self.refs = normalize_optional_tuple(self.refs, lambda value: normalize_whitespace(value))
        self.kinds = normalize_optional_tuple(self.kinds, normalize_memory_kind)
        self.tags = normalize_optional_tuple(self.tags, lambda value: normalize_memory_tag(value))
        self.urls = normalize_optional_tuple(self.urls, lambda value: normalize_url(value))
        self.domains = normalize_optional_tuple(self.domains, lambda value: normalize_whitespace(value).lower())
        self.actions = normalize_optional_tuple(self.actions, normalize_action_name)
        self.sources = normalize_optional_tuple(self.sources, lambda value: normalize_whitespace(value))
        self.text = normalize_whitespace(self.text) if self.text else None
        self.metadata = dict(self.metadata or {})
        self.include_expired = coerce_bool(self.include_expired, default=False)
        self.include_value = coerce_bool(self.include_value, default=True)
        self.match_any_tag = coerce_bool(self.match_any_tag, default=True)
        self.exact_key = coerce_bool(self.exact_key, default=True)
        self.offset = coerce_int(self.offset, default=0, minimum=0)
        self.limit = coerce_int(self.limit, default=50, minimum=0)
        self.sort_by = normalize_whitespace(self.sort_by or "updated_at")
        if self.sort_by not in SORTABLE_FIELDS:
            self.sort_by = "updated_at"

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]], *, defaults: Optional[Mapping[str, Any]] = None) -> "MemoryQuery":
        merged = merge_dicts(dict(defaults or {}), dict(payload or {}), deep=True)
        return cls(
            namespaces=tuple(ensure_list(merged.get("namespaces") or merged.get("namespace"))) or None,
            keys=tuple(ensure_list(merged.get("keys") or merged.get("key"))) or None,
            ids=tuple(ensure_list(merged.get("ids") or merged.get("id"))) or None,
            refs=tuple(ensure_list(merged.get("refs") or merged.get("ref"))) or None,
            kinds=tuple(ensure_list(merged.get("kinds") or merged.get("kind"))) or None,
            tags=tuple(ensure_list(merged.get("tags") or merged.get("tag"))) or None,
            urls=tuple(ensure_list(merged.get("urls") or merged.get("url"))) or None,
            domains=tuple(ensure_list(merged.get("domains") or merged.get("domain"))) or None,
            actions=tuple(ensure_list(merged.get("actions") or merged.get("action"))) or None,
            sources=tuple(ensure_list(merged.get("sources") or merged.get("source"))) or None,
            text=merged.get("text") or merged.get("query"),
            metadata=dict(merged.get("metadata") or {}),
            include_expired=merged.get("include_expired", False),
            include_value=merged.get("include_value", True),
            match_any_tag=merged.get("match_any_tag", True),
            exact_key=merged.get("exact_key", True),
            min_score=merged.get("min_score"),
            created_after=merged.get("created_after"),
            created_before=merged.get("created_before"),
            updated_after=merged.get("updated_after"),
            updated_before=merged.get("updated_before"),
            sort_by=merged.get("sort_by", "updated_at"),
            descending=merged.get("descending", True),
            offset=merged.get("offset", 0),
            limit=merged.get("limit", 50),
        )

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(asdict(self))


# ---------------------------------------------------------------------------
# Normalization helpers specific to memory concepts
# ---------------------------------------------------------------------------
def normalize_memory_namespace(namespace: Any) -> str:
    text = normalize_whitespace(namespace or DEFAULT_MEMORY_NAMESPACE).lower()
    text = text.replace(" ", "_")
    return text or DEFAULT_MEMORY_NAMESPACE


def normalize_memory_key(key: Any) -> str:
    text = normalize_whitespace(key)
    if not text:
        raise MissingRequiredFieldError("Memory key must be a non-empty string", context={"field": "key"})
    return text


def normalize_memory_kind(kind: Any) -> str:
    text = normalize_whitespace(kind or DEFAULT_MEMORY_KIND).lower().replace(" ", "_")
    return text or DEFAULT_MEMORY_KIND


def normalize_memory_tag(tag: Any) -> str:
    return normalize_whitespace(tag).lower().replace(" ", "_")


def normalize_memory_tags(tags: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    normalized = [normalize_memory_tag(tag) for tag in ensure_list(tags) if normalize_memory_tag(tag)]
    return tuple(dedupe_preserve_order(normalized))


def normalize_optional_tuple(values: Optional[Iterable[Any]], normalizer: Any) -> Optional[Tuple[str, ...]]:
    if values is None:
        return None
    normalized = [normalizer(value) for value in ensure_list(values) if normalizer(value)]
    return tuple(dedupe_preserve_order(normalized)) or None


def build_entry_ref(namespace: str, entry_id: str) -> str:
    return f"{normalize_memory_namespace(namespace)}{ENTRY_REF_SEPARATOR}{entry_id}"


def split_entry_ref(ref: str) -> Tuple[Optional[str], Optional[str]]:
    text = normalize_whitespace(ref)
    if ENTRY_REF_SEPARATOR not in text:
        return None, text or None
    namespace, entry_id = text.split(ENTRY_REF_SEPARATOR, 1)
    return normalize_memory_namespace(namespace), entry_id or None


def parse_memory_datetime(value: Optional[Union[str, datetime]]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def memory_datetime_to_iso(value: Union[str, datetime]) -> Optional[str]:
    dt = parse_memory_datetime(value)
    return dt.isoformat() if dt else None


def expiry_from_ttl(ttl_seconds: Optional[Union[int, float]]) -> Optional[str]:
    if ttl_seconds is None:
        return None
    seconds = coerce_float(ttl_seconds, default=0.0, minimum=0.0)
    if seconds <= 0:
        return None
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def safe_domain_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        return parse_browser_url(url).host.lower() or None
    except Exception:
        return None


def estimate_payload_bytes(value: Any) -> int:
    return len(safe_json_dumps(value, sort_keys=True, redact=True).encode("utf-8", errors="replace"))


def extract_browser_url_from_payload(payload: Any) -> Optional[str]:
    if isinstance(payload, MemoryEntry):
        return payload.url
    if not isinstance(payload, ABCMapping):
        return None
    for key in ("url", "current_url", "link", "href"):
        value = payload.get(key)
        if value and is_valid_url(str(value)):
            return normalize_url(str(value))
    page = payload.get("page") or payload.get("content")
    if isinstance(page, ABCMapping):
        return extract_browser_url_from_payload(page)
    return None


def entry_text_blob(entry: MemoryEntry) -> str:
    parts = [
        entry.namespace,
        entry.key,
        entry.kind,
        " ".join(entry.tags),
        entry.url or "",
        entry.domain or "",
        entry.action or "",
        entry.source or "",
        safe_json_dumps(entry.metadata, sort_keys=True, redact=True),
        safe_json_dumps(entry.value, sort_keys=True, redact=True),
    ]
    return normalize_whitespace(" ".join(parts)).lower()


# ---------------------------------------------------------------------------
# Browser memory backend
# ---------------------------------------------------------------------------
class BrowserMemory:
    """Thread-safe namespaced memory backend for the browser agent subsystem."""

    def __init__(self, *, config: Optional[Mapping[str, Any]] = None, load_on_startup: Optional[bool] = None) -> None:
        self.config = load_global_config()
        configured = get_config_section("browser_memory") or {}
        self.memory_config = merge_dicts(DEFAULT_BROWSER_MEMORY_CONFIG, configured, dict(config or {}), deep=True)
        self.enabled = coerce_bool(self.memory_config.get("enabled"), default=True)
        self.default_namespace = normalize_memory_namespace(self.memory_config.get("default_namespace", DEFAULT_MEMORY_NAMESPACE))
        self.max_entries = coerce_int(self.memory_config.get("max_entries"), default=5000, minimum=1)
        self.max_entries_per_namespace = coerce_int(self.memory_config.get("max_entries_per_namespace"), default=1000, minimum=1)
        self.default_ttl_seconds = self.memory_config.get("default_ttl_seconds")
        self.auto_prune = coerce_bool(self.memory_config.get("auto_prune"), default=True)
        self.prune_interval_seconds = coerce_float(self.memory_config.get("prune_interval_seconds"), default=60.0, minimum=0.0)
        self.copy_values_on_write = coerce_bool(self.memory_config.get("copy_values_on_write"), default=True)
        self.copy_values_on_read = coerce_bool(self.memory_config.get("copy_values_on_read"), default=True)
        self.redact_on_export = coerce_bool(self.memory_config.get("redact_on_export"), default=True)
        self.include_values_in_snapshots = coerce_bool(self.memory_config.get("include_values_in_snapshots"), default=True)
        self.max_value_chars = coerce_int(self.memory_config.get("max_value_chars"), default=20000, minimum=100)

        query_config = dict(self.memory_config.get("query") or {})
        self.default_query_limit = coerce_int(query_config.get("default_limit"), default=50, minimum=0)
        self.max_query_limit = coerce_int(query_config.get("max_limit"), default=500, minimum=1)
        self.default_sort_by = normalize_whitespace(query_config.get("default_sort_by") or "updated_at")
        self.default_descending = coerce_bool(query_config.get("default_descending"), default=True)

        scoring_config = dict(self.memory_config.get("scoring") or {})
        self.default_score = coerce_float(scoring_config.get("default_score"), default=1.0, minimum=0.0)
        self.min_score = coerce_float(scoring_config.get("min_score"), default=0.0, minimum=0.0)
        self.max_score = coerce_float(scoring_config.get("max_score"), default=1.0, minimum=self.min_score)

        event_config = dict(self.memory_config.get("event_log") or {})
        self.event_log_enabled = coerce_bool(event_config.get("enabled"), default=True)
        self.max_events = coerce_int(event_config.get("max_events"), default=500, minimum=0)

        persistence_config = dict(self.memory_config.get("persistence") or {})
        self.persistence_enabled = coerce_bool(persistence_config.get("enabled"), default=False)
        self.persistence_path = self._resolve_path(persistence_config.get("path"))
        self.autosave = coerce_bool(persistence_config.get("autosave"), default=False)
        self.autosave_interval_seconds = coerce_float(persistence_config.get("autosave_interval_seconds"), default=30.0, minimum=0.0)
        self.atomic_write = coerce_bool(persistence_config.get("atomic_write"), default=True)
        should_load = coerce_bool(persistence_config.get("load_on_startup"), default=False) if load_on_startup is None else bool(load_on_startup)

        self.namespace_config = dict(self.memory_config.get("namespaces") or {})
        self._lock = RLock()
        self._entries: Dict[str, OrderedDict[str, MemoryEntry]] = defaultdict(OrderedDict)
        self._key_index: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._indexes: Dict[str, Dict[str, Set[str]]] = {
            "tag": defaultdict(set),
            "url": defaultdict(set),
            "domain": defaultdict(set),
            "kind": defaultdict(set),
            "action": defaultdict(set),
            "source": defaultdict(set),
        }
        self._events: Deque[Dict[str, Any]] = deque(maxlen=self.max_events or None)
        self._counters: Dict[str, int] = defaultdict(int)
        self._last_pruned_at: Optional[datetime] = None
        self._last_saved_at: Optional[datetime] = None

        if should_load and self.persistence_path and self.persistence_path.exists():
            self.load()

        logger.info("BrowserMemory initialized with namespace '%s'", self.default_namespace)

    # ------------------------------------------------------------------
    # Configuration and lifecycle helpers
    # ------------------------------------------------------------------
    def _resolve_path(self, value: Any) -> Optional[Path]:
        if not value:
            return None
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return path
        config_path = self.config.get("__config_path__") if isinstance(self.config, ABCMapping) else None
        if config_path:
            return Path(config_path).resolve().parent / path
        return Path.cwd() / path

    def _namespace_limit(self, namespace: str) -> int:
        ns_config = self._namespace_config(namespace)
        configured = ns_config.get("max_entries", self.max_entries_per_namespace)
        return coerce_int(configured, default=self.max_entries_per_namespace, minimum=1)

    def _namespace_ttl(self, namespace: str) -> Optional[Union[int, float]]:
        ns_config = self._namespace_config(namespace)
        return ns_config.get("ttl_seconds", self.default_ttl_seconds)

    def _namespace_config(self, namespace: str) -> Dict[str, Any]:
        namespace = normalize_memory_namespace(namespace)
        value = self.namespace_config.get(namespace, {})
        return dict(value) if isinstance(value, ABCMapping) else {}

    def _assert_enabled(self) -> None:
        if not self.enabled:
            raise BrowserStateError("Browser memory is disabled", context={"enabled": self.enabled})

    def _maybe_prune(self) -> None:
        if not self.auto_prune:
            return
        now = datetime.now(timezone.utc)
        if self._last_pruned_at is None or (now - self._last_pruned_at).total_seconds() >= self.prune_interval_seconds:
            self.prune_expired(now=now)

    def _maybe_autosave(self) -> None:
        if not (self.persistence_enabled and self.autosave):
            return
        now = datetime.now(timezone.utc)
        if self._last_saved_at is None or (now - self._last_saved_at).total_seconds() >= self.autosave_interval_seconds:
            self.save()

    # ------------------------------------------------------------------
    # Indexing and event log
    # ------------------------------------------------------------------
    def _entry_ref(self, entry: MemoryEntry) -> str:
        return entry.ref

    def _index_entry(self, entry: MemoryEntry) -> None:
        ref = self._entry_ref(entry)
        self._key_index[entry.namespace][entry.key] = entry.id
        for tag in entry.tags:
            self._indexes["tag"][tag].add(ref)
        if entry.url:
            self._indexes["url"][entry.url].add(ref)
        if entry.domain:
            self._indexes["domain"][entry.domain].add(ref)
        if entry.kind:
            self._indexes["kind"][entry.kind].add(ref)
        if entry.action:
            self._indexes["action"][entry.action].add(ref)
        if entry.source:
            self._indexes["source"][entry.source].add(ref)

    def _deindex_entry(self, entry: MemoryEntry) -> None:
        ref = self._entry_ref(entry)
        self._key_index.get(entry.namespace, {}).pop(entry.key, None)
        for index in self._indexes.values():
            empty_keys: List[str] = []
            for key, refs in index.items():
                refs.discard(ref)
                if not refs:
                    empty_keys.append(key)
            for key in empty_keys:
                index.pop(key, None)

    def _record_event(self, event: str, *, entry: Optional[MemoryEntry] = None, data: Optional[Mapping[str, Any]] = None) -> None:
        if not self.event_log_enabled or self.max_events == 0:
            return
        payload = {
            "event": event,
            "at": utc_now_iso(),
            "entry_ref": entry.ref if entry else None,
            "namespace": entry.namespace if entry else None,
            "key": entry.key if entry else None,
            "kind": entry.kind if entry else None,
            "data": safe_serialize(data or {}),
        }
        self._events.append(redact_mapping(prune_none(payload)))

    # ------------------------------------------------------------------
    # Core write/read/delete API
    # ------------------------------------------------------------------
    def put(
        self,
        key: str,
        value: Any,
        *,
        namespace: Optional[str] = None,
        kind: Union[str, MemoryKind] = MemoryKind.GENERIC,
        tags: Optional[Iterable[Any]] = None,
        url: Optional[str] = None,
        action: Optional[str] = None,
        source: Optional[str] = None,
        score: Optional[float] = None,
        priority: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
        ttl_seconds: Optional[Union[int, float]] = None,
        expires_at: Optional[Union[str, datetime]] = None,
        replace: bool = True,
    ) -> MemoryEntry:
        """Store or update a browser memory entry.

        ``key`` is unique inside ``namespace``. Repeated writes update the
        existing entry by default while preserving access counters and creation
        time. Set ``replace=False`` to force duplicate-key validation.
        """

        self._assert_enabled()
        namespace = normalize_memory_namespace(namespace or self.default_namespace)
        key = normalize_memory_key(key)
        kind_value = normalize_memory_kind(kind.value if isinstance(kind, MemoryKind) else kind)
        normalized_url = normalize_url(url) if url else None
        normalized_action = normalize_action_name(action) if action else None
        ttl = ttl_seconds if ttl_seconds is not None else self._namespace_ttl(namespace)
        resolved_expires_at = memory_datetime_to_iso(expires_at) if expires_at else expiry_from_ttl(ttl)
        stored_value = copy.deepcopy(value) if self.copy_values_on_write else value
        bounded_score = coerce_float(score, default=self.default_score, minimum=self.min_score, maximum=self.max_score)

        with self._lock:
            self._maybe_prune()
            existing = self._get_entry_by_key_unlocked(namespace, key)
            if existing is not None and not replace:
                raise BrowserStateError("Memory key already exists", context={"namespace": namespace, "key": key})

            if existing is not None:
                self._deindex_entry(existing)
                existing.update(
                    value=stored_value,
                    metadata=metadata,
                    tags=tags,
                    score=bounded_score,
                    priority=priority,
                    expires_at=resolved_expires_at,
                    url=normalized_url,
                    action=normalized_action,
                    source=source,
                    kind=kind_value,
                )
                entry = existing
                self._counters["updates"] += 1
                event_name = "update"
            else:
                entry = MemoryEntry(
                    namespace=namespace,
                    key=key,
                    value=stored_value,
                    kind=kind_value,
                    tags=normalize_memory_tags(tags),
                    url=normalized_url,
                    action=normalized_action,
                    source=source,
                    score=bounded_score,
                    priority=priority,
                    metadata=dict(metadata or {}),
                    expires_at=resolved_expires_at,
                )
                self._entries[namespace][entry.id] = entry
                self._counters["writes"] += 1
                event_name = "put"

            self._entries[namespace].move_to_end(entry.id)
            self._index_entry(entry)
            self._enforce_limits_unlocked(namespace)
            self._record_event(event_name, entry=entry)
            self._maybe_autosave()
            return entry.clone()

    def get(
        self,
        key: str,
        *,
        namespace: Optional[str] = None,
        default: Any = None,
        include_expired: bool = False,
        touch: bool = True,
    ) -> Any:
        """Return a stored value by key or ``default`` when absent/expired."""

        entry = self.get_entry(key, namespace=namespace, include_expired=include_expired, touch=touch)
        if entry is None:
            return default
        value = entry.value
        return copy.deepcopy(value) if self.copy_values_on_read else value

    def get_entry(
        self,
        key_or_id_or_ref: str,
        *,
        namespace: Optional[str] = None,
        include_expired: bool = False,
        touch: bool = True,
    ) -> Optional[MemoryEntry]:
        self._assert_enabled()
        identifier = normalize_whitespace(key_or_id_or_ref)
        namespace = normalize_memory_namespace(namespace or self.default_namespace)
        with self._lock:
            self._maybe_prune()
            entry = self._resolve_entry_unlocked(identifier, namespace=namespace)
            if entry is None:
                self._counters["misses"] += 1
                return None
            if entry.is_expired() and not include_expired:
                self._counters["misses"] += 1
                return None
            if touch:
                self._deindex_entry(entry)
                entry.touch()
                self._entries[entry.namespace].move_to_end(entry.id)
                self._index_entry(entry)
            self._counters["hits"] += 1
            self._record_event("get", entry=entry)
            return entry.clone()

    def exists(self, key: str, *, namespace: Optional[str] = None, include_expired: bool = False) -> bool:
        return self.get_entry(key, namespace=namespace, include_expired=include_expired, touch=False) is not None

    def delete(self, key_or_id_or_ref: str, *, namespace: Optional[str] = None) -> bool:
        self._assert_enabled()
        identifier = normalize_whitespace(key_or_id_or_ref)
        namespace = normalize_memory_namespace(namespace or self.default_namespace)
        with self._lock:
            entry = self._resolve_entry_unlocked(identifier, namespace=namespace)
            if entry is None:
                self._counters["misses"] += 1
                return False
            self._delete_entry_unlocked(entry, event="delete")
            self._counters["deletes"] += 1
            self._maybe_autosave()
            return True

    def clear_namespace(self, namespace: Optional[str] = None) -> int:
        self._assert_enabled()
        namespace = normalize_memory_namespace(namespace or self.default_namespace)
        with self._lock:
            entries = list(self._entries.get(namespace, {}).values())
            for entry in entries:
                self._delete_entry_unlocked(entry, event="clear_namespace")
            self._entries.pop(namespace, None)
            self._key_index.pop(namespace, None)
            self._maybe_autosave()
            return len(entries)

    def clear_all(self) -> int:
        self._assert_enabled()
        with self._lock:
            total = sum(len(entries) for entries in self._entries.values())
            self._entries.clear()
            self._key_index.clear()
            for index in self._indexes.values():
                index.clear()
            self._events.clear()
            self._record_event("clear_all", data={"deleted": total})
            self._maybe_autosave()
            return total

    # ------------------------------------------------------------------
    # Browser-specific write helpers
    # ------------------------------------------------------------------
    def remember_page(
        self,
        page: Mapping[str, Any],
        *,
        key: Optional[str] = None,
        namespace: str = "pages",
        tags: Optional[Iterable[Any]] = None,
        ttl_seconds: Optional[Union[int, float]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MemoryEntry:
        """Store a page snapshot or extracted page content dictionary."""

        require_mapping(page, "page", allow_empty=False)
        url = extract_browser_url_from_payload(page)
        title = normalize_whitespace(page.get("title")) if isinstance(page, ABCMapping) else ""
        entry_key = key or stable_hash({"url": url, "title": title, "text": page.get("text")}, length=24)
        page_tags = ["page"] + ensure_list(tags)
        if url and is_probably_pdf_url(url):
            page_tags.append("pdf")
        if url and is_arxiv_url(url):
            page_tags.append("arxiv")
        return self.put(
            entry_key,
            dict(page),
            namespace=namespace,
            kind=MemoryKind.PAGE_SNAPSHOT,
            tags=page_tags,
            url=url,
            source="browser_page",
            metadata=merge_dicts({"title": title, "content_type": classify_resource_url(url or "")}, metadata or {}, deep=True),
            ttl_seconds=ttl_seconds,
        )

    def remember_action(
        self,
        action: str,
        result: Mapping[str, Any],
        *,
        request: Optional[Mapping[str, Any]] = None,
        key: Optional[str] = None,
        namespace: str = "actions",
        tags: Optional[Iterable[Any]] = None,
        ttl_seconds: Optional[Union[int, float]] = None,
    ) -> MemoryEntry:
        """Store a browser action result with normalized action metadata."""

        require_mapping(result, "result", allow_empty=False)
        normalized_action = normalize_action_name(action or result.get("action"))
        url = extract_browser_url_from_payload(result) or extract_browser_url_from_payload(request or {})
        status = result.get("status")
        error_payload = result.get("error")
        code = result.get("code")
        if code is None and isinstance(error_payload, ABCMapping):
            code = error_payload.get("code")
        entry_key = key or stable_hash({"action": normalized_action, "request": request, "result": result, "at": utc_now_iso()}, length=24)
        action_tags = ["action", normalized_action, str(status or "unknown")] + ensure_list(tags)
        return self.put(
            entry_key,
            dict(result),
            namespace=namespace,
            kind=MemoryKind.ACTION_RESULT,
            tags=action_tags,
            url=url,
            action=normalized_action,
            source="browser_action",
            metadata={"request": safe_serialize(request or {}), "status": status, "code": code},
            ttl_seconds=ttl_seconds,
        )

    def remember_task_result(
        self,
        task_data: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        namespace: str = "actions",
        tags: Optional[Iterable[Any]] = None,
        ttl_seconds: Optional[Union[int, float]] = None,
    ) -> MemoryEntry:
        """Store a BrowserAgent.perform_task-style task/result pair."""

        normalized_task = normalize_task_payload(task_data)
        action = normalized_task.get("task") or result.get("action") or "task"
        return self.remember_action(
            action,
            result,
            request=normalized_task,
            namespace=namespace,
            tags=["task_result", *ensure_list(tags)],
            ttl_seconds=ttl_seconds,
        )

    def remember_search_query(
        self,
        query: str,
        *,
        results: Optional[Sequence[Mapping[str, Any]]] = None,
        engine_url: Optional[str] = None,
        namespace: str = "search",
        ttl_seconds: Optional[Union[int, float]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MemoryEntry:
        """Store a search query and optional result summary."""

        query_text = require_non_empty_string(query, "query")
        key = stable_hash({"query": query_text, "engine_url": engine_url}, length=24)
        value = {"query": query_text, "engine_url": engine_url, "results": safe_serialize(results or [])}
        return self.put(
            key,
            value,
            namespace=namespace,
            kind=MemoryKind.SEARCH_QUERY,
            tags=["search", "query"],
            url=engine_url,
            action="search",
            source="browser_search",
            metadata=merge_dicts({"result_count": len(results or [])}, metadata or {}, deep=True),
            ttl_seconds=ttl_seconds,
        )

    def remember_search_result(
        self,
        query: str,
        result: Mapping[str, Any],
        *,
        rank: Optional[int] = None,
        namespace: str = "search",
        tags: Optional[Iterable[Any]] = None,
        ttl_seconds: Optional[Union[int, float]] = None,
    ) -> MemoryEntry:
        """Store one search result/link candidate."""

        require_mapping(result, "search_result", allow_empty=False)
        query_text = require_non_empty_string(query, "query")
        url = extract_browser_url_from_payload(result)
        title = normalize_whitespace(result.get("title") or result.get("text") or result.get("name"))
        key = stable_hash({"query": query_text, "url": url, "title": title, "rank": rank}, length=24)
        return self.put(
            key,
            dict(result),
            namespace=namespace,
            kind=MemoryKind.SEARCH_RESULT,
            tags=["search", "result", *ensure_list(tags)],
            url=url,
            action="search",
            source="browser_search_result",
            score=coerce_float(result.get("score"), default=self.default_score, minimum=self.min_score, maximum=self.max_score),
            metadata={"query": query_text, "rank": rank, "title": title},
            ttl_seconds=ttl_seconds,
        )

    def remember_search_results(
        self,
        query: str,
        results: Sequence[Mapping[str, Any]],
        *,
        namespace: str = "search",
        ttl_seconds: Optional[Union[int, float]] = None,
    ) -> List[MemoryEntry]:
        """Store a search query plus each returned search result."""

        require_sequence(results, "results", allow_empty=True)
        entries = [self.remember_search_query(query, results=results, namespace=namespace, ttl_seconds=ttl_seconds)]
        for rank, result in enumerate(results, start=1):
            if isinstance(result, ABCMapping):
                entries.append(self.remember_search_result(query, result, rank=rank, namespace=namespace, ttl_seconds=ttl_seconds))
        return entries

    def remember_workflow_execution(
        self,
        workflow_result: Mapping[str, Any],
        *,
        workflow: Optional[Sequence[Mapping[str, Any]]] = None,
        key: Optional[str] = None,
        namespace: str = "workflows",
        ttl_seconds: Optional[Union[int, float]] = None,
    ) -> MemoryEntry:
        """Store workflow execution output and optional normalized workflow plan."""

        require_mapping(workflow_result, "workflow_result", allow_empty=False)
        normalized_workflow = normalize_workflow(workflow or []) if workflow is not None else None
        status = workflow_result.get("status")
        entry_key = key or stable_hash({"workflow": normalized_workflow, "result": workflow_result, "at": utc_now_iso()}, length=24)
        return self.put(
            entry_key,
            dict(workflow_result),
            namespace=namespace,
            kind=MemoryKind.WORKFLOW_RESULT,
            tags=["workflow", str(status or "unknown")],
            action="workflow",
            source="browser_workflow",
            metadata={"workflow": safe_serialize(normalized_workflow), "status": status},
            ttl_seconds=ttl_seconds,
        )

    def remember_security_event(
        self,
        event: Mapping[str, Any],
        *,
        key: Optional[str] = None,
        namespace: str = "security",
        ttl_seconds: Optional[Union[int, float]] = None,
    ) -> MemoryEntry:
        """Store CAPTCHA, bot-detection, rate-limit, or permission events."""

        require_mapping(event, "security_event", allow_empty=False)
        url = extract_browser_url_from_payload(event)
        event_type = normalize_memory_tag(event.get("type") or event.get("event") or "security")
        entry_key = key or stable_hash({"event": event, "at": utc_now_iso()}, length=24)
        return self.put(
            entry_key,
            dict(event),
            namespace=namespace,
            kind=MemoryKind.SECURITY_EVENT,
            tags=["security", event_type],
            url=url,
            action=normalize_action_name(event.get("action")) if event.get("action") else None,
            source="browser_security",
            metadata={"event_type": event_type},
            ttl_seconds=ttl_seconds,
        )

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------
    def query(self, query: Optional[Union[MemoryQuery, Mapping[str, Any]]] = None, **overrides: Any) -> List[MemoryEntry]:
        """Return entries matching a structured memory query."""

        self._assert_enabled()
        if isinstance(query, MemoryQuery):
            query_obj = query
            if overrides:
                query_obj = MemoryQuery.from_mapping(merge_dicts(query.to_dict(), overrides, deep=True))
        else:
            defaults = {
                "limit": self.default_query_limit,
                "sort_by": self.default_sort_by,
                "descending": self.default_descending,
            }
            query_obj = MemoryQuery.from_mapping(merge_dicts(query or {}, overrides, deep=True), defaults=defaults)
        query_obj.limit = min(query_obj.limit, self.max_query_limit)

        with self._lock:
            self._maybe_prune()
            candidates = list(self._iter_entries_unlocked(query_obj.namespaces))
            filtered = [entry for entry in candidates if self._entry_matches_query(entry, query_obj)]
            filtered.sort(key=lambda entry: self._sort_value(entry, query_obj.sort_by), reverse=query_obj.descending)
            sliced = filtered[query_obj.offset : query_obj.offset + query_obj.limit if query_obj.limit else None]
            self._counters["queries"] += 1
            self._record_event("query", data={"query": query_obj.to_dict(), "matched": len(filtered), "returned": len(sliced)})
            return [entry.clone() for entry in sliced]

    def query_dicts(self, query: Optional[Union[MemoryQuery, Mapping[str, Any]]] = None, **overrides: Any) -> List[Dict[str, Any]]:
        query_obj = query if isinstance(query, MemoryQuery) else MemoryQuery.from_mapping(merge_dicts(query or {}, overrides, deep=True))
        return [entry.to_dict(include_value=query_obj.include_value, redact=self.redact_on_export) for entry in self.query(query_obj)]

    def recall(self, text: str, *, namespace: Optional[str] = None, limit: Optional[int] = None, include_value: bool = True) -> List[Dict[str, Any]]:
        """Convenience text search over browser memory."""

        query = MemoryQuery(
            namespaces=(namespace,) if namespace else None,
            text=text,
            limit=limit if limit is not None else self.default_query_limit,
            include_value=include_value,
            sort_by="updated_at",
            descending=True,
        )
        return [entry.to_dict(include_value=include_value, redact=self.redact_on_export) for entry in self.query(query)]

    def recent(self, *, namespace: Optional[str] = None, limit: int = 20, include_value: bool = True) -> List[Dict[str, Any]]:
        return self.query_dicts(
            {
                "namespace": namespace,
                "limit": limit,
                "include_value": include_value,
                "sort_by": "updated_at",
                "descending": True,
            }
        )

    def by_tag(self, tag: str, *, namespace: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        return self.query_dicts({"namespace": namespace, "tag": tag, "limit": limit})

    def by_url(self, url: str, *, namespace: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        return self.query_dicts({"namespace": namespace, "url": normalize_url(url), "limit": limit})

    def by_domain(self, domain: str, *, namespace: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        return self.query_dicts({"namespace": namespace, "domain": domain, "limit": limit})

    def by_action(self, action: str, *, namespace: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        return self.query_dicts({"namespace": namespace, "action": normalize_action_name(action), "limit": limit})

    def recall_for_task(self, task_data: Mapping[str, Any], *, limit: int = 10) -> List[Dict[str, Any]]:
        """Return likely relevant entries for a BrowserAgent task payload."""

        task = normalize_task_payload(task_data)
        action = task.get("task")
        url = extract_browser_url_from_payload(task)
        query_text = task.get("query") or task.get("text") or task.get("selector") or action
        results: List[Dict[str, Any]] = []
        if url:
            results.extend(self.by_url(url, limit=limit))
            domain = safe_domain_from_url(url)
            if domain:
                results.extend(self.by_domain(domain, limit=limit))
        if action:
            results.extend(self.by_action(action, limit=limit))
        if query_text:
            results.extend(self.recall(str(query_text), limit=limit))
        deduped: Dict[str, Dict[str, Any]] = {}
        for item in results:
            ref = item.get("ref") or item.get("id") or stable_hash(item)
            deduped[str(ref)] = item
        return list(deduped.values())[:limit]

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def add_tags(self, key_or_id_or_ref: str, tags: Iterable[Any], *, namespace: Optional[str] = None) -> Optional[MemoryEntry]:
        return self._mutate_entry(key_or_id_or_ref, namespace=namespace, event="add_tags", tags=tags)

    def remove_tags(self, key_or_id_or_ref: str, tags: Iterable[Any], *, namespace: Optional[str] = None) -> Optional[MemoryEntry]:
        remove_set = set(normalize_memory_tags(tags))

        def mutate(entry: MemoryEntry) -> None:
            entry.tags = tuple(tag for tag in entry.tags if tag not in remove_set)
            entry.updated_at = utc_now_iso()
            entry.version += 1
            entry.fingerprint = entry.build_fingerprint()

        return self._mutate_entry(key_or_id_or_ref, namespace=namespace, event="remove_tags", custom_mutator=mutate)

    def update_metadata(self, key_or_id_or_ref: str, metadata: Mapping[str, Any], *, namespace: Optional[str] = None) -> Optional[MemoryEntry]:
        return self._mutate_entry(key_or_id_or_ref, namespace=namespace, event="update_metadata", metadata=metadata)

    def reinforce(self, key_or_id_or_ref: str, *, namespace: Optional[str] = None, amount: float = 0.1) -> Optional[MemoryEntry]:
        def mutate(entry: MemoryEntry) -> None:
            entry.score = coerce_float(entry.score + amount, default=entry.score, minimum=self.min_score, maximum=self.max_score)
            entry.updated_at = utc_now_iso()
            entry.version += 1
            entry.fingerprint = entry.build_fingerprint()

        return self._mutate_entry(key_or_id_or_ref, namespace=namespace, event="reinforce", custom_mutator=mutate)

    def decay(self, key_or_id_or_ref: str, *, namespace: Optional[str] = None, amount: float = 0.1) -> Optional[MemoryEntry]:
        return self.reinforce(key_or_id_or_ref, namespace=namespace, amount=-abs(amount))

    def merge(
        self,
        key: str,
        value: Any,
        *,
        namespace: Optional[str] = None,
        strategy: str = "merge",
        kind: Union[str, MemoryKind] = MemoryKind.GENERIC,
        tags: Optional[Iterable[Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MemoryEntry:
        """Merge a value into an existing entry, or create one if absent."""

        strategy = normalize_whitespace(strategy).lower()
        if strategy not in SUPPORTED_MERGE_STRATEGIES:
            raise BrowserValidationError("Unsupported memory merge strategy", context={"strategy": strategy, "supported": sorted(SUPPORTED_MERGE_STRATEGIES)})
        namespace = normalize_memory_namespace(namespace or self.default_namespace)
        with self._lock:
            existing = self._get_entry_by_key_unlocked(namespace, normalize_memory_key(key))
            if existing is None:
                return self.put(key, value, namespace=namespace, kind=kind, tags=tags, metadata=metadata)
            merged_value = self._merge_values(existing.value, value, strategy=strategy)
            self._counters["merges"] += 1
        return self.put(
            key,
            merged_value,
            namespace=namespace,
            kind=existing.kind,
            tags=tags,
            url=existing.url,
            action=existing.action,
            source=existing.source,
            score=existing.score,
            priority=existing.priority,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Persistence, import/export, snapshots
    # ------------------------------------------------------------------
    def snapshot(self, *, include_values: Optional[bool] = None, redact: Optional[bool] = None) -> Dict[str, Any]:
        include_values = self.include_values_in_snapshots if include_values is None else include_values
        redact = self.redact_on_export if redact is None else redact
        with self._lock:
            payload = {
                "schema_version": MEMORY_SCHEMA_VERSION,
                "created_at": utc_now_iso(),
                "config": safe_serialize(self.memory_config),
                "stats": self.stats().to_dict(redact=redact),
                "entries": [entry.to_dict(include_value=include_values, redact=redact) for entry in self._iter_entries_unlocked(None)],
                "events": list(self._events),
            }
            return redact_mapping(payload) if redact else payload

    def load_snapshot(self, snapshot: Mapping[str, Any], *, merge: bool = False) -> int:
        require_mapping(snapshot, "snapshot", allow_empty=False)
        entries = snapshot.get("entries") or []
        require_sequence(entries, "snapshot.entries", allow_empty=True)
        loaded = 0
        with self._lock:
            if not merge:
                self.clear_all()
            for item in entries:
                if not isinstance(item, ABCMapping):
                    continue
                entry = MemoryEntry.from_dict(item)
                existing = self._get_entry_by_key_unlocked(entry.namespace, entry.key)
                if existing is not None:
                    self._delete_entry_unlocked(existing, event="load_replace")
                self._entries[entry.namespace][entry.id] = entry
                self._index_entry(entry)
                loaded += 1
            self._enforce_all_limits_unlocked()
            self._counters["loads"] += 1
            self._record_event("load_snapshot", data={"loaded": loaded, "merge": merge})
            self._maybe_autosave()
            return loaded

    def export_json(self, *, include_values: Optional[bool] = None, redact: Optional[bool] = None, indent: Optional[int] = 2) -> str:
        return safe_json_dumps(self.snapshot(include_values=include_values, redact=redact), indent=indent, redact=False)

    def import_json(self, payload: str, *, merge: bool = False) -> int:
        loaded = safe_json_loads(payload)
        if not isinstance(loaded, ABCMapping):
            raise BrowserValidationError("Memory JSON payload must decode to a mapping", context={"type": type(loaded).__name__})
        return self.load_snapshot(loaded, merge=merge)

    def save(self, path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        target = Path(path).expanduser() if path else self.persistence_path
        if target is None:
            raise BrowserConfigurationError("Browser memory persistence path is not configured")
        with self._lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = self.export_json(include_values=True, redact=False, indent=2)
            if self.atomic_write:
                tmp_path = target.with_suffix(f"{target.suffix}.tmp")
                tmp_path.write_text(payload, encoding="utf-8")
                os.replace(tmp_path, target)
            else:
                target.write_text(payload, encoding="utf-8")
            self._last_saved_at = datetime.now(timezone.utc)
            self._counters["saves"] += 1
            self._record_event("save", data={"path": str(target)})
            return success_result(action="memory.save", message="Browser memory saved", data={"path": str(target), "entries": self.total_entries()})

    def load(self, path: Optional[Union[str, Path]] = None, *, merge: bool = False) -> Dict[str, Any]:
        target = Path(path).expanduser() if path else self.persistence_path
        if target is None:
            raise BrowserConfigurationError("Browser memory persistence path is not configured")
        if not target.exists():
            raise BrowserStateError("Browser memory persistence file does not exist", context={"path": str(target)})
        payload = target.read_text(encoding="utf-8")
        loaded_count = self.import_json(payload, merge=merge)
        return success_result(action="memory.load", message="Browser memory loaded", data={"path": str(target), "entries": loaded_count})

    # ------------------------------------------------------------------
    # Maintenance and stats
    # ------------------------------------------------------------------
    def prune_expired(self, *, now: Optional[datetime] = None) -> int:
        self._assert_enabled()
        now = now or datetime.now(timezone.utc)
        removed = 0
        with self._lock:
            for entry in list(self._iter_entries_unlocked(None)):
                if entry.is_expired(now):
                    self._delete_entry_unlocked(entry, event="prune_expired")
                    removed += 1
            self._last_pruned_at = now
            self._counters["prunes"] += removed
            if removed:
                self._maybe_autosave()
            return removed

    def compact(self) -> Dict[str, Any]:
        """Rebuild indexes, prune expired entries, and enforce capacity limits."""

        with self._lock:
            pruned = self.prune_expired()
            self._rebuild_indexes_unlocked()
            before = self.total_entries()
            self._enforce_all_limits_unlocked()
            after = self.total_entries()
            evicted = max(0, before - after)
            self._record_event("compact", data={"pruned": pruned, "evicted": evicted})
            self._maybe_autosave()
            return success_result(action="memory.compact", message="Browser memory compacted", data={"pruned": pruned, "evicted": evicted, "entries": after})

    def total_entries(self) -> int:
        with self._lock:
            return sum(len(entries) for entries in self._entries.values())

    def namespaces(self) -> List[str]:
        with self._lock:
            return sorted(self._entries.keys())

    def stats(self) -> MemoryStats:
        with self._lock:
            entries = list(self._iter_entries_unlocked(None))
            namespace_sizes = {namespace: len(items) for namespace, items in self._entries.items()}
            kind_counts: Dict[str, int] = defaultdict(int)
            tag_counts: Dict[str, int] = defaultdict(int)
            action_counts: Dict[str, int] = defaultdict(int)
            domain_counts: Dict[str, int] = defaultdict(int)
            expired_count = 0
            oldest_created_at: Optional[str] = None
            newest_updated_at: Optional[str] = None
            bytes_estimate = 0
            for entry in entries:
                kind_counts[entry.kind] += 1
                if entry.action:
                    action_counts[entry.action] += 1
                if entry.domain:
                    domain_counts[entry.domain] += 1
                for tag in entry.tags:
                    tag_counts[tag] += 1
                if entry.is_expired():
                    expired_count += 1
                if oldest_created_at is None or entry.created_at < oldest_created_at:
                    oldest_created_at = entry.created_at
                if newest_updated_at is None or entry.updated_at > newest_updated_at:
                    newest_updated_at = entry.updated_at
                bytes_estimate += estimate_payload_bytes(entry.to_dict(include_value=True, redact=True))
            return MemoryStats(
                namespaces_count=len(namespace_sizes),
                entries_total=len(entries),
                expired_count=expired_count,
                events_total=len(self._events),
                hits=self._counters["hits"],
                misses=self._counters["misses"],
                writes=self._counters["writes"],
                updates=self._counters["updates"],
                deletes=self._counters["deletes"],
                evictions=self._counters["evictions"],
                prunes=self._counters["prunes"],
                queries=self._counters["queries"],
                merges=self._counters["merges"],
                saves=self._counters["saves"],
                loads=self._counters["loads"],
                namespace_sizes=dict(namespace_sizes),
                kind_counts=dict(kind_counts),
                tag_counts=dict(tag_counts),
                action_counts=dict(action_counts),
                domain_counts=dict(domain_counts),
                current_bytes_estimate=bytes_estimate,
                oldest_created_at=oldest_created_at,
                newest_updated_at=newest_updated_at,
            )

    def events(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._events)
            if limit is not None:
                items = items[-max(0, int(limit)):]
            return copy.deepcopy(items)

    # ------------------------------------------------------------------
    # Result-style wrappers for integration points that prefer dict results
    # ------------------------------------------------------------------
    def safe_put(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        start = monotonic_ms()
        try:
            entry = self.put(*args, **kwargs)
            return success_result(action="memory.put", message="Memory entry stored", data={"entry": entry.to_dict(redact=self.redact_on_export)}, duration_ms=elapsed_ms(start))
        except Exception as exc:
            browser_error = wrap_browser_exception(exc, action="memory.put") if "wrap_browser_exception" in globals() else exc
            return error_result(action="memory.put", message="Memory entry could not be stored", error=browser_error, duration_ms=elapsed_ms(start))

    def safe_query(self, query: Optional[Union[MemoryQuery, Mapping[str, Any]]] = None, **overrides: Any) -> Dict[str, Any]:
        start = monotonic_ms()
        try:
            results = self.query_dicts(query, **overrides)
            return success_result(action="memory.query", message="Memory query completed", data={"results": results, "count": len(results)}, duration_ms=elapsed_ms(start))
        except Exception as exc:
            browser_error = wrap_browser_exception(exc, action="memory.query") if "wrap_browser_exception" in globals() else exc
            return error_result(action="memory.query", message="Memory query failed", error=browser_error, duration_ms=elapsed_ms(start))

    # ------------------------------------------------------------------
    # Internal mechanics
    # ------------------------------------------------------------------
    def _get_entry_by_key_unlocked(self, namespace: str, key: str) -> Optional[MemoryEntry]:
        entry_id = self._key_index.get(namespace, {}).get(key)
        if not entry_id:
            return None
        return self._entries.get(namespace, {}).get(entry_id)

    def _resolve_entry_unlocked(self, identifier: str, *, namespace: str) -> Optional[MemoryEntry]:
        ref_namespace, ref_id = split_entry_ref(identifier)
        if ref_namespace and ref_id:
            return self._entries.get(ref_namespace, {}).get(ref_id)
        by_key = self._get_entry_by_key_unlocked(namespace, identifier)
        if by_key is not None:
            return by_key
        if identifier in self._entries.get(namespace, {}):
            return self._entries[namespace][identifier]
        for entries in self._entries.values():
            if identifier in entries:
                return entries[identifier]
        return None

    def _delete_entry_unlocked(self, entry: MemoryEntry, *, event: str) -> None:
        self._deindex_entry(entry)
        self._entries.get(entry.namespace, {}).pop(entry.id, None)
        self._record_event(event, entry=entry)

    def _iter_entries_unlocked(self, namespaces: Optional[Iterable[str]]) -> Iterable[MemoryEntry]:
        namespace_list = [normalize_memory_namespace(ns) for ns in namespaces] if namespaces else list(self._entries.keys())
        for namespace in namespace_list:
            yield from self._entries.get(namespace, {}).values()

    def _entry_matches_query(self, entry: MemoryEntry, query: MemoryQuery) -> bool:
        if entry.is_expired() and not query.include_expired:
            return False
        if query.ids and entry.id not in query.ids:
            return False
        if query.refs and entry.ref not in query.refs:
            return False
        if query.keys:
            if query.exact_key:
                if entry.key not in query.keys:
                    return False
            else:
                key_text = entry.key.lower()
                if not any(key.lower() in key_text for key in query.keys):
                    return False
        if query.kinds and entry.kind not in query.kinds:
            return False
        if query.urls and entry.url not in query.urls:
            return False
        if query.domains and (entry.domain or "").lower() not in query.domains:
            return False
        if query.actions and entry.action not in query.actions:
            return False
        if query.sources and entry.source not in query.sources:
            return False
        if query.min_score is not None and entry.score < query.min_score:
            return False
        if query.tags:
            entry_tags = set(entry.tags)
            query_tags = set(query.tags)
            if query.match_any_tag:
                if not entry_tags.intersection(query_tags):
                    return False
            elif not query_tags.issubset(entry_tags):
                return False
        if query.metadata:
            for key, expected in query.metadata.items():
                actual = entry.metadata.get(key)
                if isinstance(expected, (list, tuple, set)):
                    if actual not in expected:
                        return False
                elif actual != expected:
                    return False
        if not self._within_datetime_bounds(entry.created_at, after=query.created_after, before=query.created_before):
            return False
        if not self._within_datetime_bounds(entry.updated_at, after=query.updated_after, before=query.updated_before):
            return False
        if query.text:
            tokens = tokenize_query(query.text)
            blob = entry_text_blob(entry)
            if not all(token.lower() in blob for token in tokens):
                return False
        return True

    def _within_datetime_bounds(self, value: Optional[str], *, after: Optional[str], before: Optional[str]) -> bool:
        if not after and not before:
            return True
        dt = parse_memory_datetime(value)
        if dt is None:
            return False
        after_dt = parse_memory_datetime(after)
        before_dt = parse_memory_datetime(before)
        if after_dt and dt < after_dt:
            return False
        if before_dt and dt > before_dt:
            return False
        return True

    def _sort_value(self, entry: MemoryEntry, field_name: str) -> Any:
        value = getattr(entry, field_name, None)
        if value is None:
            return "" if field_name in {"created_at", "updated_at", "last_accessed_at", "key", "kind", "namespace"} else 0
        return value

    def _mutate_entry(
        self,
        key_or_id_or_ref: str,
        *,
        namespace: Optional[str],
        event: str,
        custom_mutator: Optional[Any] = None,
        **update_kwargs: Any,
    ) -> Optional[MemoryEntry]:
        self._assert_enabled()
        namespace = normalize_memory_namespace(namespace or self.default_namespace)
        with self._lock:
            entry = self._resolve_entry_unlocked(normalize_whitespace(key_or_id_or_ref), namespace=namespace)
            if entry is None:
                self._counters["misses"] += 1
                return None
            self._deindex_entry(entry)
            if custom_mutator is not None:
                custom_mutator(entry)
            else:
                entry.update(**update_kwargs)
            self._index_entry(entry)
            self._entries[entry.namespace].move_to_end(entry.id)
            self._record_event(event, entry=entry, data=update_kwargs)
            self._maybe_autosave()
            return entry.clone()

    def _merge_values(self, existing: Any, incoming: Any, *, strategy: str) -> Any:
        if strategy == "replace":
            return incoming
        if strategy == "keep_existing":
            return existing
        if strategy == "merge":
            if isinstance(existing, ABCMapping) and isinstance(incoming, ABCMapping):
                return merge_dicts(existing, incoming, deep=True)
            if isinstance(existing, list) and isinstance(incoming, list):
                return [*existing, *incoming]
            return incoming
        if strategy == "append":
            base = existing if isinstance(existing, list) else [existing]
            return [*base, incoming]
        if strategy == "extend":
            base = existing if isinstance(existing, list) else [existing]
            return [*base, *ensure_list(incoming)]
        if strategy == "extend_unique":
            base = existing if isinstance(existing, list) else [existing]
            return dedupe_preserve_order([*base, *ensure_list(incoming)])
        return incoming

    def _enforce_limits_unlocked(self, namespace: str) -> None:
        namespace_entries = self._entries.get(namespace, OrderedDict())
        namespace_limit = self._namespace_limit(namespace)
        while len(namespace_entries) > namespace_limit:
            _, entry = namespace_entries.popitem(last=False)
            self._deindex_entry(entry)
            self._counters["evictions"] += 1
            self._record_event("evict_namespace_lru", entry=entry, data={"limit": namespace_limit})
        while self.total_entries() > self.max_entries:
            oldest = self._oldest_lru_entry_unlocked()
            if oldest is None:
                break
            self._delete_entry_unlocked(oldest, event="evict_global_lru")
            self._counters["evictions"] += 1

    def _enforce_all_limits_unlocked(self) -> None:
        for namespace in list(self._entries.keys()):
            self._enforce_limits_unlocked(namespace)

    def _oldest_lru_entry_unlocked(self) -> Optional[MemoryEntry]:
        oldest: Optional[MemoryEntry] = None
        for entries in self._entries.values():
            if not entries:
                continue
            candidate = next(iter(entries.values()))
            if oldest is None or (candidate.last_accessed_at or candidate.updated_at or candidate.created_at) < (oldest.last_accessed_at or oldest.updated_at or oldest.created_at):
                oldest = candidate
        return oldest

    def _rebuild_indexes_unlocked(self) -> None:
        self._key_index.clear()
        for index in self._indexes.values():
            index.clear()
        for entry in self._iter_entries_unlocked(None):
            self._index_entry(entry)

    def __len__(self) -> int:
        return self.total_entries()

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self.exists(key)

    def __repr__(self) -> str:
        return f"BrowserMemory(entries={self.total_entries()}, namespaces={self.namespaces()})"


if __name__ == "__main__":
    print("\n=== Running Browser Memory ===\n")
    printer.status("TEST", "Browser Memory initialized", "info")

    memory = BrowserMemory(
        config={
            "enabled": True,
            "max_entries": 25,
            "max_entries_per_namespace": 10,
            "auto_prune": True,
            "default_ttl_seconds": None,
            "persistence": {"enabled": False, "autosave": False, "load_on_startup": False},
            "event_log": {"enabled": True, "max_events": 50},
        }
    )
    print(memory)

    print("\n* * * * * Phase 1 - Basic put/get * * * * *\n")
    entry = memory.put(
        "current-page",
        {"url": "https://example.com/docs?utm_source=test", "title": "Example Docs", "text": "Browser memory test page"},
        namespace="pages",
        kind=MemoryKind.PAGE_SNAPSHOT,
        tags=["page", "docs", "test"],
        url="https://example.com/docs?utm_source=test",
        metadata={"title": "Example Docs"},
    )
    print(entry.to_dict(include_value=False))
    assert memory.get("current-page", namespace="pages")["title"] == "Example Docs"

    print("\n* * * * * Phase 2 - Browser-specific records * * * * *\n")
    page_entry = memory.remember_page(
        {
            "url": "https://example.com/articles/browser-memory",
            "title": "Browser Memory Article",
            "text": "A detailed article about browser memory and agentic browsing.",
        },
        tags=["article"],
    )
    print(page_entry.to_dict(include_value=False))

    action_entry = memory.remember_action(
        "navigate",
        {"status": "success", "action": "navigate", "url": "https://example.com/articles/browser-memory"},
        request={"task": "navigate", "url": "https://example.com/articles/browser-memory"},
    )
    print(action_entry.to_dict(include_value=False))

    search_entries = memory.remember_search_results(
        "browser memory selenium",
        [
            {"link": "https://example.com/articles/browser-memory", "title": "Browser Memory Article", "score": 0.95},
            {"link": "https://example.com/docs", "title": "Example Docs", "score": 0.75},
        ],
    )
    print(f"Stored search entries: {len(search_entries)}")

    print("\n* * * * * Phase 3 - Query and recall * * * * *\n")
    docs = memory.query_dicts({"tag": "docs", "limit": 5, "include_value": False})
    print(docs)
    assert docs

    recalled = memory.recall_for_task({"task": "navigate", "url": "https://example.com/articles/browser-memory"}, limit=5)
    print(recalled)
    assert recalled

    print("\n* * * * * Phase 4 - Merge, stats, snapshot * * * * *\n")
    memory.merge("current-page", {"observations": ["loaded", "readable"]}, namespace="pages", strategy="merge")
    stats = memory.stats().to_dict()
    print(stats)
    assert stats["entries_total"] >= 4

    snapshot = memory.snapshot(include_values=True)
    print({"snapshot_entries": len(snapshot["entries"]), "events": len(snapshot["events"])})
    assert snapshot["entries"]

    print("\n* * * * * Phase 5 - Expiry and maintenance * * * * *\n")
    memory.put("short-lived", {"ok": True}, namespace="scratch", ttl_seconds=0.01)
    import time as _time
    _time.sleep(0.02)
    removed = memory.prune_expired()
    print({"expired_removed": removed})
    assert removed >= 1

    print("\n=== Test ran successfully ===\n")