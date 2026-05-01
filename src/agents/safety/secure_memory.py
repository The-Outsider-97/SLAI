"""
Production-grade secure memory for the Safety Agent subsystem.

This module centralizes short-lived and checkpointable memory used by safety,
security, compliance, and model-governance components. It intentionally keeps
memory persistence policy inside SecureMemory while delegating reusable helper
concerns to safety_helpers and structured incident handling to security_error.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile

from collections import defaultdict, OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.safety_helpers import *
from .utils.security_error import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Secure Memory")
printer = PrettyPrinter()

MODULE_VERSION = "2.1.0"
CHECKPOINT_SCHEMA_VERSION = "secure_memory.checkpoint.v3"
ENTRY_SCHEMA_VERSION = "secure_memory.entry.v3"
AUDIT_SCHEMA_VERSION = "secure_memory.audit.v2"


@dataclass
class MemoryMetadata:
    """Structured metadata stored with every memory entry."""

    entry_id: str
    created_at: str
    updated_at: str
    last_accessed_at: Optional[str]
    expires_at: Optional[str]
    ttl_seconds: Optional[int]
    tags: List[str]
    sensitivity: float
    relevance: float
    purpose: str
    owner: str
    classification: str
    source: str
    status: str
    data_type: str
    data_size_bytes: int
    data_fingerprint: str
    metadata_fingerprint: str
    access_count: int = 0
    revision: int = 1
    legal_hold: bool = False
    schema_version: str = ENTRY_SCHEMA_VERSION
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AccessDecision:
    """Result of secure access policy evaluation."""

    allowed: bool
    reason: str
    action: str
    required_level: int
    supplied_level: int
    principal: str
    purpose: str
    entry_id: Optional[str] = None
    missing_fields: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryAuditEvent:
    """Audit-safe memory event envelope."""

    event_id: str
    timestamp: str
    event_type: str
    action: str
    allowed: bool
    reason: str
    entry_id: Optional[str]
    principal: str
    purpose: str
    context: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SecureMemory:
    """
    Secure in-memory store with retention, access control, audit logging,
    relevance-aware eviction, and signed checkpoint support.

    Existing subsystem calls remain supported:
    - add(entry, tags=None, sensitivity=1.0)
    - get(entry_id, access_context)
    - recall(tag, top_k=None)
    - search_secure(query, tag_filter=None)
    - create_checkpoint(name=None)
    - load_checkpoint(path)
    - get_statistics()
    - audit_access(max_results=100)
    - sanitize_memory(tag=None)
    - update_relevance(entry_id, relevance)
    - bootstrap_if_empty()
    """

    def __init__(self):
        self.config = load_global_config()
        self.memory_config = get_config_section("secure_memory")
        self.complience_config = get_config_section("compliance_checker")
        self.phishing_model_path = self.complience_config.get("phishing_model_path")
        self._setup_defaults()
        self._validate_configuration()

        self.store: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.purpose_index: Dict[str, Set[str]] = defaultdict(set)
        self.owner_index: Dict[str, Set[str]] = defaultdict(set)
        self.relevance_scores: Dict[str, float] = {}
        self.access_log: List[Dict[str, Any]] = []
        self.lock = RLock()

        self.stats: Dict[str, Any] = {
            "entries_created": 0,
            "entries_updated": 0,
            "entries_deleted": 0,
            "entries_sanitized": 0,
            "expired_entries": 0,
            "access_count": 0,
            "access_allowed": 0,
            "access_denied": 0,
            "misses": 0,
            "evictions": 0,
            "failed_checks": 0,
            "checkpoint_success": 0,
            "checkpoint_failure": 0,
            "checkpoint_loads": 0,
            "search_count": 0,
            "recall_count": 0,
            "audit_events": 0,
        }

        checkpoint_dir = self._checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info("SecureMemory initialized with %s policy", self._cfg("eviction_policy", "LRU"))

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _setup_defaults(self) -> None:
        """Retain config-loader behavior while ensuring required keys exist."""

        self.memory_config = dict(self.memory_config or {})
        self.memory_config.setdefault("max_size", 5000)
        self.memory_config.setdefault("eviction_policy", "LRU")
        self.memory_config.setdefault("checkpoint_dir", "src/agents/safety/checkpoints/secure_checkpoints")
        self.memory_config.setdefault("checkpoint_freq", 1000)
        self.memory_config.setdefault("relevance_decay", 0.95)
        self.memory_config.setdefault("min_relevance", 0.1)
        self.memory_config.setdefault("max_access_log", 10000)
        self.memory_config.setdefault("default_ttl_seconds", 86400)
        self.memory_config.setdefault("min_sensitivity", 0.0)
        self.memory_config.setdefault("max_sensitivity", 1.0)
        self.memory_config.setdefault("default_tags", ["security", "sensitive"])
        self.memory_config.setdefault("checkpoint_security", {})
        self.memory_config.setdefault("access_validation", {})

    def _cfg(self, path: Union[str, Sequence[str]], default: Any = None) -> Any:
        return get_nested(self.memory_config, path, default)

    def _validate_configuration(self) -> None:
        max_size = coerce_int(self._cfg("max_size"), 0)
        if max_size <= 0:
            raise ConfigurationTamperingError(
                "secure_memory.max_size",
                "max_size must be a positive integer",
                component="secure_memory",
            )

        policy = str(self._cfg("eviction_policy", "LRU")).upper()
        allowed_policies = {"LRU", "FIFO", "LFU", "LEAST_RELEVANT"}
        if policy not in allowed_policies:
            raise ConfigurationTamperingError(
                "secure_memory.eviction_policy",
                f"Unsupported eviction policy: {policy}",
                component="secure_memory",
            )

        checkpoint_mode = str(self._cfg("checkpoint_security.data_mode", "redacted")).lower()
        if checkpoint_mode not in {"metadata_only", "redacted", "full"}:
            raise ConfigurationTamperingError(
                "secure_memory.checkpoint_security.data_mode",
                f"Unsupported checkpoint data mode: {checkpoint_mode}",
                component="secure_memory",
            )

        safe_hash_algorithm(str(self._cfg("checkpoint_security.hash_algorithm", get_helper_setting("hash_algorithm", "sha256"))))

    def _checkpoint_dir(self) -> Path:
        return Path(str(self._cfg("checkpoint_dir"))).expanduser()

    def _checkpoint_salt(self) -> str:
        return str(self._cfg("checkpoint_security.signature_salt", get_helper_setting("hash_salt", "")))

    def _internal_context(self, action: str = "internal") -> Dict[str, Any]:
        access_cfg = self._cfg("access_validation", {}) or {}
        return {
            "auth_token": "internal_secure_memory_context",
            "access_level": coerce_int(access_cfg.get("internal_access_level", access_cfg.get("min_access_level", 0)), 0),
            "purpose": action,
            "principal": "secure_memory",
            "component": "secure_memory",
            "request_id": generate_request_id(),
        }

    # ------------------------------------------------------------------
    # Entry lifecycle
    # ------------------------------------------------------------------

    def add(
        self,
        entry: Any,
        tags: Optional[List[str]] = None,
        sensitivity: float = 1.0,
        *,
        ttl_seconds: Optional[int] = None,
        purpose: Optional[str] = None,
        owner: Optional[str] = None,
        classification: Optional[str] = None,
        source: str = "runtime",
        metadata: Optional[Mapping[str, Any]] = None,
        entry_id: Optional[str] = None,
    ) -> str:
        """Add an entry with security metadata and return its entry ID."""

        with self.lock:
            self._purge_expired_locked()
            normalized_tags = self._normalize_tags(tags)
            normalized_sensitivity = self._normalize_sensitivity(sensitivity)
            normalized_ttl = self._normalize_ttl(ttl_seconds, normalized_sensitivity)
            now = utc_iso()
            expires_at = self._calculate_expiration(normalized_ttl)
            normalized_entry_id = self._new_entry_id(entry_id)
            data_fingerprint = self._fingerprint_entry_data(entry)
            data_size = len(stable_json(redact_value(entry)).encode("utf-8", errors="replace"))

            entry_metadata = MemoryMetadata(
                entry_id=normalized_entry_id,
                created_at=now,
                updated_at=now,
                last_accessed_at=None,
                expires_at=expires_at,
                ttl_seconds=normalized_ttl,
                tags=normalized_tags,
                sensitivity=normalized_sensitivity,
                relevance=1.0,
                purpose=normalize_text(purpose or str(self._cfg("default_purpose", "safety_security_memory")), max_length=128),
                owner=normalize_text(owner or str(self._cfg("default_owner", "safety_agent")), max_length=128),
                classification=normalize_text(classification or self._classification_for_sensitivity(normalized_sensitivity), max_length=64),
                source=normalize_text(source, max_length=128),
                status="active",
                data_type=type(entry).__name__,
                data_size_bytes=data_size,
                data_fingerprint=data_fingerprint,
                metadata_fingerprint="pending",
                extra=redact_value(dict(metadata or {})),
            )
            entry_metadata.metadata_fingerprint = self._fingerprint_metadata(entry_metadata)

            self.store[normalized_entry_id] = {
                "data": copy.deepcopy(entry),
                "meta": entry_metadata.to_dict(),
            }
            self.relevance_scores[normalized_entry_id] = entry_metadata.relevance
            self._index_entry_locked(normalized_entry_id, entry_metadata.to_dict())
            self.stats["entries_created"] += 1
            self.stats["access_count"] += 1

            self._record_audit_event_locked(
                event_type="memory.add",
                action="add",
                allowed=True,
                reason="entry_created",
                entry_id=normalized_entry_id,
                context=self._internal_context("add"),
                metadata={"tags": normalized_tags, "sensitivity": normalized_sensitivity, "purpose": entry_metadata.purpose},
            )
            self._manage_capacity_locked()
            self._maybe_checkpoint_locked()
            return normalized_entry_id

    def get(self, entry_id: str, access_context: Dict) -> Optional[Dict[str, Any]]:
        """Secure retrieval with access validation and audit logging."""

        with self.lock:
            self._purge_expired_locked()
            normalized_entry_id = normalize_identifier(entry_id, max_length=128)
            entry = self.store.get(normalized_entry_id)
            if entry is None:
                self.stats["misses"] += 1
                self.stats["failed_checks"] += 1
                self._record_audit_event_locked(
                    event_type="memory.get",
                    action="get",
                    allowed=False,
                    reason="entry_not_found",
                    entry_id=normalized_entry_id,
                    context=access_context or {},
                )
                return None

            decision = self._evaluate_access(access_context or {}, entry=entry, action="get")
            if not decision.allowed:
                self._handle_access_denied_locked(decision, normalized_entry_id, access_context or {})
                return None

            self._touch_entry_locked(normalized_entry_id, entry)
            self.stats["access_count"] += 1
            self.stats["access_allowed"] += 1
            self._record_audit_event_locked(
                event_type="memory.get",
                action="get",
                allowed=True,
                reason=decision.reason,
                entry_id=normalized_entry_id,
                context=access_context or {},
                metadata={"access_count": entry["meta"].get("access_count", 0)},
            )
            return copy.deepcopy(entry)

    def update(
        self,
        entry_id: str,
        entry: Any,
        access_context: Optional[Dict[str, Any]] = None,
        *,
        tags: Optional[List[str]] = None,
        sensitivity: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Update an existing entry while preserving auditability."""

        with self.lock:
            normalized_entry_id = normalize_identifier(entry_id, max_length=128)
            current = self.store.get(normalized_entry_id)
            if current is None:
                self.stats["misses"] += 1
                return False

            context = access_context or self._internal_context("update")
            decision = self._evaluate_access(context, entry=current, action="update")
            if not decision.allowed:
                self._handle_access_denied_locked(decision, normalized_entry_id, context)
                return False

            old_meta = current["meta"]
            self._deindex_entry_locked(normalized_entry_id, old_meta)
            new_tags = self._normalize_tags(tags if tags is not None else list(old_meta.get("tags", [])))
            new_sensitivity = self._normalize_sensitivity(old_meta.get("sensitivity", 1.0) if sensitivity is None else sensitivity)
            new_ttl = self._normalize_ttl(ttl_seconds if ttl_seconds is not None else old_meta.get("ttl_seconds"), new_sensitivity)
            now = utc_iso()
            current["data"] = copy.deepcopy(entry)
            current["meta"].update({
                "updated_at": now,
                "expires_at": self._calculate_expiration(new_ttl),
                "ttl_seconds": new_ttl,
                "tags": new_tags,
                "sensitivity": new_sensitivity,
                "classification": self._classification_for_sensitivity(new_sensitivity),
                "data_type": type(entry).__name__,
                "data_size_bytes": len(stable_json(redact_value(entry)).encode("utf-8", errors="replace")),
                "data_fingerprint": self._fingerprint_entry_data(entry),
                "revision": coerce_int(old_meta.get("revision", 1), 1) + 1,
                "extra": redact_value(dict(metadata or old_meta.get("extra", {}))),
            })
            current["meta"]["metadata_fingerprint"] = self._fingerprint_metadata(current["meta"])
            self._index_entry_locked(normalized_entry_id, current["meta"])
            self.stats["entries_updated"] += 1
            self._record_audit_event_locked(
                event_type="memory.update",
                action="update",
                allowed=True,
                reason="entry_updated",
                entry_id=normalized_entry_id,
                context=context,
                metadata={"revision": current["meta"].get("revision")},
            )
            return True

    def delete(self, entry_id: str, access_context: Optional[Dict[str, Any]] = None, *, reason: str = "delete_requested") -> bool:
        """Remove an entry and emit an audit-safe deletion event."""

        with self.lock:
            normalized_entry_id = normalize_identifier(entry_id, max_length=128)
            entry = self.store.get(normalized_entry_id)
            if entry is None:
                self.stats["misses"] += 1
                return False
            context = access_context or self._internal_context("delete")
            decision = self._evaluate_access(context, entry=entry, action="delete")
            if not decision.allowed:
                self._handle_access_denied_locked(decision, normalized_entry_id, context)
                return False
            self._remove_entry_locked(normalized_entry_id, reason=reason)
            return True

    def recall(self, tag: str, top_k: Optional[int] = None, access_context: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Retrieve entries by tag with optional limit, sorted by relevance."""

        with self.lock:
            self._purge_expired_locked()
            normalized_tag = normalize_identifier(tag, max_length=96)
            context = access_context or self._internal_context("recall")
            entries: List[Dict[str, Any]] = []
            for entry_id in list(self.tag_index.get(normalized_tag, set())):
                entry = self.store.get(entry_id)
                if not entry:
                    continue
                decision = self._evaluate_access(context, entry=entry, action="recall")
                if decision.allowed:
                    self._touch_entry_locked(entry_id, entry, move_lru=False)
                    entries.append(copy.deepcopy(entry))
                else:
                    self.stats["failed_checks"] += 1

            entries.sort(key=lambda e: coerce_float(e.get("meta", {}).get("relevance", 0.0)), reverse=True)
            self.stats["recall_count"] += 1
            limited = entries[:top_k] if top_k else entries
            self._record_audit_event_locked(
                event_type="memory.recall",
                action="recall",
                allowed=True,
                reason="tag_recall_completed",
                entry_id=None,
                context=context,
                metadata={"tag": normalized_tag, "result_count": len(limited)},
            )
            return limited

    # ------------------------------------------------------------------
    # Access control and audit
    # ------------------------------------------------------------------

    def _validate_access(self, context: Dict) -> bool:
        """Backwards-compatible boolean access validation."""

        return self._evaluate_access(context or {}, entry=None, action="validate").allowed

    def _evaluate_access(self, context: Mapping[str, Any], *, entry: Optional[Mapping[str, Any]], action: str) -> AccessDecision:
        access_cfg = self._cfg("access_validation", {}) or {}
        required_fields = list(access_cfg.get("required_fields", []))
        min_level = coerce_int(access_cfg.get("min_access_level", 0), 0)
        supplied_level = coerce_int(context.get("access_level", 0), 0)
        principal = normalize_text(context.get("principal") or context.get("user_id") or context.get("actor_id") or "unknown", max_length=128)
        purpose = normalize_text(context.get("purpose") or context.get("action") or action, max_length=128)
        missing = tuple(field for field in required_fields if field not in context or context.get(field) in (None, ""))

        if missing:
            return AccessDecision(False, "missing_required_access_context", action, min_level, supplied_level, principal, purpose, missing_fields=missing)
        if supplied_level < min_level:
            return AccessDecision(False, "insufficient_access_level", action, min_level, supplied_level, principal, purpose)

        allowed_purposes = set(str(item) for item in access_cfg.get("allowed_purposes", []) or [])
        if allowed_purposes and purpose not in allowed_purposes:
            return AccessDecision(False, "purpose_not_allowed", action, min_level, supplied_level, principal, purpose)

        if entry is not None:
            meta = entry.get("meta", {}) if isinstance(entry, Mapping) else {}
            sensitivity = coerce_float(meta.get("sensitivity", 1.0), 1.0, minimum=0.0, maximum=1.0)
            sensitivity_levels = self._cfg("sensitivity_access_levels", {}) or {}
            required_for_sensitivity = coerce_int(
                sensitivity_levels.get(self._classification_for_sensitivity(sensitivity), min_level),
                min_level,
            )
            if supplied_level < required_for_sensitivity:
                return AccessDecision(False, "insufficient_sensitivity_access_level", action, required_for_sensitivity, supplied_level, principal, purpose, entry_id=meta.get("entry_id"))
            if meta.get("status") in {"sanitized", "deleted", "expired"}:
                return AccessDecision(False, f"entry_status_{meta.get('status')}", action, min_level, supplied_level, principal, purpose, entry_id=meta.get("entry_id"))

        return AccessDecision(True, "access_allowed", action, min_level, supplied_level, principal, purpose, entry_id=(entry or {}).get("meta", {}).get("entry_id") if isinstance(entry, Mapping) else None)

    def _handle_access_denied_locked(self, decision: AccessDecision, entry_id: str, access_context: Mapping[str, Any]) -> None:
        self.stats["access_denied"] += 1
        self.stats["failed_checks"] += 1
        self._record_audit_event_locked(
            event_type="memory.access_denied",
            action=decision.action,
            allowed=False,
            reason=decision.reason,
            entry_id=entry_id,
            context=access_context,
            metadata=decision.to_dict(),
        )
        logger.warning("Unauthorized secure memory access: %s", stable_json(safe_log_payload("secure_memory.access_denied", decision.to_dict())))
        if coerce_bool(self._cfg("access_validation.raise_on_denied", False)):
            raise UnauthorizedAccessError(
                resource=f"secure_memory:{entry_id}",
                policy_violated=decision.reason,
                attempted_action=decision.action,
                user_id=decision.principal,
                component="secure_memory",
                context={"decision": decision.to_dict(), "access_context": sanitize_for_logging(access_context)},
            )

    def _record_audit_event_locked(
        self,
        *,
        event_type: str,
        action: str,
        allowed: bool,
        reason: str,
        entry_id: Optional[str],
        context: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        try:
            event = MemoryAuditEvent(
                event_id=generate_identifier("mem_evt"),
                timestamp=utc_iso(),
                event_type=normalize_text(event_type, max_length=96),
                action=normalize_text(action, max_length=64),
                allowed=bool(allowed),
                reason=normalize_text(reason, max_length=160),
                entry_id=normalize_identifier(entry_id, max_length=128) if entry_id else None,
                principal=normalize_text(context.get("principal") or context.get("user_id") or context.get("actor_id") or "unknown", max_length=128),
                purpose=normalize_text(context.get("purpose") or action, max_length=128),
                context=sanitize_for_logging(dict(context)),
                metadata=sanitize_for_logging(dict(metadata or {})),
            )
            self.access_log.append(event.to_dict())
            max_log = coerce_int(self._cfg("max_access_log", 10000), 10000, minimum=100)
            if len(self.access_log) > max_log:
                del self.access_log[: len(self.access_log) - max_log]
            self.stats["audit_events"] += 1
        except Exception as exc:
            self.stats["failed_checks"] += 1
            raise AuditLogFailureError(
                "secure_memory.access_log",
                f"Failed to append audit event: {type(exc).__name__}",
                component="secure_memory",
                cause=exc,
            )

    # ------------------------------------------------------------------
    # Indexing, retention, and eviction
    # ------------------------------------------------------------------

    def _normalize_tags(self, tags: Optional[Iterable[str]]) -> List[str]:
        defaults = self._cfg("default_tags", []) or []
        combined = list(defaults) + list(tags or [])
        return dedupe_preserve_order(normalize_identifier(tag, max_length=96) for tag in combined if str(tag).strip())

    def _normalize_sensitivity(self, sensitivity: Any) -> float:
        minimum = coerce_float(self._cfg("min_sensitivity", 0.0), 0.0, minimum=0.0, maximum=1.0)
        maximum = coerce_float(self._cfg("max_sensitivity", 1.0), 1.0, minimum=minimum, maximum=1.0)
        return coerce_float(sensitivity, maximum, minimum=minimum, maximum=maximum)

    def _normalize_ttl(self, ttl_seconds: Optional[int], sensitivity: float) -> Optional[int]:
        if ttl_seconds is None:
            ttl_seconds = self._cfg("retention.sensitivity_ttl_seconds.restricted" if sensitivity >= 0.85 else "default_ttl_seconds", None)
        if ttl_seconds is None:
            return None
        ttl = coerce_int(ttl_seconds, 0, minimum=0)
        max_ttl = self._cfg("retention.max_ttl_seconds", None)
        if max_ttl is not None:
            ttl = min(ttl, coerce_int(max_ttl, ttl, minimum=0))
        return ttl if ttl > 0 else None

    def _calculate_expiration(self, ttl_seconds: Optional[int]) -> Optional[str]:
        if not ttl_seconds:
            return None
        return utc_iso(utc_now().timestamp() + ttl_seconds)

    def _is_expired(self, meta: Mapping[str, Any]) -> bool:
        if meta.get("legal_hold"):
            return False
        expires_at = meta.get("expires_at")
        if not expires_at:
            return False
        return seconds_until(parse_iso_datetime(str(expires_at))) <= 0.0

    def _purge_expired_locked(self) -> None:
        expired = [entry_id for entry_id, entry in self.store.items() if self._is_expired(entry.get("meta", {}))]
        for entry_id in expired:
            self._remove_entry_locked(entry_id, reason="expired")
            self.stats["expired_entries"] += 1

    def _index_entry_locked(self, entry_id: str, meta: Mapping[str, Any]) -> None:
        for tag in meta.get("tags", []) or []:
            self.tag_index[normalize_identifier(tag, max_length=96)].add(entry_id)
        self.purpose_index[normalize_identifier(meta.get("purpose", "unknown"), max_length=96)].add(entry_id)
        self.owner_index[normalize_identifier(meta.get("owner", "unknown"), max_length=96)].add(entry_id)

    def _deindex_entry_locked(self, entry_id: str, meta: Mapping[str, Any]) -> None:
        for tag in meta.get("tags", []) or []:
            normalized_tag = normalize_identifier(tag, max_length=96)
            self.tag_index.get(normalized_tag, set()).discard(entry_id)
            if normalized_tag in self.tag_index and not self.tag_index[normalized_tag]:
                del self.tag_index[normalized_tag]
        for index_name, index in (("purpose", self.purpose_index), ("owner", self.owner_index)):
            normalized_value = normalize_identifier(meta.get(index_name, "unknown"), max_length=96)
            index.get(normalized_value, set()).discard(entry_id)
            if normalized_value in index and not index[normalized_value]:
                del index[normalized_value]

    def _manage_capacity_locked(self) -> None:
        max_size = coerce_int(self._cfg("max_size", 5000), 5000, minimum=1)
        while len(self.store) > max_size:
            candidate = self._select_eviction_candidate_locked()
            if candidate is None:
                break
            self._remove_entry_locked(candidate, reason="evicted")
            self.stats["evictions"] += 1

    def _select_eviction_candidate_locked(self) -> Optional[str]:
        if not self.store:
            return None
        policy = str(self._cfg("eviction_policy", "LRU")).upper()
        candidates = [(entry_id, entry) for entry_id, entry in self.store.items() if not entry.get("meta", {}).get("legal_hold")]
        if not candidates:
            return None
        if policy == "FIFO":
            return candidates[0][0]
        if policy == "LFU":
            return min(candidates, key=lambda item: coerce_int(item[1].get("meta", {}).get("access_count", 0), 0))[0]
        if policy == "LEAST_RELEVANT":
            return min(candidates, key=lambda item: coerce_float(item[1].get("meta", {}).get("relevance", 0.0), 0.0))[0]
        return candidates[0][0]

    def _remove_entry_locked(self, entry_id: str, *, reason: str) -> None:
        entry = self.store.get(entry_id)
        if not entry:
            return
        meta = entry.get("meta", {})
        self._deindex_entry_locked(entry_id, meta)
        entry["data"] = None
        entry["meta"]["status"] = reason
        self.store.pop(entry_id, None)
        self.relevance_scores.pop(entry_id, None)
        self.stats["entries_deleted"] += 1
        self._record_audit_event_locked(
            event_type="memory.remove",
            action="remove",
            allowed=True,
            reason=reason,
            entry_id=entry_id,
            context=self._internal_context("remove"),
            metadata={"reason": reason, "tags": meta.get("tags", [])},
        )

    def _touch_entry_locked(self, entry_id: str, entry: MutableMapping[str, Any], *, move_lru: bool = True) -> None:
        now = utc_iso()
        meta = entry["meta"]
        meta["access_count"] = coerce_int(meta.get("access_count", 0), 0) + 1
        meta["last_accessed_at"] = now
        decay = coerce_float(self._cfg("relevance_decay", 0.95), 0.95, minimum=0.0, maximum=1.0)
        current_relevance = coerce_float(meta.get("relevance", 1.0), 1.0, minimum=0.0, maximum=1.0)
        meta["relevance"] = min(1.0, max(current_relevance * decay, coerce_float(self._cfg("min_relevance", 0.1), 0.1)))
        meta["metadata_fingerprint"] = self._fingerprint_metadata(meta)
        self.relevance_scores[entry_id] = meta["relevance"]
        if move_lru and str(self._cfg("eviction_policy", "LRU")).upper() == "LRU":
            self.store.move_to_end(entry_id)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def create_checkpoint(self, name: Optional[str] = None) -> bool:
        """Create a signed checkpoint according to secure_config.yaml policy."""

        with self.lock:
            checkpoint_name = name or f"secure_memory_{utc_now().strftime('%Y%m%d_%H%M%S')}.json"
            checkpoint_path = self._checkpoint_dir() / normalize_text(checkpoint_name, max_length=160)
            if checkpoint_path.suffix == "":
                checkpoint_path = checkpoint_path.with_suffix(".json")
            try:
                payload = self._checkpoint_payload_locked()
                signature = self._sign_payload(payload)
                envelope = {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "module_version": MODULE_VERSION,
                    "created_at": utc_iso(),
                    "signature_algorithm": safe_hash_algorithm(str(self._cfg("checkpoint_security.hash_algorithm", get_helper_setting("hash_algorithm", "sha256")))),
                    "payload_fingerprint": fingerprint(payload),
                    "signature": signature,
                    "payload": payload,
                }
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(checkpoint_path.parent), delete=False) as tmp:
                    json.dump(envelope, tmp, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    tmp.flush()
                    os.fsync(tmp.fileno())
                    tmp_path = Path(tmp.name)
                os.replace(tmp_path, checkpoint_path)
                self.stats["checkpoint_success"] += 1
                self._record_audit_event_locked(
                    event_type="memory.checkpoint_create",
                    action="create_checkpoint",
                    allowed=True,
                    reason="checkpoint_created",
                    entry_id=None,
                    context=self._internal_context("checkpoint_create"),
                    metadata={"path": str(checkpoint_path), "payload_fingerprint": envelope["payload_fingerprint"]},
                )
                logger.info("Secure checkpoint created: %s", checkpoint_path)
                return True
            except SecurityError:
                self.stats["checkpoint_failure"] += 1
                raise
            except Exception as exc:
                self.stats["checkpoint_failure"] += 1
                logger.error("Checkpoint failed: %s", redact_text(str(exc)))
                if coerce_bool(self._cfg("checkpoint_security.raise_on_failure", False)):
                    raise SystemIntegrityError(
                        "secure_memory.checkpoint",           # component
                        "Checkpoint creation failed",         # anomaly_description
                        actual_state=str(exc),
                        cause=exc,
                    ) from exc
                return False

    def load_checkpoint(self, path: str) -> bool:
        """Load a signed checkpoint after schema and signature verification."""

        checkpoint_path = Path(path).expanduser()
        try:
            raw = load_text_file(checkpoint_path, max_bytes=coerce_int(self._cfg("checkpoint_security.max_checkpoint_bytes", 50_000_000), 50_000_000))
            envelope = parse_json_object(raw, context="secure_memory_checkpoint")
            if envelope.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                raise SystemIntegrityError(
                    "secure_memory.checkpoint",
                    "Unsupported or missing checkpoint schema version",
                    expected_state=CHECKPOINT_SCHEMA_VERSION,
                    actual_state=str(envelope.get("schema_version")),
                    component="secure_memory", # pyright: ignore[reportCallIssue]
                )
            payload = envelope.get("payload")
            expected_signature = str(envelope.get("signature", ""))
            actual_signature = self._sign_payload(payload)
            if coerce_bool(self._cfg("checkpoint_security.verify_signature", True)) and not constant_time_equals(expected_signature, actual_signature):
                raise SystemIntegrityError(
                    "secure_memory.checkpoint",
                    "Checkpoint signature verification failed",
                    expected_state="valid_signature",
                    actual_state="invalid_signature",
                )

            with self.lock:
                self._restore_payload_locked(payload) # pyright: ignore[reportArgumentType]
                self.stats["checkpoint_loads"] += 1
                self._record_audit_event_locked(
                    event_type="memory.checkpoint_load",
                    action="load_checkpoint",
                    allowed=True,
                    reason="checkpoint_loaded",
                    entry_id=None,
                    context=self._internal_context("checkpoint_load"),
                    metadata={"path": str(checkpoint_path), "entries": len(self.store)},
                )
            logger.info("Loaded secure checkpoint from %s", checkpoint_path)
            return True
        except SecurityError:
            raise
        except Exception as exc:
            logger.error("Checkpoint load failed: %s", redact_text(str(exc)))
            if coerce_bool(self._cfg("checkpoint_security.raise_on_failure", True)):
                raise SystemIntegrityError(
                    "secure_memory.checkpoint",           # component
                    "Checkpoint load failed",             # anomaly_description
                    actual_state=str(exc),
                    cause=exc,
                ) from exc
            return False

    def _checkpoint_payload_locked(self) -> Dict[str, Any]:
        data_mode = str(self._cfg("checkpoint_security.data_mode", "redacted")).lower()
        entries: Dict[str, Any] = {}
        for entry_id, entry in self.store.items():
            meta = copy.deepcopy(entry.get("meta", {}))
            if data_mode == "metadata_only":
                data = None
            elif data_mode == "full":
                data = to_jsonable(entry.get("data"))
            else:
                data = redact_value(entry.get("data"))
            entries[entry_id] = {"data": data, "meta": redact_value(meta)}
        return {
            "entries": entries,
            "stats": redact_value(self.stats),
            "created_at": utc_iso(),
            "entry_count": len(entries),
            "config_fingerprint": fingerprint(self.memory_config),
            "data_mode": data_mode,
        }

    def _restore_payload_locked(self, payload: Mapping[str, Any]) -> None:
        entries = payload.get("entries", {}) if isinstance(payload, Mapping) else {}
        if not isinstance(entries, Mapping):
            raise SystemIntegrityError(
                "secure_memory.checkpoint",
                "Checkpoint entries payload is not a mapping",
                component="secure_memory", # pyright: ignore[reportCallIssue]
            )
        self.store = OrderedDict()
        self.tag_index = defaultdict(set)
        self.purpose_index = defaultdict(set)
        self.owner_index = defaultdict(set)
        self.relevance_scores = {}
        for raw_entry_id, raw_entry in entries.items():
            entry_id = normalize_identifier(raw_entry_id, max_length=128)
            if not isinstance(raw_entry, Mapping):
                continue
            meta = dict(raw_entry.get("meta", {}))
            meta.setdefault("entry_id", entry_id)
            meta.setdefault("status", "active")
            meta.setdefault("tags", [])
            data = raw_entry.get("data")
            self.store[entry_id] = {"data": data, "meta": meta}
            self.relevance_scores[entry_id] = coerce_float(meta.get("relevance", 1.0), 1.0, minimum=0.0, maximum=1.0)
            self._index_entry_locked(entry_id, meta)
        restored_stats = payload.get("stats", {}) if isinstance(payload, Mapping) else {}
        if isinstance(restored_stats, Mapping):
            self.stats.update({k: v for k, v in restored_stats.items() if k in self.stats})

    def _sign_payload(self, payload: Any) -> str:
        algorithm = str(self._cfg("checkpoint_security.hash_algorithm", get_helper_setting("hash_algorithm", "sha256")))
        return hash_text(stable_json(payload), algorithm=algorithm, salt=self._checkpoint_salt())

    def _maybe_checkpoint_locked(self) -> None:
        frequency = coerce_int(self._cfg("checkpoint_freq", 0), 0, minimum=0)
        if frequency and self.stats["entries_created"] % frequency == 0:
            self.create_checkpoint()

    # ------------------------------------------------------------------
    # Query, statistics, and maintenance
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return audit-safe security statistics and memory metrics."""

        with self.lock:
            relevance_values = list(self.relevance_scores.values())
            avg_relevance = sum(relevance_values) / len(relevance_values) if relevance_values else 0.0
            sensitivity_values = [coerce_float(entry.get("meta", {}).get("sensitivity", 0.0), 0.0) for entry in self.store.values()]
            avg_sensitivity = sum(sensitivity_values) / len(sensitivity_values) if sensitivity_values else 0.0
            return sanitize_for_logging({
                "schema_version": AUDIT_SCHEMA_VERSION,
                "module_version": MODULE_VERSION,
                "total_entries": len(self.store),
                "active_tags": len(self.tag_index),
                "active_purposes": len(self.purpose_index),
                "active_owners": len(self.owner_index),
                "avg_relevance": avg_relevance,
                "avg_sensitivity": avg_sensitivity,
                "security_stats": copy.deepcopy(self.stats),
                "checkpoint_dir": str(self._checkpoint_dir()),
                "eviction_policy": self._cfg("eviction_policy"),
            })

    def audit_access(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """Return recent audit events with sensitive fields redacted."""

        with self.lock:
            limit = coerce_int(max_results, 100, minimum=1, maximum=coerce_int(self._cfg("max_access_log", 10000), 10000))
            return copy.deepcopy(self.access_log[-limit:])

    def sanitize_memory(self, tag: Optional[str] = None, access_context: Optional[Dict[str, Any]] = None) -> int:
        """Sanitize data payloads by tag or across all memory entries."""

        with self.lock:
            context = access_context or self._internal_context("sanitize_memory")
            targets = list(self.tag_index.get(normalize_identifier(tag, max_length=96), set())) if tag else list(self.store.keys())
            count = 0
            for entry_id in targets:
                entry = self.store.get(entry_id)
                if not entry:
                    continue
                decision = self._evaluate_access(context, entry=entry, action="sanitize")
                if not decision.allowed:
                    self._handle_access_denied_locked(decision, entry_id, context)
                    continue
                entry["data"] = None
                entry["meta"]["status"] = "sanitized"
                entry["meta"]["updated_at"] = utc_iso()
                entry["meta"]["data_fingerprint"] = fingerprint(None)
                entry["meta"]["metadata_fingerprint"] = self._fingerprint_metadata(entry["meta"])
                self.relevance_scores[entry_id] = 0.0
                count += 1
            self.stats["entries_sanitized"] += count
            self._record_audit_event_locked(
                event_type="memory.sanitize",
                action="sanitize",
                allowed=True,
                reason="sanitize_completed",
                entry_id=None,
                context=context,
                metadata={"tag": tag, "sanitized_count": count},
            )
            return count

    def update_relevance(self, entry_id: str, relevance: float) -> bool:
        """Manually adjust an entry relevance score."""

        with self.lock:
            normalized_entry_id = normalize_identifier(entry_id, max_length=128)
            if normalized_entry_id not in self.store:
                return False
            score = coerce_float(relevance, 0.0, minimum=0.0, maximum=1.0)
            self.relevance_scores[normalized_entry_id] = score
            self.store[normalized_entry_id]["meta"]["relevance"] = score
            self.store[normalized_entry_id]["meta"]["updated_at"] = utc_iso()
            self.store[normalized_entry_id]["meta"]["metadata_fingerprint"] = self._fingerprint_metadata(self.store[normalized_entry_id]["meta"])
            return True

    def search_secure(
        self,
        query: str,
        tag_filter: Optional[str] = None,
        *,
        access_context: Optional[Dict[str, Any]] = None,
        include_snippet: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Secure content search with tag filtering and redacted results."""

        with self.lock:
            self._purge_expired_locked()
            context = access_context or self._internal_context("search_secure")
            normalized_query = normalize_text(query, max_length=256, lowercase=True)
            normalized_tag = normalize_identifier(tag_filter, max_length=96) if tag_filter is not None else None
            max_results = coerce_int(limit, coerce_int(self._cfg("search.max_results", 50), 50), minimum=1)
            results: List[Dict[str, Any]] = []
            candidate_ids = list(self.tag_index.get(normalized_tag, set())) if normalized_tag else list(self.store.keys())
            for entry_id in candidate_ids:
                entry = self.store.get(entry_id)
                if not entry:
                    continue
                decision = self._evaluate_access(context, entry=entry, action="search")
                if not decision.allowed:
                    continue
                searchable = stable_json(redact_value(entry.get("data"))).lower()
                meta_text = stable_json(redact_value(entry.get("meta", {}))).lower()
                if normalized_query in searchable or normalized_query in meta_text:
                    result = {
                        "entry_id": entry_id,
                        "meta": redact_value(entry.get("meta", {})),
                        "score": self._score_search_result(normalized_query, searchable, meta_text),
                    }
                    if include_snippet:
                        result["snippet"] = truncate_text(redact_text(searchable), coerce_int(self._cfg("search.snippet_length", 240), 240))
                    results.append(result)
                    if len(results) >= max_results:
                        break
            results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
            self.stats["search_count"] += 1
            self._record_audit_event_locked(
                event_type="memory.search",
                action="search",
                allowed=True,
                reason="search_completed",
                entry_id=None,
                context=context,
                metadata={"query_fingerprint": fingerprint(normalized_query), "tag_filter": normalized_tag, "result_count": len(results)},
            )
            return results

    def bootstrap_if_empty(self) -> None:
        """Bootstrap secure memory with essential baseline entries if absent."""

        baseline_entries = [
            ("data_classification", {"email.body": "Confidential", "training_features_url": "Restricted", "log_entries": "Internal"}, 0.8),
            ("consent_records", {"consent_granted": True, "timestamp": utc_iso()}, 0.9),
            ("feature_extraction", {"features": ["url_length", "has_ip", "domain_entropy"], "input_size": 1450}, 0.7),
            ("retention_policy", {"expiration_days": 365, "auto_delete": True, "policy_start": utc_iso()}, 0.85),
            ("trusted_hashes", {"phishing_model.json": "b03e9ad5428be299e66b3dd552e89edb23a3b19c6b3c2fc309c75b0eabed7a85"}, 0.9),
            ("data_usage_purpose", {"declared_purpose": "Phishing detection and cybersecurity threat mitigation"}, 0.9),
            ("subject_requests", {"accessed": True, "corrected": True, "deleted": True, "timestamp": utc_iso()}, 0.85),
        ]
        for tag, payload, sensitivity in baseline_entries:
            if not self.recall(tag=tag, top_k=1):
                self.add(payload, tags=[tag, "bootstrap"], sensitivity=sensitivity, purpose="secure_memory_bootstrap", source="bootstrap")

    # ------------------------------------------------------------------
    # Integrity helpers that rely on safety_helpers primitives
    # ------------------------------------------------------------------

    def _new_entry_id(self, preferred: Optional[str] = None) -> str:
        if preferred:
            normalized = normalize_identifier(preferred, max_length=128)
            assert_safe_condition(
                normalized not in self.store,
                "Secure memory entry ID already exists",
                context={"entry_id": normalized},
                error_type=SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION,
                severity=SecuritySeverity.HIGH,
            )
            return normalized
        return normalize_identifier(generate_identifier("secure"), max_length=128)

    def _fingerprint_entry_data(self, entry: Any) -> str:
        return fingerprint(to_jsonable(entry), length=coerce_int(self._cfg("fingerprint_length", get_helper_setting("fingerprint_length", 16)), 16, minimum=8, maximum=64))

    def _fingerprint_metadata(self, meta: Union[MemoryMetadata, Mapping[str, Any]]) -> str:
        data = meta.to_dict() if isinstance(meta, MemoryMetadata) else dict(meta)
        data = {k: v for k, v in data.items() if k != "metadata_fingerprint"}
        return fingerprint(redact_value(data), length=coerce_int(self._cfg("fingerprint_length", get_helper_setting("fingerprint_length", 16)), 16, minimum=8, maximum=64))

    def _classification_for_sensitivity(self, sensitivity: float) -> str:
        thresholds = self._cfg("classification_thresholds", {}) or {}
        restricted = coerce_float(thresholds.get("restricted", 0.85), 0.85)
        confidential = coerce_float(thresholds.get("confidential", 0.60), 0.60)
        internal = coerce_float(thresholds.get("internal", 0.25), 0.25)
        if sensitivity >= restricted:
            return "restricted"
        if sensitivity >= confidential:
            return "confidential"
        if sensitivity >= internal:
            return "internal"
        return "public"

    def _score_search_result(self, query: str, body: str, meta: str) -> float:
        if not query:
            return 0.0
        body_hits = body.count(query)
        meta_hits = meta.count(query)
        return combine_risk_scores(min(body_hits / 5.0, 1.0), min(meta_hits / 5.0, 1.0), method="max")

    def __len__(self) -> int:
        return len(self.store)

    def __contains__(self, entry_id: object) -> bool:
        return isinstance(entry_id, str) and normalize_identifier(entry_id, max_length=128) in self.store


__all__ = [
    "MODULE_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "ENTRY_SCHEMA_VERSION",
    "AUDIT_SCHEMA_VERSION",
    "MemoryMetadata",
    "AccessDecision",
    "MemoryAuditEvent",
    "SecureMemory",
]


if __name__ == "__main__":
    print("\n=== Running Secure Memory ===\n")
    printer.status("TEST", "Secure Memory initialized", "info")

    memory = SecureMemory()
    access_context = {
        "auth_token": "test-auth-token",
        "access_level": 5,
        "purpose": "self_test",
        "principal": "secure-memory-test",
        "request_id": generate_request_id(),
    }

    entry_payload = {
        "user_id": "user-123",
        "email": "alice@example.com",
        "api_key": "sk-test-secret",
        "note": "Prompt injection and privacy test payload",
    }
    entry_id = memory.add(
        entry_payload,
        tags=["self_test", "privacy", "prompt_security"],
        sensitivity=0.92,
        ttl_seconds=3600,
        purpose="self_test",
        owner="safety_agent",
        metadata={"source_ip": "192.168.1.10", "token": "never-log-me"},
    )
    assert entry_id in memory
    printer.status("TEST", f"Entry added: {entry_id}", "info")

    retrieved = memory.get(entry_id, access_context)
    assert retrieved is not None
    assert retrieved["meta"]["classification"] == "restricted"
    assert retrieved["data"]["note"].startswith("Prompt injection")
    printer.status("TEST", f"Retrieved metadata: {stable_json(redact_value(retrieved['meta']))}", "info")

    denied = memory.get(entry_id, {"auth_token": "x", "access_level": 0, "purpose": "self_test", "principal": "low-access"})
    assert denied is None
    printer.status("TEST", "Access denial returned safely", "info")

    recalled = memory.recall("self_test", top_k=5)
    assert recalled
    search_results = memory.search_secure("privacy", tag_filter="self_test", access_context=access_context)
    assert search_results
    assert "alice@example.com" not in stable_json(search_results)
    assert "sk-test-secret" not in stable_json(search_results)
    printer.status("TEST", f"Search result count: {len(search_results)}", "info")

    stats = memory.get_statistics()
    assert stats["total_entries"] >= 1
    assert "security_stats" in stats
    printer.status("TEST", f"Stats: {stable_json(stats)}", "info")

    audit = memory.audit_access(max_results=20)
    audit_blob = stable_json(audit)
    assert "sk-test-secret" not in audit_blob
    assert "alice@example.com" not in audit_blob
    assert "never-log-me" not in audit_blob
    printer.status("TEST", f"Audit events: {len(audit)}", "info")

    checkpoint_name = "secure_memory_self_test.json"
    assert memory.create_checkpoint(checkpoint_name)
    checkpoint_path = str(memory._checkpoint_dir() / checkpoint_name)
    restored = SecureMemory()
    assert restored.load_checkpoint(checkpoint_path)
    assert len(restored) >= 1
    printer.status("TEST", "Checkpoint created and loaded", "info")

    updated = memory.update(entry_id, {"status": "updated", "secret": "redact-me"}, access_context=access_context)
    assert updated
    assert memory.update_relevance(entry_id, 0.42)
    sanitized_count = memory.sanitize_memory(tag="self_test", access_context=access_context)
    assert sanitized_count >= 1
    sanitized_entry = memory.store.get(entry_id)
    assert sanitized_entry is None or sanitized_entry["data"] is None or sanitized_entry["meta"].get("status") == "sanitized"
    printer.status("TEST", "Update, relevance, and sanitization paths passed", "info")

    memory.bootstrap_if_empty()
    assert memory.recall("trusted_hashes", top_k=1)
    printer.status("TEST", "Bootstrap path passed", "info")

    print("\n=== Test ran successfully ===\n")
