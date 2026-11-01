"""
Shared secure evidence memory for the SLAI Safety Agent subsystem.

The module provides a process-local, capability-protected evidence store with
TTL/retention, indexing, audit records, relevance-aware eviction, and
HMAC-authenticated checkpoints. Safety subsystem modules should obtain the
shared namespace through ``SecureMemory.shared()`` so compliance, reward,
cyber, guard, STPA, and monitoring components observe the same evidence plane.

No raw user payload is written to logs. Checkpoint mode defaults to redacted.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import secrets
import tempfile

from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Union

from .utils.config_loader import get_config_section, load_global_config
from .utils.safety_helpers import *
from .utils.security_error import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Secure Memory")
printer = PrettyPrinter()

MODULE_VERSION = "2.3.0"
CHECKPOINT_SCHEMA_VERSION = "secure_memory.checkpoint.v4"
ENTRY_SCHEMA_VERSION = "secure_memory.entry.v4"
AUDIT_SCHEMA_VERSION = "secure_memory.audit.v3"


@dataclass
class MemoryMetadata:
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


@dataclass
class _MemoryBackend:
    namespace: str
    capability: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    store: "OrderedDict[str, Dict[str, Any]]" = field(default_factory=OrderedDict)
    tag_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    purpose_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    owner_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    access_log: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=lambda: {
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
    })
    lock: RLock = field(default_factory=RLock)


class SecureMemory:
    """Capability-protected private evidence store for the Safety subsystem."""

    _registry_lock = RLock()
    _shared_backends: Dict[str, _MemoryBackend] = {}

    def __init__(
        self,
        *,
        namespace: Optional[str] = None,
        shared_backend: bool = False,
        backend: Optional[_MemoryBackend] = None,
    ) -> None:
        self.config = load_global_config()
        self.memory_config = dict(get_config_section("secure_memory") or {})
        self._setup_defaults()
        self._validate_configuration()
        self.namespace = normalize_identifier(
            namespace or self.memory_config.get("namespace", "safety"),
            max_length=96,
            default="safety",
        )
        if backend is not None:
            self._backend = backend
        elif shared_backend:
            with self._registry_lock:
                self._backend = self._shared_backends.setdefault(self.namespace, _MemoryBackend(self.namespace))
        else:
            self._backend = _MemoryBackend(self.namespace)
        self._bind_backend_views()
        self._checkpoint_dir().mkdir(parents=True, exist_ok=True)

    @classmethod
    def shared(cls, namespace: Optional[str] = None) -> "SecureMemory":
        """Return an instance bound to the process-shared Safety evidence backend."""
        return cls(namespace=namespace, shared_backend=True)

    def _bind_backend_views(self) -> None:
        self.store = self._backend.store
        self.tag_index = self._backend.tag_index
        self.purpose_index = self._backend.purpose_index
        self.owner_index = self._backend.owner_index
        self.relevance_scores = self._backend.relevance_scores
        self.access_log = self._backend.access_log
        self.stats = self._backend.stats
        self.lock = self._backend.lock

    def _setup_defaults(self) -> None:
        cfg = self.memory_config
        cfg.setdefault("max_size", 5000)
        cfg.setdefault("eviction_policy", "LRU")
        cfg.setdefault("checkpoint_dir", "src/agents/safety/checkpoints/secure_checkpoints")
        cfg.setdefault("checkpoint_freq", 1000)
        cfg.setdefault("relevance_decay", 0.995)
        cfg.setdefault("access_reinforcement", 0.02)
        cfg.setdefault("min_relevance", 0.05)
        cfg.setdefault("max_access_log", 10000)
        cfg.setdefault("default_ttl_seconds", 86400)
        cfg.setdefault("min_sensitivity", 0.0)
        cfg.setdefault("max_sensitivity", 1.0)
        cfg.setdefault("default_tags", ["security", "sensitive"])
        cfg.setdefault("checkpoint_security", {})
        cfg.setdefault("access_validation", {})
        cfg.setdefault("classification_thresholds", {})
        cfg.setdefault("retention", {})
        cfg.setdefault("search", {})

    def _cfg(self, path: Union[str, Sequence[str]], default: Any = None) -> Any:
        return get_nested(self.memory_config, path, default)

    def _validate_configuration(self) -> None:
        if coerce_int(self._cfg("max_size"), 0) <= 0:
            raise ConfigurationTamperingError(
                "secure_memory.max_size", "max_size must be positive", component="secure_memory"
            )
        if str(self._cfg("eviction_policy", "LRU")).upper() not in {"LRU", "FIFO", "LFU", "LEAST_RELEVANT"}:
            raise ConfigurationTamperingError(
                "secure_memory.eviction_policy", "Unsupported eviction policy", component="secure_memory"
            )
        mode = str(self._cfg("checkpoint_security.data_mode", "redacted")).lower()
        if mode not in {"metadata_only", "redacted", "full"}:
            raise ConfigurationTamperingError(
                "secure_memory.checkpoint_security.data_mode", f"Unsupported data_mode: {mode}", component="secure_memory"
            )

    # ------------------------------------------------------------------
    # Capability and access context
    # ------------------------------------------------------------------
    def internal_context(
        self,
        purpose: str = "internal",
        *,
        principal: str = "safety_subsystem",
        access_level: Optional[int] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        access_cfg = self._cfg("access_validation", {}) or {}
        return {
            "auth_token": self._backend.capability,
            "access_level": coerce_int(
                access_level,
                coerce_int(access_cfg.get("internal_access_level", access_cfg.get("min_access_level", 0)), 0),
            ),
            "purpose": normalize_identifier(purpose, max_length=96, default="internal"),
            "principal": normalize_text(principal, max_length=128),
            "component": normalize_identifier(principal, max_length=128, default="safety_subsystem"),
            "request_id": generate_request_id(),
            "tenant_id": normalize_identifier(tenant_id, max_length=128) if tenant_id else None,
        }

    def _internal_context(self, action: str = "internal") -> Dict[str, Any]:
        return self.internal_context(action, principal="secure_memory")

    def _evaluate_access(
        self,
        context: Mapping[str, Any],
        *,
        entry: Optional[Mapping[str, Any]],
        action: str,
    ) -> AccessDecision:
        cfg = self._cfg("access_validation", {}) or {}
        required_fields = list(cfg.get("required_fields", []))
        min_level = coerce_int(cfg.get("min_access_level", 0), 0)
        supplied_level = coerce_int(context.get("access_level", 0), 0)
        principal = normalize_text(context.get("principal") or context.get("user_id") or "unknown", max_length=128)
        purpose = normalize_text(context.get("purpose") or action, max_length=128)
        missing = tuple(field for field in required_fields if context.get(field) in (None, ""))
        if missing:
            return AccessDecision(False, "missing_required_access_context", action, min_level, supplied_level, principal, purpose, missing_fields=missing)

        if coerce_bool(cfg.get("require_internal_capability", True), True):
            supplied = str(context.get("auth_token", ""))
            if not supplied or not hmac.compare_digest(supplied, self._backend.capability):
                return AccessDecision(False, "invalid_internal_capability", action, min_level, supplied_level, principal, purpose)
        if supplied_level < min_level:
            return AccessDecision(False, "insufficient_access_level", action, min_level, supplied_level, principal, purpose)

        allowed_purposes = {str(v) for v in (cfg.get("allowed_purposes", []) or [])}
        if allowed_purposes and purpose not in allowed_purposes:
            return AccessDecision(False, "purpose_not_allowed", action, min_level, supplied_level, principal, purpose)

        if entry is not None:
            meta = entry.get("meta", {}) if isinstance(entry, Mapping) else {}
            sensitivity = coerce_float(meta.get("sensitivity", 1.0), 1.0, minimum=0.0, maximum=1.0)
            required_by_class = self._cfg("sensitivity_access_levels", {}) or {}
            required_level = coerce_int(required_by_class.get(self._classification_for_sensitivity(sensitivity), min_level), min_level)
            if supplied_level < required_level:
                return AccessDecision(False, "insufficient_sensitivity_access_level", action, required_level, supplied_level, principal, purpose, entry_id=meta.get("entry_id"))
            if meta.get("status") not in {None, "active"}:
                return AccessDecision(False, f"entry_status_{meta.get('status')}", action, required_level, supplied_level, principal, purpose, entry_id=meta.get("entry_id"))
            entry_tenant = get_nested(meta, "extra.tenant_id", None)
            caller_tenant = context.get("tenant_id")
            if entry_tenant and caller_tenant and normalize_identifier(entry_tenant) != normalize_identifier(caller_tenant):
                return AccessDecision(False, "tenant_mismatch", action, required_level, supplied_level, principal, purpose, entry_id=meta.get("entry_id"))
        return AccessDecision(True, "access_allowed", action, min_level, supplied_level, principal, purpose, entry_id=get_nested(entry or {}, "meta.entry_id", None))

    def _validate_access(self, context: Dict) -> bool:
        return self._evaluate_access(context or {}, entry=None, action="validate").allowed

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
        with self.lock:
            self._purge_expired_locked()
            normalized_entry_id = self._new_entry_id(entry_id)
            normalized_sensitivity = self._normalize_sensitivity(sensitivity)
            normalized_ttl = self._normalize_ttl(ttl_seconds, normalized_sensitivity)
            normalized_tags = self._normalize_tags(tags)
            now = utc_iso()
            extra = redact_value(dict(metadata or {}))
            meta_obj = MemoryMetadata(
                entry_id=normalized_entry_id,
                created_at=now,
                updated_at=now,
                last_accessed_at=None,
                expires_at=self._calculate_expiration(normalized_ttl),
                ttl_seconds=normalized_ttl,
                tags=normalized_tags,
                sensitivity=normalized_sensitivity,
                relevance=1.0,
                purpose=normalize_text(purpose or self._cfg("default_purpose", "safety_security_memory"), max_length=128),
                owner=normalize_text(owner or self._cfg("default_owner", "safety_agent"), max_length=128),
                classification=normalize_text(classification or self._classification_for_sensitivity(normalized_sensitivity), max_length=64),
                source=normalize_text(source, max_length=128),
                status="active",
                data_type=type(entry).__name__,
                data_size_bytes=len(stable_json(redact_value(entry)).encode("utf-8", errors="replace")),
                data_fingerprint=self._fingerprint_entry_data(entry),
                metadata_fingerprint="pending",
                extra=extra,
            )
            meta_obj.metadata_fingerprint = self._fingerprint_metadata(meta_obj)
            self.store[normalized_entry_id] = {"data": copy.deepcopy(entry), "meta": meta_obj.to_dict()}
            self.relevance_scores[normalized_entry_id] = 1.0
            self._index_entry_locked(normalized_entry_id, meta_obj.to_dict())
            self.stats["entries_created"] += 1
            self._record_audit_event_locked(
                event_type="memory.add",
                action="add",
                allowed=True,
                reason="entry_created",
                entry_id=normalized_entry_id,
                context=self._internal_context("add"),
                metadata={"tags": normalized_tags, "sensitivity": normalized_sensitivity, "purpose": meta_obj.purpose},
            )
            self._manage_capacity_locked()
            self._maybe_checkpoint_locked()
            return normalized_entry_id

    def get(self, entry_id: str, access_context: Dict) -> Optional[Dict[str, Any]]:
        with self.lock:
            self._purge_expired_locked()
            normalized = normalize_identifier(entry_id, max_length=128)
            entry = self.store.get(normalized)
            if entry is None:
                self.stats["misses"] += 1
                return None
            decision = self._evaluate_access(access_context or {}, entry=entry, action="get")
            if not decision.allowed:
                self._handle_access_denied_locked(decision, normalized, access_context or {})
                return None
            self._touch_entry_locked(normalized, entry)
            self.stats["access_count"] += 1
            self.stats["access_allowed"] += 1
            self._record_audit_event_locked(event_type="memory.get", action="get", allowed=True, reason=decision.reason, entry_id=normalized, context=access_context or {})
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
        with self.lock:
            normalized = normalize_identifier(entry_id, max_length=128)
            current = self.store.get(normalized)
            if current is None:
                self.stats["misses"] += 1
                return False
            context = access_context or self._internal_context("update")
            decision = self._evaluate_access(context, entry=current, action="update")
            if not decision.allowed:
                self._handle_access_denied_locked(decision, normalized, context)
                return False
            old_meta = current["meta"]
            self._deindex_entry_locked(normalized, old_meta)
            new_sensitivity = self._normalize_sensitivity(old_meta.get("sensitivity", 1.0) if sensitivity is None else sensitivity)
            new_ttl = self._normalize_ttl(old_meta.get("ttl_seconds") if ttl_seconds is None else ttl_seconds, new_sensitivity)
            current["data"] = copy.deepcopy(entry)
            current["meta"].update({
                "updated_at": utc_iso(),
                "expires_at": self._calculate_expiration(new_ttl),
                "ttl_seconds": new_ttl,
                "tags": self._normalize_tags(tags if tags is not None else old_meta.get("tags", [])),
                "sensitivity": new_sensitivity,
                "classification": self._classification_for_sensitivity(new_sensitivity),
                "data_type": type(entry).__name__,
                "data_size_bytes": len(stable_json(redact_value(entry)).encode("utf-8", errors="replace")),
                "data_fingerprint": self._fingerprint_entry_data(entry),
                "revision": coerce_int(old_meta.get("revision", 1), 1) + 1,
                "extra": redact_value(dict(metadata if metadata is not None else old_meta.get("extra", {}))),
            })
            current["meta"]["metadata_fingerprint"] = self._fingerprint_metadata(current["meta"])
            self._index_entry_locked(normalized, current["meta"])
            self.stats["entries_updated"] += 1
            self._record_audit_event_locked(event_type="memory.update", action="update", allowed=True, reason="entry_updated", entry_id=normalized, context=context)
            return True

    def delete(self, entry_id: str, access_context: Optional[Dict[str, Any]] = None, *, reason: str = "delete_requested") -> bool:
        with self.lock:
            normalized = normalize_identifier(entry_id, max_length=128)
            entry = self.store.get(normalized)
            if entry is None:
                self.stats["misses"] += 1
                return False
            context = access_context or self._internal_context("delete")
            decision = self._evaluate_access(context, entry=entry, action="delete")
            if not decision.allowed:
                self._handle_access_denied_locked(decision, normalized, context)
                return False
            self._remove_entry_locked(normalized, reason=reason)
            return True

    def recall(self, tag: str, top_k: Optional[int] = None, access_context: Optional[Dict[str, Any]] = None) -> List[Any]:
        with self.lock:
            self._purge_expired_locked()
            normalized_tag = normalize_identifier(tag, max_length=96)
            context = access_context or self._internal_context("recall")
            results: List[Dict[str, Any]] = []
            for entry_id in list(self.tag_index.get(normalized_tag, set())):
                entry = self.store.get(entry_id)
                if not entry:
                    continue
                decision = self._evaluate_access(context, entry=entry, action="recall")
                if decision.allowed:
                    self._touch_entry_locked(entry_id, entry, move_lru=False)
                    results.append(copy.deepcopy(entry))
                else:
                    self.stats["failed_checks"] += 1
            results.sort(key=lambda value: coerce_float(get_nested(value, "meta.relevance", 0.0), 0.0), reverse=True)
            if top_k:
                results = results[: coerce_int(top_k, len(results), minimum=1)]
            self.stats["recall_count"] += 1
            self._record_audit_event_locked(
                event_type="memory.recall", action="recall", allowed=True, reason="tag_recall_completed", entry_id=None,
                context=context, metadata={"tag": normalized_tag, "result_count": len(results)},
            )
            return results

    # ------------------------------------------------------------------
    # Search, maintenance, and evidence semantics
    # ------------------------------------------------------------------
    def search_secure(
        self,
        query: str,
        tag_filter: Optional[str] = None,
        *,
        access_context: Optional[Dict[str, Any]] = None,
        include_snippet: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        with self.lock:
            self._purge_expired_locked()
            context = access_context or self._internal_context("search_secure")
            normalized_query = normalize_text(query, max_length=256, lowercase=True)
            normalized_tag = normalize_identifier(tag_filter, max_length=96) if tag_filter else None
            candidate_ids = list(self.tag_index.get(normalized_tag, set())) if normalized_tag else list(self.store.keys())
            max_results = coerce_int(limit, coerce_int(self._cfg("search.max_results", 50), 50), minimum=1)
            results: List[Dict[str, Any]] = []
            for entry_id in candidate_ids:
                entry = self.store.get(entry_id)
                if not entry or not self._evaluate_access(context, entry=entry, action="search").allowed:
                    continue
                body = stable_json(redact_value(entry.get("data"))).lower()
                meta = stable_json(redact_value(entry.get("meta", {}))).lower()
                if normalized_query not in body and normalized_query not in meta:
                    continue
                result = {
                    "entry_id": entry_id,
                    "meta": redact_value(entry.get("meta", {})),
                    "score": self._score_search_result(normalized_query, body, meta),
                }
                if include_snippet:
                    result["snippet"] = truncate_text(redact_text(body), coerce_int(self._cfg("search.snippet_length", 240), 240))
                results.append(result)
            results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
            self.stats["search_count"] += 1
            return results[:max_results]

    def is_evidence_eligible(self, entry: Mapping[str, Any]) -> bool:
        """Return whether an entry is eligible for compliance/assurance decisions."""
        meta = entry.get("meta", {}) if isinstance(entry, Mapping) else {}
        extra = meta.get("extra", {}) if isinstance(meta, Mapping) else {}
        if coerce_bool(extra.get("synthetic"), False):
            return coerce_bool(extra.get("eligible_for_compliance"), False)
        return coerce_bool(extra.get("eligible_for_compliance"), True)

    def bootstrap_if_empty(self) -> None:
        """Create a non-evidentiary bootstrap manifest only.

        Earlier revisions inserted synthetic consent, data-subject-rights, and
        trusted-hash records that could be misinterpreted as real compliance
        evidence. Production bootstrap now records only schema expectations.
        """
        if self.recall("secure_memory_bootstrap", top_k=1):
            return
        self.add(
            {
                "schema_version": "secure_memory.bootstrap_manifest.v2",
                "purpose": "declare expected evidence tags; not compliance evidence",
                "expected_evidence_tags": [
                    "data_classification", "consent_records", "data_usage_purpose", "subject_requests",
                    "retention_policy", "trusted_hashes", "feature_extraction",
                ],
                "created_at": utc_iso(),
            },
            tags=["secure_memory_bootstrap", "control_definition"],
            sensitivity=0.25,
            purpose="secure_memory_bootstrap_manifest",
            source="bootstrap",
            metadata={"synthetic": True, "eligible_for_compliance": False},
        )

    def sanitize_memory(self, tag: Optional[str] = None, access_context: Optional[Dict[str, Any]] = None) -> int:
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
                entry["meta"]["data_fingerprint"] = self._fingerprint_entry_data(None)
                entry["meta"]["metadata_fingerprint"] = self._fingerprint_metadata(entry["meta"])
                self.relevance_scores[entry_id] = 0.0
                count += 1
            self.stats["entries_sanitized"] += count
            return count

    def update_relevance(self, entry_id: str, relevance: float) -> bool:
        with self.lock:
            normalized = normalize_identifier(entry_id, max_length=128)
            if normalized not in self.store:
                return False
            value = coerce_float(relevance, 0.0, minimum=0.0, maximum=1.0)
            self.relevance_scores[normalized] = value
            self.store[normalized]["meta"]["relevance"] = value
            self.store[normalized]["meta"]["updated_at"] = utc_iso()
            self.store[normalized]["meta"]["metadata_fingerprint"] = self._fingerprint_metadata(self.store[normalized]["meta"])
            return True

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _checkpoint_dir(self) -> Path:
        return Path(str(self._cfg("checkpoint_dir"))).expanduser()

    def _checkpoint_key(self) -> bytes:
        security = self._cfg("checkpoint_security", {}) or {}
        env_name = str(security.get("hmac_key_env", "SLAI_SECURE_MEMORY_HMAC_KEY"))
        env_value = os.getenv(env_name)
        if env_value:
            return env_value.encode("utf-8")
        legacy = str(security.get("signature_salt") or get_helper_setting("hash_salt", ""))
        if legacy:
            logger.warning("SecureMemory checkpoint HMAC key is derived from config fallback; set %s in production.", env_name)
            return hashlib.sha256(legacy.encode("utf-8")).digest()
        if coerce_bool(security.get("require_external_hmac_key", False), False):
            raise ConfigurationTamperingError(
                "secure_memory.checkpoint_security.hmac_key_env",
                f"Required checkpoint HMAC key environment variable {env_name} is missing",
                component="secure_memory",
            )
        return hashlib.sha256(self._backend.capability.encode("utf-8")).digest()

    def _sign_payload(self, payload: Any) -> str:
        algorithm = safe_hash_algorithm(str(self._cfg("checkpoint_security.hash_algorithm", "sha256")))
        return hmac.new(self._checkpoint_key(), stable_json(payload).encode("utf-8"), algorithm).hexdigest()

    def create_checkpoint(self, name: Optional[str] = None) -> bool:
        with self.lock:
            safe_name = normalize_identifier(name or f"secure_memory_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}", max_length=140)
            checkpoint_path = self._checkpoint_dir() / f"{safe_name}.json"
            try:
                payload = self._checkpoint_payload_locked()
                envelope = {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "module_version": MODULE_VERSION,
                    "created_at": utc_iso(),
                    "signature_algorithm": safe_hash_algorithm(str(self._cfg("checkpoint_security.hash_algorithm", "sha256"))),
                    "payload_fingerprint": fingerprint(payload),
                    "signature": self._sign_payload(payload),
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
                return True
            except SecurityError:
                self.stats["checkpoint_failure"] += 1
                raise
            except Exception as exc:
                self.stats["checkpoint_failure"] += 1
                if coerce_bool(self._cfg("checkpoint_security.raise_on_failure", False), False):
                    raise SecurityError(
                        SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION,
                        "Secure-memory checkpoint creation failed.",
                        component="secure_memory",
                        severity=SecuritySeverity.HIGH,
                        context={"error_type": type(exc).__name__},
                        cause=exc,
                    ) from exc
                return False

    def load_checkpoint(self, path: str) -> bool:
        checkpoint_path = Path(path).expanduser()
        try:
            raw = load_text_file(checkpoint_path, max_bytes=coerce_int(self._cfg("checkpoint_security.max_checkpoint_bytes", 50_000_000), 50_000_000))
            envelope = parse_json_object(raw, context="secure_memory_checkpoint")
            if envelope.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                raise SecurityError(
                    SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION,
                    "Unsupported secure-memory checkpoint schema.",
                    component="secure_memory",
                    context={"observed_schema": envelope.get("schema_version")},
                )
            payload = envelope.get("payload")
            expected = str(envelope.get("signature", ""))
            if coerce_bool(self._cfg("checkpoint_security.verify_signature", True), True) and not hmac.compare_digest(expected, self._sign_payload(payload)):
                raise SecurityError(
                    SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION,
                    "Secure-memory checkpoint HMAC verification failed.",
                    component="secure_memory",
                    severity=SecuritySeverity.CRITICAL,
                    response_action=SecurityResponseAction.BLOCK,
                )
            with self.lock:
                self._restore_payload_locked(payload) # type: ignore
                self.stats["checkpoint_loads"] += 1
            return True
        except SecurityError:
            raise
        except Exception as exc:
            if coerce_bool(self._cfg("checkpoint_security.raise_on_failure", True), True):
                raise SecurityError(
                    SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION,
                    "Secure-memory checkpoint load failed.",
                    component="secure_memory",
                    severity=SecuritySeverity.HIGH,
                    context={"error_type": type(exc).__name__},
                    cause=exc,
                ) from exc
            return False

    def _checkpoint_payload_locked(self) -> Dict[str, Any]:
        mode = str(self._cfg("checkpoint_security.data_mode", "redacted")).lower()
        entries: Dict[str, Any] = {}
        for entry_id, entry in self.store.items():
            if mode == "metadata_only":
                data = None
            elif mode == "full":
                if not coerce_bool(self._cfg("checkpoint_security.allow_plaintext_full_mode", False), False):
                    raise ConfigurationTamperingError(
                        "secure_memory.checkpoint_security.data_mode",
                        "full checkpoint mode requires an encrypted persistence provider; plaintext full mode is disabled",
                        component="secure_memory",
                    )
                data = to_jsonable(entry.get("data"))
            else:
                data = redact_value(entry.get("data"))
            entries[entry_id] = {"data": data, "meta": redact_value(copy.deepcopy(entry.get("meta", {})))}
        return {
            "entries": entries,
            "stats": redact_value(self.stats),
            "created_at": utc_iso(),
            "entry_count": len(entries),
            "config_fingerprint": fingerprint(self.memory_config),
            "namespace": self.namespace,
            "data_mode": mode,
        }

    def _restore_payload_locked(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise SecurityError(SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION, "Checkpoint payload is not a mapping.", component="secure_memory")
        entries = payload.get("entries", {})
        if not isinstance(entries, Mapping):
            raise SecurityError(SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION, "Checkpoint entries are not a mapping.", component="secure_memory")
        self.store.clear(); self.tag_index.clear(); self.purpose_index.clear(); self.owner_index.clear(); self.relevance_scores.clear()
        for raw_id, raw_entry in entries.items():
            if not isinstance(raw_entry, Mapping):
                continue
            entry_id = normalize_identifier(raw_id, max_length=128)
            meta = dict(raw_entry.get("meta", {}))
            meta.setdefault("entry_id", entry_id)
            meta.setdefault("status", "active")
            meta.setdefault("tags", [])
            data = raw_entry.get("data")
            expected_data_fp = meta.get("data_fingerprint")
            if expected_data_fp and str(payload.get("data_mode")) == "full" and not constant_time_equals(str(expected_data_fp), self._fingerprint_entry_data(data)):
                raise SecurityError(SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION, "Checkpoint entry data fingerprint mismatch.", component="secure_memory", context={"entry_id": entry_id})
            expected_meta_fp = meta.get("metadata_fingerprint")
            if expected_meta_fp and not constant_time_equals(str(expected_meta_fp), self._fingerprint_metadata(meta)):
                # Redacted checkpoints may change metadata-sensitive representations; fail only if configured.
                if coerce_bool(self._cfg("checkpoint_security.require_entry_metadata_fingerprint", False), False):
                    raise SecurityError(SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION, "Checkpoint metadata fingerprint mismatch.", component="secure_memory", context={"entry_id": entry_id})
            self.store[entry_id] = {"data": data, "meta": meta}
            self.relevance_scores[entry_id] = coerce_float(meta.get("relevance", 1.0), 1.0, minimum=0.0, maximum=1.0)
            self._index_entry_locked(entry_id, meta)

    def _maybe_checkpoint_locked(self) -> None:
        frequency = coerce_int(self._cfg("checkpoint_freq", 0), 0, minimum=0)
        if frequency and self.stats["entries_created"] % frequency == 0:
            self.create_checkpoint()

    # ------------------------------------------------------------------
    # Audit, retention, indexing
    # ------------------------------------------------------------------
    def _handle_access_denied_locked(self, decision: AccessDecision, entry_id: str, context: Mapping[str, Any]) -> None:
        self.stats["access_denied"] += 1
        self.stats["failed_checks"] += 1
        self._record_audit_event_locked(event_type="memory.access_denied", action=decision.action, allowed=False, reason=decision.reason, entry_id=entry_id, context=context, metadata=decision.to_dict())
        if coerce_bool(self._cfg("access_validation.raise_on_denied", False), False):
            raise SecurityError(
                SecurityErrorType.ACCESS_VIOLATION,
                "Secure-memory access denied.",
                component="secure_memory",
                severity=SecuritySeverity.HIGH,
                context={"entry_id": entry_id, "reason": decision.reason, "principal": decision.principal},
                response_action=SecurityResponseAction.BLOCK,
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
        event = MemoryAuditEvent(
            event_id=generate_identifier("mem_evt"), timestamp=utc_iso(), event_type=normalize_text(event_type, max_length=96),
            action=normalize_text(action, max_length=64), allowed=bool(allowed), reason=normalize_text(reason, max_length=160),
            entry_id=normalize_identifier(entry_id, max_length=128) if entry_id else None,
            principal=normalize_text(context.get("principal") or context.get("user_id") or "unknown", max_length=128),
            purpose=normalize_text(context.get("purpose") or action, max_length=128), context=sanitize_for_logging(dict(context)),
            metadata=sanitize_for_logging(dict(metadata or {})),
        )
        self.access_log.append(event.to_dict())
        max_log = coerce_int(self._cfg("max_access_log", 10000), 10000, minimum=100)
        if len(self.access_log) > max_log:
            del self.access_log[: len(self.access_log) - max_log]
        self.stats["audit_events"] += 1

    def _normalize_tags(self, tags: Optional[Iterable[str]]) -> List[str]:
        values = list(self._cfg("default_tags", []) or []) + list(tags or [])
        return dedupe_preserve_order(normalize_identifier(v, max_length=96) for v in values if str(v).strip())

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
        return (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat().replace("+00:00", "Z")

    def _is_expired(self, meta: Mapping[str, Any]) -> bool:
        if meta.get("legal_hold") or not meta.get("expires_at"):
            return False
        try:
            return parse_iso_datetime(str(meta["expires_at"])) <= datetime.now(timezone.utc)
        except Exception:
            return True

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
            key = normalize_identifier(tag, max_length=96)
            self.tag_index.get(key, set()).discard(entry_id)
            if key in self.tag_index and not self.tag_index[key]:
                del self.tag_index[key]
        for field_name, index in (("purpose", self.purpose_index), ("owner", self.owner_index)):
            key = normalize_identifier(meta.get(field_name, "unknown"), max_length=96)
            index.get(key, set()).discard(entry_id)
            if key in index and not index[key]:
                del index[key]

    def _manage_capacity_locked(self) -> None:
        max_size = coerce_int(self._cfg("max_size", 5000), 5000, minimum=1)
        while len(self.store) > max_size:
            candidate = self._select_eviction_candidate_locked()
            if candidate is None:
                break
            self._remove_entry_locked(candidate, reason="evicted")
            self.stats["evictions"] += 1

    def _select_eviction_candidate_locked(self) -> Optional[str]:
        candidates = [(k, v) for k, v in self.store.items() if not get_nested(v, "meta.legal_hold", False)]
        if not candidates:
            return None
        policy = str(self._cfg("eviction_policy", "LRU")).upper()
        if policy == "FIFO": return candidates[0][0]
        if policy == "LFU": return min(candidates, key=lambda item: coerce_int(get_nested(item[1], "meta.access_count", 0), 0))[0]
        if policy == "LEAST_RELEVANT": return min(candidates, key=lambda item: coerce_float(get_nested(item[1], "meta.relevance", 0.0), 0.0))[0]
        return candidates[0][0]

    def _remove_entry_locked(self, entry_id: str, *, reason: str) -> None:
        entry = self.store.get(entry_id)
        if not entry:
            return
        meta = entry.get("meta", {})
        self._deindex_entry_locked(entry_id, meta)
        self.store.pop(entry_id, None)
        self.relevance_scores.pop(entry_id, None)
        self.stats["entries_deleted"] += 1
        self._record_audit_event_locked(event_type="memory.remove", action="remove", allowed=True, reason=reason, entry_id=entry_id, context=self._internal_context("remove"), metadata={"reason": reason})

    def _touch_entry_locked(self, entry_id: str, entry: MutableMapping[str, Any], *, move_lru: bool = True) -> None:
        meta = entry["meta"]
        meta["access_count"] = coerce_int(meta.get("access_count", 0), 0) + 1
        meta["last_accessed_at"] = utc_iso()
        decay = coerce_float(self._cfg("relevance_decay", 0.995), 0.995, minimum=0.0, maximum=1.0)
        reinforce = coerce_float(self._cfg("access_reinforcement", 0.02), 0.02, minimum=0.0, maximum=1.0)
        minimum = coerce_float(self._cfg("min_relevance", 0.05), 0.05, minimum=0.0, maximum=1.0)
        current = coerce_float(meta.get("relevance", 1.0), 1.0, minimum=0.0, maximum=1.0)
        meta["relevance"] = max(minimum, min(1.0, current * decay + reinforce))
        meta["metadata_fingerprint"] = self._fingerprint_metadata(meta)
        self.relevance_scores[entry_id] = meta["relevance"]
        if move_lru and str(self._cfg("eviction_policy", "LRU")).upper() == "LRU":
            self.store.move_to_end(entry_id)

    # ------------------------------------------------------------------
    # Integrity and reporting
    # ------------------------------------------------------------------
    def _new_entry_id(self, preferred: Optional[str] = None) -> str:
        if preferred:
            normalized = normalize_identifier(preferred, max_length=128)
            if normalized in self.store:
                raise SecurityError(SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION, "Secure-memory entry id already exists.", component="secure_memory", context={"entry_id": normalized})
            return normalized
        return normalize_identifier(generate_identifier("secure"), max_length=128)

    def _fingerprint_entry_data(self, entry: Any) -> str:
        return fingerprint(to_jsonable(entry), length=coerce_int(self._cfg("fingerprint_length", 16), 16, minimum=8, maximum=64))

    def _fingerprint_metadata(self, meta: Union[MemoryMetadata, Mapping[str, Any]]) -> str:
        data = meta.to_dict() if isinstance(meta, MemoryMetadata) else dict(meta)
        data.pop("metadata_fingerprint", None)
        return fingerprint(redact_value(data), length=coerce_int(self._cfg("fingerprint_length", 16), 16, minimum=8, maximum=64))

    def _classification_for_sensitivity(self, sensitivity: float) -> str:
        thresholds = self._cfg("classification_thresholds", {}) or {}
        if sensitivity >= coerce_float(thresholds.get("restricted", 0.85), 0.85): return "restricted"
        if sensitivity >= coerce_float(thresholds.get("confidential", 0.60), 0.60): return "confidential"
        if sensitivity >= coerce_float(thresholds.get("internal", 0.25), 0.25): return "internal"
        return "public"

    def _score_search_result(self, query: str, body: str, meta: str) -> float:
        if not query: return 0.0
        return combine_risk_scores(min(body.count(query) / 5.0, 1.0), min(meta.count(query) / 5.0, 1.0), method="max")

    def get_statistics(self) -> Dict[str, Any]:
        with self.lock:
            relevance = list(self.relevance_scores.values())
            sensitivity = [coerce_float(get_nested(v, "meta.sensitivity", 0.0), 0.0) for v in self.store.values()]
            return sanitize_for_logging({
                "schema_version": AUDIT_SCHEMA_VERSION,
                "module_version": MODULE_VERSION,
                "namespace": self.namespace,
                "total_entries": len(self.store),
                "active_tags": len(self.tag_index),
                "active_purposes": len(self.purpose_index),
                "active_owners": len(self.owner_index),
                "avg_relevance": sum(relevance) / len(relevance) if relevance else 0.0,
                "avg_sensitivity": sum(sensitivity) / len(sensitivity) if sensitivity else 0.0,
                "security_stats": copy.deepcopy(self.stats),
                "checkpoint_dir": str(self._checkpoint_dir()),
                "eviction_policy": self._cfg("eviction_policy"),
            })

    def audit_access(self, max_results: int = 100) -> List[Dict[str, Any]]:
        with self.lock:
            limit = coerce_int(max_results, 100, minimum=1, maximum=coerce_int(self._cfg("max_access_log", 10000), 10000))
            return copy.deepcopy(self.access_log[-limit:])

    def __len__(self) -> int:
        return len(self.store)

    def __contains__(self, entry_id: object) -> bool:
        return isinstance(entry_id, str) and normalize_identifier(entry_id, max_length=128) in self.store


__all__ = [
    "MODULE_VERSION", "CHECKPOINT_SCHEMA_VERSION", "ENTRY_SCHEMA_VERSION", "AUDIT_SCHEMA_VERSION",
    "MemoryMetadata", "AccessDecision", "MemoryAuditEvent", "SecureMemory",
]
