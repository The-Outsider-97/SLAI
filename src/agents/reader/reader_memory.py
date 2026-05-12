from __future__ import annotations

"""Production-grade persistent memory for the Reader subsystem.

ReaderMemory is the state boundary for Reader workflows. It stores deterministic
checkpoints for replay/debugging and hash-keyed cache entries for repeat batch
runs. The module deliberately delegates reusable filesystem, JSON, hashing,
redaction, and small utility behavior to ``reader_helpers.py`` and delegates all
Reader-specific error taxonomy to ``reader_error.py``.

Design goals
------------
- Keep the existing public API compatible: ``write_checkpoint()``,
  ``set_cache()``, and ``get_cache()`` keep their original behavior.
- Persist versioned, integrity-checked checkpoint/cache envelopes.
- Use reader_config.yaml for every tunable memory setting.
- Avoid duplicating helper functions already centralized in reader_helpers.py.
- Raise Reader-domain persistence errors with actionable context.
- Support cache TTL, checkpoint lookup, listing, deletion, pruning, and stats.
"""

import fnmatch
import os
import threading
import uuid

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .utils.config_loader import get_config_section, load_reader_config
from .utils.reader_error import *
from .utils.reader_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Reader Memory")
printer = PrettyPrinter()


READER_MEMORY_SCHEMA_VERSION = "2.0"
DEFAULT_CHECKPOINT_DIR = "tmp/reader/checkpoints"
DEFAULT_CACHE_DIR = "tmp/reader/cache"
DEFAULT_AUDIT_DIR = "tmp/reader/audit"
DEFAULT_CACHE_TTL_SECONDS = 0
DEFAULT_MAX_CACHE_ENTRIES = 10_000
DEFAULT_MAX_CHECKPOINT_ENTRIES = 5_000
DEFAULT_MAX_CACHE_ENTRY_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES = 25 * 1024 * 1024
DEFAULT_JSON_INDENT = None
DEFAULT_INDEX_FILENAME = "index.json"


@dataclass(frozen=True)
class ReaderMemoryPaths:
    """Resolved storage directories used by ReaderMemory."""

    checkpoint_dir: Path
    cache_dir: Path
    audit_dir: Path

    def to_dict(self) -> Dict[str, str]:
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "cache_dir": str(self.cache_dir),
            "audit_dir": str(self.audit_dir),
        }


@dataclass(frozen=True)
class ReaderMemoryEntry:
    """Small public summary for a persisted Reader memory entry."""

    entry_type: str
    identifier: str
    path: str
    created_at: float
    size_bytes: int
    namespace: Optional[str] = None
    step: Optional[str] = None
    run_id: Optional[str] = None
    expired: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return compact_none(
            {
                "entry_type": self.entry_type,
                "identifier": self.identifier,
                "path": self.path,
                "created_at": self.created_at,
                "size_bytes": self.size_bytes,
                "namespace": self.namespace,
                "step": self.step,
                "run_id": self.run_id,
                "expired": self.expired,
            }
        )


class ReaderMemory:
    """Persistent checkpoint + cache store for Reader workflows.

    The constructor accepts an optional config override for tests, but production
    defaults come from ``reader_config.yaml`` via ``get_config_section``.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        load_reader_config()
        self.memory_config = dict(get_config_section("reader_memory") or {})
        if config:
            self.memory_config.update(dict(config))
        self.config = self.memory_config

        self.cache_ttl_seconds = coerce_int(
            self.memory_config.get("cache_ttl_seconds", DEFAULT_CACHE_TTL_SECONDS),
            DEFAULT_CACHE_TTL_SECONDS,
            minimum=0,
        )
        self.max_cache_entries = coerce_int(
            self.memory_config.get("max_cache_entries", DEFAULT_MAX_CACHE_ENTRIES),
            DEFAULT_MAX_CACHE_ENTRIES,
            minimum=1,
        )
        self.max_checkpoint_entries = coerce_int(
            self.memory_config.get("max_checkpoint_entries", DEFAULT_MAX_CHECKPOINT_ENTRIES),
            DEFAULT_MAX_CHECKPOINT_ENTRIES,
            minimum=1,
        )
        self.max_cache_entry_bytes = coerce_int(
            self.memory_config.get("max_cache_entry_bytes", DEFAULT_MAX_CACHE_ENTRY_BYTES),
            DEFAULT_MAX_CACHE_ENTRY_BYTES,
            minimum=1024,
        )
        self.max_checkpoint_payload_bytes = coerce_int(
            self.memory_config.get("max_checkpoint_payload_bytes", DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES),
            DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            minimum=1024,
        )
        self.json_indent = self.memory_config.get("json_indent", DEFAULT_JSON_INDENT)
        if self.json_indent is not None:
            self.json_indent = coerce_int(self.json_indent, 2, minimum=0, maximum=8)

        self.cache_namespace = safe_filename(self.memory_config.get("cache_namespace", "reader"), fallback="reader")
        self.checkpoint_prefix = safe_filename(self.memory_config.get("checkpoint_prefix", "reader"), fallback="reader")
        self.redact_sensitive_payloads = coerce_bool(self.memory_config.get("redact_sensitive_payloads", True), True)
        self.enable_integrity_hash = coerce_bool(self.memory_config.get("enable_integrity_hash", True), True)
        self.enable_audit_log = coerce_bool(self.memory_config.get("enable_audit_log", True), True)
        self.prune_on_init = coerce_bool(self.memory_config.get("prune_on_init", False), False)
        self.write_indexes = coerce_bool(self.memory_config.get("write_indexes", True), True)

        self.checkpoint_index_file = safe_filename(
            self.memory_config.get("checkpoint_index_file", DEFAULT_INDEX_FILENAME),
            fallback=DEFAULT_INDEX_FILENAME,
        )
        self.cache_index_file = safe_filename(
            self.memory_config.get("cache_index_file", DEFAULT_INDEX_FILENAME),
            fallback=DEFAULT_INDEX_FILENAME,
        )

        self.paths = ReaderMemoryPaths(
            checkpoint_dir=ensure_directory(self.memory_config.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR), purpose="checkpoint_dir"),
            cache_dir=ensure_directory(self.memory_config.get("cache_dir", DEFAULT_CACHE_DIR), purpose="cache_dir"),
            audit_dir=ensure_directory(self.memory_config.get("audit_dir", DEFAULT_AUDIT_DIR), purpose="audit_dir"),
        )
        self.checkpoint_dir = self.paths.checkpoint_dir
        self.cache_dir = self.paths.cache_dir
        self.audit_dir = self.paths.audit_dir
        self._lock = threading.RLock()

        if self.prune_on_init:
            self.prune()

        logger.info("Reader Memory initialized | %s", self.paths.to_dict())

    # ------------------------------------------------------------------
    # Generic envelope and path helpers
    # ------------------------------------------------------------------

    def _raise_checkpoint_error(self, message: str, context: Optional[Mapping[str, Any]] = None, cause: Optional[BaseException] = None) -> ReaderError:
        cls = globals().get("CheckpointPersistenceError")
        if isinstance(cls, type):
            return cls(message, context=context, cause=cause)
        return PersistenceError(message, dict(context or {}), cause)  # type: ignore[arg-type]

    def _raise_cache_error(self, message: str, context: Optional[Mapping[str, Any]] = None, cause: Optional[BaseException] = None) -> ReaderError:
        cls = globals().get("CachePersistenceError")
        if isinstance(cls, type):
            return cls(message, context=context, cause=cause)
        return PersistenceError(message, dict(context or {}), cause)  # type: ignore[arg-type]

    def _normalize_run_id(self, run_id: Optional[str]) -> Optional[str]:
        if run_id is None:
            return None
        normalized = safe_filename(str(run_id), fallback="run")
        return normalized or None

    def _normalize_step(self, step: str) -> str:
        normalized = safe_filename(str(step or "checkpoint"), fallback="checkpoint")
        if not normalized:
            raise ReaderValidationError("Reader checkpoint step cannot be empty", {"step": step})
        return normalized

    def _cache_key(self, payload: Mapping[str, Any], *, namespace: Optional[str] = None) -> str:
        """Return a stable cache key for a Reader cache payload.

        Kept as a method for backward compatibility with earlier ReaderMemory.
        """

        ns = safe_filename(namespace or self.cache_namespace, fallback="reader")
        return stable_hash({"namespace": ns, "payload": json_safe(payload)})

    def _json_size(self, payload: Mapping[str, Any]) -> int:
        return len(stable_json_dumps(payload, redact=self.redact_sensitive_payloads).encode("utf-8"))

    def _assert_payload_size(self, payload: Mapping[str, Any], *, max_bytes: int, purpose: str) -> None:
        size_bytes = self._json_size(payload)
        if size_bytes > max_bytes:
            error_context = {"size_bytes": size_bytes, "max_bytes": max_bytes, "purpose": purpose}
            if purpose == "checkpoint":
                raise self._raise_checkpoint_error("Reader checkpoint payload exceeds configured size limit", error_context)
            raise self._raise_cache_error("Reader cache payload exceeds configured size limit", error_context)

    def _entry_path(self, directory: Path, identifier: str) -> Path:
        filename = safe_filename(identifier, fallback="entry")
        if not filename.endswith(".json"):
            filename = f"{filename}.json"
        return directory / filename

    def _checkpoint_path(self, checkpoint_id: str) -> Path:
        return self._entry_path(self.checkpoint_dir, checkpoint_id)

    def _cache_path(self, cache_key_value: str) -> Path:
        return self._entry_path(self.cache_dir, cache_key_value)

    def _index_path(self, entry_type: str) -> Path:
        if entry_type == "checkpoint":
            return self.checkpoint_dir / self.checkpoint_index_file
        if entry_type == "cache":
            return self.cache_dir / self.cache_index_file
        return self.audit_dir / DEFAULT_INDEX_FILENAME

    def _build_envelope(self, *, entry_type: str, identifier: str, payload: Mapping[str, Any],
                        metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        created_at = utc_timestamp()
        envelope: Dict[str, Any] = {
            "schema_version": READER_MEMORY_SCHEMA_VERSION,
            "entry_type": entry_type,
            "identifier": identifier,
            "created_at": created_at,
            "created_at_iso": utc_now().isoformat(),
            "payload": json_safe(payload, redact=False),
            "metadata": json_safe(metadata or {}, redact=False),
        }
        if self.enable_integrity_hash:
            envelope["integrity"] = {
                "algorithm": "sha256",
                "payload_hash": stable_hash(envelope["payload"], redact=self.redact_sensitive_payloads),
                "envelope_hash": stable_hash({k: v for k, v in envelope.items() if k != "integrity"}, redact=self.redact_sensitive_payloads),
            }
        return envelope

    def _write_payload(self, path: Path, payload: Mapping[str, Any], *, purpose: str) -> str:
        try:
            atomic_write_json(
                path,
                payload,
                redact=self.redact_sensitive_payloads,
                indent=self.json_indent,
                purpose=purpose,
            )
            return str(path)
        except ReaderError:
            raise
        except Exception as exc:
            if purpose == "checkpoint":
                raise self._raise_checkpoint_error("Failed writing Reader checkpoint", {"path": str(path)}, exc) from exc
            if purpose == "cache":
                raise self._raise_cache_error("Failed writing Reader cache payload", {"path": str(path)}, exc) from exc
            raise PersistenceError(f"Failed writing Reader memory payload: {path}", {"path": str(path)}, exc) from exc

    def _read_payload(self, path: Path, *, purpose: str) -> Dict[str, Any]:
        try:
            return read_json_file(path, purpose=purpose)
        except ReaderError:
            raise
        except Exception as exc:
            if purpose == "checkpoint":
                raise self._raise_checkpoint_error("Failed reading Reader checkpoint", {"path": str(path)}, exc) from exc
            if purpose == "cache":
                raise self._raise_cache_error("Failed reading Reader cache payload", {"path": str(path)}, exc) from exc
            raise PersistenceError(f"Failed reading Reader memory payload: {path}", {"path": str(path)}, exc) from exc

    def _verify_envelope(self, envelope: Mapping[str, Any], *, path: Optional[Path] = None) -> bool:
        if not self.enable_integrity_hash:
            return True
        integrity = envelope.get("integrity")
        if not isinstance(integrity, Mapping):
            raise PersistenceError("Reader memory envelope is missing integrity metadata", {"path": str(path) if path else None})
        expected_payload_hash = integrity.get("payload_hash")
        actual_payload_hash = stable_hash(envelope.get("payload", {}), redact=self.redact_sensitive_payloads)
        if expected_payload_hash != actual_payload_hash:
            raise PersistenceError(
                "Reader memory payload integrity check failed",
                {"path": str(path) if path else None, "expected": expected_payload_hash, "actual": actual_payload_hash},
            )
        return True

    def _write_index(self, entry_type: str) -> None:
        if not self.write_indexes:
            return
        try:
            entries = [entry.to_dict() for entry in self._scan_entries(entry_type)]
            payload = self._build_envelope(
                entry_type=f"{entry_type}_index",
                identifier=f"{entry_type}_index",
                payload={"entries": entries, "count": len(entries)},
                metadata={"generated_by": "ReaderMemory"},
            )
            self._write_payload(self._index_path(entry_type), payload, purpose="persistence")
        except Exception as exc:
            logger.warning("Failed updating Reader memory %s index: %s", entry_type, exc)

    def _audit(self, event: str, payload: Optional[Mapping[str, Any]] = None) -> Optional[str]:
        if not self.enable_audit_log:
            return None
        try:
            event_id = f"{utc_compact_timestamp()}_{safe_filename(event, fallback='event')}_{uuid.uuid4().hex[:8]}"
            envelope = self._build_envelope(
                entry_type="audit",
                identifier=event_id,
                payload={"event": event, "data": dict(payload or {})},
                metadata={"component": "ReaderMemory"},
            )
            return self._write_payload(self.audit_dir / f"{event_id}.json", envelope, purpose="persistence")
        except Exception as exc:
            logger.warning("Failed writing Reader memory audit event %s: %s", event, exc)
            return None

    # ------------------------------------------------------------------
    # Checkpoint API
    # ------------------------------------------------------------------

    def write_checkpoint(
        self,
        step: str,
        data: Mapping[str, Any],
        run_id: Optional[str] = None,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Persist a versioned checkpoint and return the checkpoint path."""

        step_name = self._normalize_step(step)
        normalized_run_id = self._normalize_run_id(run_id)
        timestamp = utc_compact_timestamp()
        checkpoint_id_parts = [self.checkpoint_prefix]
        if normalized_run_id:
            checkpoint_id_parts.append(normalized_run_id)
        checkpoint_id_parts.extend([timestamp, step_name, uuid.uuid4().hex[:8]])
        checkpoint_id = "_".join(checkpoint_id_parts)
        checkpoint_path = self._checkpoint_path(checkpoint_id)

        checkpoint_payload = {
            "checkpoint_id": checkpoint_id,
            "step": step_name,
            "timestamp": timestamp,
            "run_id": normalized_run_id,
            "payload": json_safe(data, redact=False),
        }
        self._assert_payload_size(checkpoint_payload, max_bytes=self.max_checkpoint_payload_bytes, purpose="checkpoint")

        envelope = self._build_envelope(
            entry_type="checkpoint",
            identifier=checkpoint_id,
            payload=checkpoint_payload,
            metadata={
                "step": step_name,
                "run_id": normalized_run_id,
                "payload_bytes": self._json_size(checkpoint_payload),
                **dict(metadata or {}),
            },
        )

        with self._lock:
            path = self._write_payload(checkpoint_path, envelope, purpose="checkpoint")
            self._audit("checkpoint_written", {"checkpoint_id": checkpoint_id, "path": path, "step": step_name, "run_id": normalized_run_id})
            self.prune_checkpoints(max_entries=self.max_checkpoint_entries)
            self._write_index("checkpoint")
            return path

    def read_checkpoint(self, checkpoint: str | Path, *, verify_integrity: bool = True) -> Dict[str, Any]:
        """Read a checkpoint by path or checkpoint id."""

        path = expand_path(checkpoint)
        if not path.exists():
            path = self._checkpoint_path(str(checkpoint))
        if not path.exists():
            raise self._raise_checkpoint_error("Reader checkpoint does not exist", {"checkpoint": str(checkpoint)})
        envelope = self._read_payload(path, purpose="checkpoint")
        if verify_integrity:
            self._verify_envelope(envelope, path=path)
        return envelope

    def latest_checkpoint(self, *, step: Optional[str] = None, run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        entries = self.list_checkpoints(step=step, run_id=run_id, limit=1, newest_first=True)
        if not entries:
            return None
        return self.read_checkpoint(entries[0]["path"])

    def list_checkpoints(
        self,
        *,
        step: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: Optional[int] = None,
        newest_first: bool = True,
    ) -> List[Dict[str, Any]]:
        entries = [entry for entry in self._scan_entries("checkpoint")]
        if step:
            step_name = self._normalize_step(step)
            entries = [entry for entry in entries if entry.step == step_name]
        if run_id:
            normalized_run_id = self._normalize_run_id(run_id)
            entries = [entry for entry in entries if entry.run_id == normalized_run_id]
        entries.sort(key=lambda item: item.created_at, reverse=newest_first)
        if limit is not None:
            entries = entries[: max(0, int(limit))]
        return [entry.to_dict() for entry in entries]

    def delete_checkpoint(self, checkpoint: str | Path) -> bool:
        path = expand_path(checkpoint)
        if not path.exists():
            path = self._checkpoint_path(str(checkpoint))
        if not path.exists():
            return False
        try:
            path.unlink()
            self._audit("checkpoint_deleted", {"path": str(path)})
            self._write_index("checkpoint")
            return True
        except Exception as exc:
            raise self._raise_checkpoint_error("Failed deleting Reader checkpoint", {"path": str(path)}, exc) from exc

    def prune_checkpoints(self, *, max_entries: Optional[int] = None, older_than_seconds: Optional[int] = None) -> int:
        limit = max_entries if max_entries is not None else self.max_checkpoint_entries
        entries = self._scan_entries("checkpoint")
        now = utc_timestamp()
        to_delete: List[ReaderMemoryEntry] = []

        if older_than_seconds is not None and older_than_seconds > 0:
            to_delete.extend([entry for entry in entries if now - entry.created_at > older_than_seconds])

        remaining = [entry for entry in entries if entry not in to_delete]
        remaining.sort(key=lambda item: item.created_at, reverse=True)
        if limit and len(remaining) > limit:
            to_delete.extend(remaining[limit:])

        deleted = 0
        for entry in dedupe_preserve_order(to_delete):
            try:
                Path(entry.path).unlink(missing_ok=True)
                deleted += 1
            except Exception as exc:
                logger.warning("Failed pruning Reader checkpoint %s: %s", entry.path, exc)
        if deleted:
            self._audit("checkpoints_pruned", {"deleted": deleted})
        return deleted

    # ------------------------------------------------------------------
    # Cache API
    # ------------------------------------------------------------------

    def set_cache(
        self,
        key_payload: Mapping[str, Any],
        value: Mapping[str, Any],
        *,
        ttl_seconds: Optional[int] = None,
        namespace: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Persist a hash-keyed cache value and return the cache path."""

        ns = safe_filename(namespace or self.cache_namespace, fallback="reader")
        key = self._cache_key(key_payload, namespace=ns)
        created_at = utc_timestamp()
        ttl = self.cache_ttl_seconds if ttl_seconds is None else coerce_int(ttl_seconds, self.cache_ttl_seconds, minimum=0)
        expires_at = created_at + ttl if ttl > 0 else None
        cache_path = self._cache_path(key)

        cache_payload = {
            "key": key,
            "namespace": ns,
            "created_at": created_at,
            "expires_at": expires_at,
            "key_payload_hash": stable_hash(key_payload),
            "key_payload": json_safe(key_payload, redact=False),
            "value": json_safe(value, redact=False),
        }
        self._assert_payload_size(cache_payload, max_bytes=self.max_cache_entry_bytes, purpose="cache")

        envelope = self._build_envelope(
            entry_type="cache",
            identifier=key,
            payload=cache_payload,
            metadata={
                "namespace": ns,
                "ttl_seconds": ttl,
                "expires_at": expires_at,
                "payload_bytes": self._json_size(cache_payload),
                **dict(metadata or {}),
            },
        )

        with self._lock:
            path = self._write_payload(cache_path, envelope, purpose="cache")
            self._audit("cache_written", {"key": key, "namespace": ns, "path": path})
            self.prune_cache(max_entries=self.max_cache_entries, expired_only=False)
            self._write_index("cache")
            return path

    def get_cache(
        self,
        key_payload: Mapping[str, Any],
        *,
        namespace: Optional[str] = None,
        default: Optional[Dict[str, Any]] = None,
        verify_integrity: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Return cached value for key payload, or ``None`` when missing/expired."""

        record = self.get_cache_record(
            key_payload,
            namespace=namespace,
            verify_integrity=verify_integrity,
            delete_expired=True,
        )
        if record is None:
            return default
        payload = record.get("payload", {})
        if not isinstance(payload, Mapping):
            raise self._raise_cache_error("Reader cache envelope payload is malformed", {"key_payload": key_payload})
        value = payload.get("value", default)
        return dict(value) if isinstance(value, Mapping) else value  # type: ignore[return-value]

    def get_cache_record(
        self,
        key_payload: Mapping[str, Any],
        *,
        namespace: Optional[str] = None,
        verify_integrity: bool = True,
        delete_expired: bool = False,
    ) -> Optional[Dict[str, Any]]:
        ns = safe_filename(namespace or self.cache_namespace, fallback="reader")
        key = self._cache_key(key_payload, namespace=ns)
        cache_path = self._cache_path(key)
        if not cache_path.exists():
            return None
        envelope = self._read_payload(cache_path, purpose="cache")
        if verify_integrity:
            self._verify_envelope(envelope, path=cache_path)
        if self._cache_envelope_expired(envelope):
            if delete_expired:
                try:
                    cache_path.unlink(missing_ok=True)
                    self._audit("cache_expired", {"key": key, "path": str(cache_path)})
                except Exception as exc:
                    logger.warning("Failed removing expired Reader cache file %s: %s", cache_path, exc)
            return None
        return envelope

    def delete_cache(self, key_payload: Mapping[str, Any], *, namespace: Optional[str] = None) -> bool:
        ns = safe_filename(namespace or self.cache_namespace, fallback="reader")
        key = self._cache_key(key_payload, namespace=ns)
        cache_path = self._cache_path(key)
        if not cache_path.exists():
            return False
        try:
            cache_path.unlink()
            self._audit("cache_deleted", {"key": key, "path": str(cache_path)})
            self._write_index("cache")
            return True
        except Exception as exc:
            raise self._raise_cache_error("Failed deleting Reader cache payload", {"path": str(cache_path)}, exc) from exc

    def invalidate_cache(self, key_payload: Mapping[str, Any], *, namespace: Optional[str] = None) -> bool:
        return self.delete_cache(key_payload, namespace=namespace)

    def cache_exists(self, key_payload: Mapping[str, Any], *, namespace: Optional[str] = None) -> bool:
        return self.get_cache_record(key_payload, namespace=namespace, delete_expired=True) is not None

    def list_cache_entries(
        self,
        *,
        namespace: Optional[str] = None,
        include_expired: bool = False,
        limit: Optional[int] = None,
        newest_first: bool = True,
    ) -> List[Dict[str, Any]]:
        entries = self._scan_entries("cache")
        if namespace:
            ns = safe_filename(namespace, fallback="reader")
            entries = [entry for entry in entries if entry.namespace == ns]
        if not include_expired:
            entries = [entry for entry in entries if not entry.expired]
        entries.sort(key=lambda item: item.created_at, reverse=newest_first)
        if limit is not None:
            entries = entries[: max(0, int(limit))]
        return [entry.to_dict() for entry in entries]

    def prune_cache(
        self,
        *,
        max_entries: Optional[int] = None,
        expired_only: bool = True,
        older_than_seconds: Optional[int] = None,
    ) -> int:
        limit = max_entries if max_entries is not None else self.max_cache_entries
        entries = self._scan_entries("cache")
        now = utc_timestamp()
        to_delete: List[ReaderMemoryEntry] = []

        if expired_only:
            to_delete.extend([entry for entry in entries if entry.expired])
        if older_than_seconds is not None and older_than_seconds > 0:
            to_delete.extend([entry for entry in entries if now - entry.created_at > older_than_seconds])

        remaining = [entry for entry in entries if entry not in to_delete]
        remaining.sort(key=lambda item: item.created_at, reverse=True)
        if not expired_only and limit and len(remaining) > limit:
            to_delete.extend(remaining[limit:])

        deleted = 0
        for entry in dedupe_preserve_order(to_delete):
            try:
                Path(entry.path).unlink(missing_ok=True)
                deleted += 1
            except Exception as exc:
                logger.warning("Failed pruning Reader cache %s: %s", entry.path, exc)
        if deleted:
            self._audit("cache_pruned", {"deleted": deleted})
        return deleted

    def clear_cache(self, *, expired_only: bool = False) -> int:
        if expired_only:
            return self.prune_cache(expired_only=True)
        deleted = 0
        for entry in self._scan_entries("cache"):
            try:
                Path(entry.path).unlink(missing_ok=True)
                deleted += 1
            except Exception as exc:
                logger.warning("Failed clearing Reader cache %s: %s", entry.path, exc)
        if deleted:
            self._audit("cache_cleared", {"deleted": deleted, "expired_only": expired_only})
            self._write_index("cache")
        return deleted

    # ------------------------------------------------------------------
    # Scanning, stats, and maintenance
    # ------------------------------------------------------------------

    def _cache_envelope_expired(self, envelope: Mapping[str, Any]) -> bool:
        payload = envelope.get("payload", {})
        metadata = envelope.get("metadata", {})
        expires_at = None
        if isinstance(payload, Mapping):
            expires_at = payload.get("expires_at")
        if expires_at is None and isinstance(metadata, Mapping):
            expires_at = metadata.get("expires_at")
        if expires_at is None:
            created_at = None
            if isinstance(payload, Mapping):
                created_at = payload.get("created_at")
            if created_at and self.cache_ttl_seconds > 0:
                expires_at = float(created_at) + self.cache_ttl_seconds
        return bool(expires_at is not None and utc_timestamp() > float(expires_at))

    def _scan_entries(self, entry_type: str) -> List[ReaderMemoryEntry]:
        directory = self.checkpoint_dir if entry_type == "checkpoint" else self.cache_dir
        pattern = "*.json"
        index_names = {DEFAULT_INDEX_FILENAME, self.checkpoint_index_file, self.cache_index_file}
        entries: List[ReaderMemoryEntry] = []
        for path in directory.glob(pattern):
            if path.name in index_names:
                continue
            try:
                payload = self._read_payload(path, purpose=entry_type if entry_type in {"checkpoint", "cache"} else "persistence")
                stat = path.stat()
                metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), Mapping) else {}
                body = payload.get("payload", {}) if isinstance(payload.get("payload", {}), Mapping) else {}
                created_at = float(payload.get("created_at", stat.st_mtime) or stat.st_mtime)
                expired = self._cache_envelope_expired(payload) if entry_type == "cache" else False
                entries.append(
                    ReaderMemoryEntry(
                        entry_type=entry_type,
                        identifier=str(payload.get("identifier", path.stem)),
                        path=str(path),
                        created_at=created_at,
                        size_bytes=int(stat.st_size),
                        namespace=str(body.get("namespace") or metadata.get("namespace")) if entry_type == "cache" else None,
                        step=str(body.get("step") or metadata.get("step")) if entry_type == "checkpoint" else None,
                        run_id=str(body.get("run_id") or metadata.get("run_id")) if entry_type == "checkpoint" and (body.get("run_id") or metadata.get("run_id")) else None,
                        expired=expired,
                    )
                )
            except Exception as exc:
                logger.warning("Skipping unreadable Reader %s entry %s: %s", entry_type, path, exc)
        return entries

    def prune(self) -> Dict[str, int]:
        with self._lock:
            result = {
                "cache_deleted": self.prune_cache(max_entries=self.max_cache_entries, expired_only=False),
                "checkpoint_deleted": self.prune_checkpoints(max_entries=self.max_checkpoint_entries),
            }
            self._write_index("cache")
            self._write_index("checkpoint")
            return result

    def stats(self) -> Dict[str, Any]:
        checkpoint_entries = self._scan_entries("checkpoint")
        cache_entries = self._scan_entries("cache")
        expired_cache_count = sum(1 for entry in cache_entries if entry.expired)
        return {
            "schema_version": READER_MEMORY_SCHEMA_VERSION,
            "paths": self.paths.to_dict(),
            "checkpoint_count": len(checkpoint_entries),
            "cache_count": len(cache_entries),
            "expired_cache_count": expired_cache_count,
            "checkpoint_bytes": sum(entry.size_bytes for entry in checkpoint_entries),
            "cache_bytes": sum(entry.size_bytes for entry in cache_entries),
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "max_cache_entries": self.max_cache_entries,
            "max_checkpoint_entries": self.max_checkpoint_entries,
            "redact_sensitive_payloads": self.redact_sensitive_payloads,
            "integrity_enabled": self.enable_integrity_hash,
        }

    def find_checkpoints(self, pattern: str = "*", *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        matched = [entry for entry in self._scan_entries("checkpoint") if fnmatch.fnmatch(Path(entry.path).name, pattern)]
        matched.sort(key=lambda item: item.created_at, reverse=True)
        if limit is not None:
            matched = matched[: max(0, int(limit))]
        return [entry.to_dict() for entry in matched]

    def healthcheck(self) -> Dict[str, Any]:
        """Validate directories and basic read/write behavior."""

        probe_id = f"healthcheck_{uuid.uuid4().hex[:8]}"
        probe_path = self.audit_dir / f"{probe_id}.json"
        try:
            envelope = self._build_envelope(
                entry_type="healthcheck",
                identifier=probe_id,
                payload={"ok": True, "timestamp": utc_timestamp()},
            )
            self._write_payload(probe_path, envelope, purpose="persistence")
            loaded = self._read_payload(probe_path, purpose="persistence")
            self._verify_envelope(loaded, path=probe_path)
            probe_path.unlink(missing_ok=True)
            return {"status": "ok", "paths": self.paths.to_dict(), "stats": self.stats()}
        except Exception as exc:
            return build_error_result(exc, operation="reader_memory_healthcheck", include_debug=True)


if __name__ == "__main__":
    print("\n=== Running Reader Memory ===\n")
    printer.status("TEST", "Reader Memory initialized", "info")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        memory = ReaderMemory(
            config={
                "checkpoint_dir": str(base / "checkpoints"),
                "cache_dir": str(base / "cache"),
                "audit_dir": str(base / "audit"),
                "cache_ttl_seconds": 1,
                "max_cache_entries": 5,
                "max_checkpoint_entries": 5,
                "redact_sensitive_payloads": True,
                "enable_integrity_hash": True,
                "enable_audit_log": True,
                "write_indexes": True,
                "prune_on_init": False,
            }
        )

        checkpoint_path = memory.write_checkpoint(
            "parse",
            {"source": "sample.txt", "token": "secret", "status": "ok"},
            run_id="run-001",
        )
        assert Path(checkpoint_path).exists()
        checkpoint = memory.read_checkpoint(checkpoint_path)
        assert checkpoint["payload"]["step"] == "parse"
        assert memory.latest_checkpoint(step="parse", run_id="run-001") is not None
        assert len(memory.list_checkpoints(step="parse")) == 1

        key_payload = {"action": "parse", "file": "sample.txt"}
        cache_path = memory.set_cache(key_payload, {"content": "hello", "metadata": {"size": 5}})
        assert Path(cache_path).exists()
        cached = memory.get_cache(key_payload)
        assert cached is not None
        assert cached["content"] == "hello"
        assert memory.cache_exists(key_payload) is True
        assert len(memory.list_cache_entries()) == 1

        stats = memory.stats()
        assert stats["checkpoint_count"] >= 1
        assert stats["cache_count"] >= 1
        assert memory.healthcheck()["status"] == "ok"

        assert memory.delete_cache(key_payload) is True
        assert memory.get_cache(key_payload) is None
        assert memory.delete_checkpoint(checkpoint_path) is True

    print("\n=== Test ran successfully ===\n")
