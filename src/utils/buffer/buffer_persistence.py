from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import struct
import tempfile
import time
import numpy as np  # pyright: ignore[reportMissingImports]

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, Iterator, Mapping, MutableMapping, Optional, Protocol, Union

from .utils.config_loader import get_config_section, load_global_config
from .utils.buffer_errors import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Buffer Persistence")
printer = PrettyPrinter()

MAGIC = b"SLAI_BUFFER_CHECKPOINT\x01\n"
HEADER_STRUCT = struct.Struct(">Q")
CURRENT_FORMAT_VERSION = "slai.buffer.checkpoint/1"
DEFAULT_SCHEMA_VERSION = "buffer_checkpoint.v1"

EncryptionHook = Callable[[bytes], bytes]
DecryptionHook = Callable[[bytes], bytes]


class CheckpointAdapter(Protocol):
    """Callable contract for schema migration adapters."""

    def __call__(self, checkpoint: "BufferCheckpoint") -> "BufferCheckpoint":
        """Return a migrated checkpoint."""
        ...


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    return str(value)


def _safe_metadata(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if metadata is None:
        return {}
    return json.loads(json.dumps(dict(metadata), default=_json_default, sort_keys=True))


def _telemetry_increment(telemetry: Any, name: str, amount: float = 1.0) -> None:
    if telemetry is not None and hasattr(telemetry, "increment"):
        telemetry.increment(name, amount)


def _telemetry_observe(telemetry: Any, name: str, value: float) -> None:
    if telemetry is not None and hasattr(telemetry, "observe"):
        telemetry.observe(name, value)


def _telemetry_snapshot(telemetry: Any) -> Dict[str, Any]:
    if telemetry is None or not hasattr(telemetry, "snapshot"):
        return {}
    try:
        return dict(telemetry.snapshot())
    except Exception as exc:
        logger.warning("Unable to attach telemetry snapshot to checkpoint: %s", exc)
        return {}


@dataclass(frozen=True)
class BufferCheckpointConfig:
    """Config contract for shared buffer checkpoint persistence."""

    enabled: bool = True
    default_directory: str = "checkpoints/buffer"
    schema_version: str = DEFAULT_SCHEMA_VERSION
    serializer: str = "pickle"
    allow_pickle: bool = True
    compression: str = "gzip"
    compression_level: int = 6
    atomic_writes: bool = True
    create_directories: bool = True
    checksum_algorithm: str = "sha256"
    encryption_enabled: bool = False
    encryption_mode: str = "hook"
    require_encryption_hook: bool = False
    lock_timeout_seconds: float = 5.0
    allow_legacy_npz: bool = True
    strict_schema_version: bool = False

    @classmethod
    def from_config(cls, user_config: Optional[Mapping[str, Any]] = None) -> "BufferCheckpointConfig":
        load_global_config()
        cfg = dict(get_config_section("persistence") or {})
        if user_config:
            cfg.update(dict(user_config.get("persistence", {}) if "persistence" in user_config else user_config))

        serializer = str(cfg.get("serializer", "pickle")).strip().lower()
        compression = str(cfg.get("compression", "gzip")).strip().lower()
        checksum_algorithm = str(cfg.get("checksum_algorithm", "sha256")).strip().lower()
        encryption_mode = str(cfg.get("encryption_mode", "hook")).strip().lower()

        if serializer not in {"pickle", "json"}:
            raise ConfigValueError("persistence.serializer", serializer, "one of: pickle, json", section="persistence")
        if compression not in {"none", "gzip"}:
            raise ConfigValueError("persistence.compression", compression, "one of: none, gzip", section="persistence")
        if checksum_algorithm != "sha256":
            raise ConfigValueError("persistence.checksum_algorithm", checksum_algorithm, "sha256", section="persistence")
        if encryption_mode != "hook":
            raise ConfigValueError("persistence.encryption_mode", encryption_mode, "hook", section="persistence")

        compression_level = int(cfg.get("compression_level", 6))
        if not (0 <= compression_level <= 9):
            raise ConfigValueError("persistence.compression_level", compression_level, "integer in [0, 9]", section="persistence")

        lock_timeout = float(cfg.get("lock_timeout_seconds", 5.0))
        if lock_timeout < 0:
            raise ConfigValueError("persistence.lock_timeout_seconds", lock_timeout, ">= 0", section="persistence")

        schema_version = str(cfg.get("schema_version", DEFAULT_SCHEMA_VERSION)).strip()
        if not schema_version:
            raise ConfigValueError("persistence.schema_version", schema_version, "non-empty string", section="persistence")

        return cls(
            enabled=bool(cfg.get("enabled", True)),
            default_directory=str(cfg.get("default_directory", "checkpoints/buffer")),
            schema_version=schema_version,
            serializer=serializer,
            allow_pickle=bool(cfg.get("allow_pickle", True)),
            compression=compression,
            compression_level=compression_level,
            atomic_writes=bool(cfg.get("atomic_writes", True)),
            create_directories=bool(cfg.get("create_directories", True)),
            checksum_algorithm=checksum_algorithm,
            encryption_enabled=bool(cfg.get("encryption_enabled", False)),
            encryption_mode=encryption_mode,
            require_encryption_hook=bool(cfg.get("require_encryption_hook", False)),
            lock_timeout_seconds=lock_timeout,
            allow_legacy_npz=bool(cfg.get("allow_legacy_npz", True)),
            strict_schema_version=bool(cfg.get("strict_schema_version", False)),
        )


@dataclass(frozen=True)
class BufferCheckpointManifest:
    """Stable, JSON-serializable checkpoint envelope metadata."""

    format_version: str
    schema_version: str
    component_name: str
    created_at: str
    serializer: str
    compression: str
    encrypted: bool
    checksum_algorithm: str
    payload_sha256: str
    stored_payload_sha256: str
    payload_nbytes: int
    stored_payload_nbytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    telemetry_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BufferCheckpointManifest":
        required = {
            "format_version", "schema_version", "component_name", "created_at", "serializer",
            "compression", "encrypted", "checksum_algorithm", "payload_sha256",
            "stored_payload_sha256", "payload_nbytes", "stored_payload_nbytes",
        }
        missing = sorted(required.difference(payload.keys()))
        if missing:
            raise BufferSerializationError("manifest", f"missing manifest fields: {missing}")
        return cls(
            format_version=str(payload["format_version"]),
            schema_version=str(payload["schema_version"]),
            component_name=str(payload["component_name"]),
            created_at=str(payload["created_at"]),
            serializer=str(payload["serializer"]),
            compression=str(payload["compression"]),
            encrypted=bool(payload["encrypted"]),
            checksum_algorithm=str(payload["checksum_algorithm"]),
            payload_sha256=str(payload["payload_sha256"]),
            stored_payload_sha256=str(payload["stored_payload_sha256"]),
            payload_nbytes=int(payload["payload_nbytes"]),
            stored_payload_nbytes=int(payload["stored_payload_nbytes"]),
            metadata=dict(payload.get("metadata", {}) or {}),
            telemetry_snapshot=dict(payload.get("telemetry_snapshot", {}) or {}),
        )


@dataclass(frozen=True)
class BufferCheckpoint:
    """Loaded checkpoint result containing state and manifest."""

    state: Any
    manifest: BufferCheckpointManifest
    path: Optional[str] = None

    @property
    def schema_version(self) -> str:
        return self.manifest.schema_version

    @property
    def component_name(self) -> str:
        return self.manifest.component_name

    def to_summary(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "component_name": self.component_name,
            "schema_version": self.schema_version,
            "format_version": self.manifest.format_version,
            "serializer": self.manifest.serializer,
            "compression": self.manifest.compression,
            "encrypted": self.manifest.encrypted,
            "payload_nbytes": self.manifest.payload_nbytes,
            "stored_payload_nbytes": self.manifest.stored_payload_nbytes,
            "created_at": self.manifest.created_at,
        }


class BufferCheckpointIO:
    """Shared, versioned save/load service for buffer state.

    The class owns checkpoint I/O for buffer modules so replay, reservoir,
    sequence, and network buffers do not each invent brittle persistence formats.
    It writes a small JSON manifest plus a serialized payload, supports gzip,
    exposes encryption hooks without pretending to implement crypto itself, and
    provides schema-version adapters for backward-compatible loads.
    """

    def __init__(
        self,
        user_config: Optional[Mapping[str, Any]] = None,
        *,
        telemetry: Optional[Any] = None,
        encrypt_hook: Optional[EncryptionHook] = None,
        decrypt_hook: Optional[DecryptionHook] = None,
    ) -> None:
        self.config = BufferCheckpointConfig.from_config(user_config=user_config)
        self.telemetry = telemetry
        self.encrypt_hook = encrypt_hook
        self.decrypt_hook = decrypt_hook
        self._lock = RLock()
        self._adapters: Dict[str, CheckpointAdapter] = {}

    def register_adapter(self, from_schema_version: str, adapter: CheckpointAdapter) -> None:
        key = str(from_schema_version).strip()
        if not key:
            raise ConfigValueError("from_schema_version", from_schema_version, "non-empty string")
        self._adapters[key] = adapter

    @contextmanager
    def _optional_external_lock(self, lock: Optional[Any], operation: str) -> Iterator[None]:
        if lock is None:
            yield
            return
        start = time.perf_counter()
        timeout = self.config.lock_timeout_seconds
        acquired = lock.acquire(timeout=timeout) if timeout > 0 else lock.acquire(blocking=False)
        waited = time.perf_counter() - start
        self._record_lock(operation, waited, acquired)
        if not acquired:
            raise BufferLockTimeoutError(operation=operation, timeout_seconds=timeout)
        try:
            yield
        finally:
            lock.release()

    def _record_lock(self, operation: str, waited: float, acquired: bool) -> None:
        if self.telemetry is not None and hasattr(self.telemetry, "record_lock_contention"):
            self.telemetry.record_lock_contention(operation, waited, acquired=acquired)
            return
        _telemetry_observe(self.telemetry, "lock_wait_seconds", waited)
        if not acquired or waited > 0:
            _telemetry_increment(self.telemetry, "lock_contention_count")

    def _resolve_path(self, filepath: Optional[Union[str, Path]], component_name: str) -> Path:
        if filepath is None:
            filename = f"{component_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.slai-buffer"
            path = Path(self.config.default_directory) / filename
        else:
            path = Path(filepath)
        return path.expanduser().resolve()

    def _serialize(self, state: Any, serializer: Optional[str] = None) -> bytes:
        selected = serializer or self.config.serializer
        if selected == "pickle":
            if not self.config.allow_pickle:
                raise BufferSerializationError("serialize", "pickle serializer is disabled by persistence.allow_pickle")
            return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        if selected == "json":
            return json.dumps(state, default=_json_default, sort_keys=True, separators=(",", ":")).encode("utf-8")
        raise BufferSerializationError("serialize", f"unsupported serializer: {selected}")

    def _deserialize(self, payload: bytes, serializer: str) -> Any:
        if serializer == "pickle":
            if not self.config.allow_pickle:
                raise BufferSerializationError("deserialize", "pickle deserialization is disabled by persistence.allow_pickle")
            return pickle.loads(payload)
        if serializer == "json":
            return json.loads(payload.decode("utf-8"))
        raise BufferSerializationError("deserialize", f"unsupported serializer: {serializer}")

    def _encode_payload(self, raw: bytes, *, compression: str, encryption_enabled: bool) -> bytes:
        payload = raw
        if compression == "gzip":
            payload = gzip.compress(payload, compresslevel=self.config.compression_level)
        elif compression != "none":
            raise BufferSerializationError("compress", f"unsupported compression: {compression}")

        if encryption_enabled:
            if self.encrypt_hook is None:
                raise BufferSerializationError("encrypt", "encryption_enabled=True requires encrypt_hook")
            payload = self.encrypt_hook(payload)
            if not isinstance(payload, bytes):
                raise BufferSerializationError("encrypt", "encrypt_hook must return bytes")
        elif self.config.require_encryption_hook:
            raise BufferSerializationError("encrypt", "persistence requires encryption but encryption was not enabled")
        return payload

    def _decode_payload(self, stored: bytes, manifest: BufferCheckpointManifest) -> bytes:
        payload = stored
        if manifest.encrypted:
            if self.decrypt_hook is None:
                raise BufferSerializationError("decrypt", "encrypted checkpoint requires decrypt_hook")
            payload = self.decrypt_hook(payload)
            if not isinstance(payload, bytes):
                raise BufferSerializationError("decrypt", "decrypt_hook must return bytes")

        if manifest.compression == "gzip":
            payload = gzip.decompress(payload)
        elif manifest.compression != "none":
            raise BufferSerializationError("decompress", f"unsupported compression: {manifest.compression}")
        return payload

    def save_checkpoint(
        self,
        state: Any,
        filepath: Optional[Union[str, Path]] = None,
        *,
        component_name: str = "buffer",
        schema_version: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        telemetry: Optional[Any] = None,
        lock: Optional[Any] = None,
        compression: Optional[str] = None,
        encryption_enabled: Optional[bool] = None,
    ) -> Path:
        if not self.config.enabled:
            raise BufferSaveError(str(filepath or "<auto>"), "persistence is disabled")

        active_telemetry = telemetry or self.telemetry
        target = self._resolve_path(filepath, component_name=component_name)
        started = time.perf_counter()

        try:
            with self._optional_external_lock(lock, operation=f"{component_name}.checkpoint_save"):
                with self._lock:
                    if self.config.create_directories:
                        target.parent.mkdir(parents=True, exist_ok=True)

                    selected_compression = (compression or self.config.compression).strip().lower()
                    selected_encryption = self.config.encryption_enabled if encryption_enabled is None else bool(encryption_enabled)
                    raw = self._serialize(state)
                    stored = self._encode_payload(raw, compression=selected_compression, encryption_enabled=selected_encryption)

                    manifest = BufferCheckpointManifest(
                        format_version=CURRENT_FORMAT_VERSION,
                        schema_version=schema_version or self.config.schema_version,
                        component_name=str(component_name),
                        created_at=_utcnow_iso(),
                        serializer=self.config.serializer,
                        compression=selected_compression,
                        encrypted=selected_encryption,
                        checksum_algorithm=self.config.checksum_algorithm,
                        payload_sha256=_sha256(raw),
                        stored_payload_sha256=_sha256(stored),
                        payload_nbytes=len(raw),
                        stored_payload_nbytes=len(stored),
                        metadata=_safe_metadata(metadata),
                        telemetry_snapshot=_safe_metadata(_telemetry_snapshot(active_telemetry)),
                    )
                    header = json.dumps(manifest.to_dict(), default=_json_default, sort_keys=True).encode("utf-8")
                    blob = MAGIC + HEADER_STRUCT.pack(len(header)) + header + stored

                    if self.config.atomic_writes:
                        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
                        try:
                            with os.fdopen(fd, "wb") as handle:
                                handle.write(blob)
                                handle.flush()
                                os.fsync(handle.fileno())
                            os.replace(tmp_name, target)
                        finally:
                            if os.path.exists(tmp_name):
                                os.remove(tmp_name)
                    else:
                        with open(target, "wb") as handle:
                            handle.write(blob)

                    _telemetry_increment(active_telemetry, "checkpoint_save_count")
                    _telemetry_observe(active_telemetry, "checkpoint_payload_bytes", len(raw))
                    _telemetry_observe(active_telemetry, "checkpoint_stored_bytes", len(stored))
                    logger.info("Saved %s checkpoint to %s", component_name, target)
                    return target
        except BufferPersistenceError:
            _telemetry_increment(active_telemetry, "checkpoint_save_error_count")
            raise
        except Exception as exc:
            _telemetry_increment(active_telemetry, "checkpoint_save_error_count")
            raise BufferSaveError(str(target), str(exc)) from exc
        finally:
            _telemetry_observe(active_telemetry, "checkpoint_save_latency_seconds", time.perf_counter() - started)

    def load_checkpoint(
        self,
        filepath: Union[str, Path],
        *,
        expected_component: Optional[str] = None,
        telemetry: Optional[Any] = None,
        lock: Optional[Any] = None,
        apply_adapters: bool = True,
    ) -> BufferCheckpoint:
        active_telemetry = telemetry or self.telemetry
        source = Path(filepath).expanduser().resolve()
        started = time.perf_counter()

        try:
            with self._optional_external_lock(lock, operation="checkpoint_load"):
                with self._lock:
                    checkpoint = self._read_checkpoint(source)
                    if expected_component and checkpoint.component_name != expected_component:
                        raise BufferLoadError(str(source), f"expected component '{expected_component}', got '{checkpoint.component_name}'")

                    if apply_adapters:
                        checkpoint = self._apply_adapter(checkpoint)
                    elif self.config.strict_schema_version and checkpoint.schema_version != self.config.schema_version:
                        raise BufferLoadError(str(source), f"schema version {checkpoint.schema_version} does not match {self.config.schema_version}")

                    _telemetry_increment(active_telemetry, "checkpoint_load_count")
                    _telemetry_observe(active_telemetry, "checkpoint_loaded_payload_bytes", checkpoint.manifest.payload_nbytes)
                    logger.info("Loaded %s checkpoint from %s", checkpoint.component_name, source)
                    return checkpoint
        except BufferPersistenceError:
            _telemetry_increment(active_telemetry, "checkpoint_load_error_count")
            raise
        except Exception as exc:
            _telemetry_increment(active_telemetry, "checkpoint_load_error_count")
            raise BufferLoadError(str(source), str(exc)) from exc
        finally:
            _telemetry_observe(active_telemetry, "checkpoint_load_latency_seconds", time.perf_counter() - started)

    def _read_checkpoint(self, source: Path) -> BufferCheckpoint:
        if not source.exists():
            raise BufferLoadError(str(source), "file does not exist")
        with open(source, "rb") as handle:
            prefix = handle.read(len(MAGIC))
            if prefix != MAGIC:
                if self.config.allow_legacy_npz:
                    return self._load_legacy_npz(source)
                raise BufferLoadError(str(source), "unrecognized checkpoint magic header")

            header_len_raw = handle.read(HEADER_STRUCT.size)
            if len(header_len_raw) != HEADER_STRUCT.size:
                raise BufferLoadError(str(source), "truncated checkpoint header length")
            header_len = HEADER_STRUCT.unpack(header_len_raw)[0]
            header_bytes = handle.read(header_len)
            if len(header_bytes) != header_len:
                raise BufferLoadError(str(source), "truncated checkpoint header")
            stored = handle.read()

        manifest = BufferCheckpointManifest.from_dict(json.loads(header_bytes.decode("utf-8")))
        if manifest.format_version != CURRENT_FORMAT_VERSION:
            raise BufferLoadError(str(source), f"unsupported format_version: {manifest.format_version}")
        if _sha256(stored) != manifest.stored_payload_sha256:
            raise BufferLoadError(str(source), "stored payload checksum mismatch")

        raw = self._decode_payload(stored, manifest)
        if _sha256(raw) != manifest.payload_sha256:
            raise BufferLoadError(str(source), "decoded payload checksum mismatch")
        state = self._deserialize(raw, manifest.serializer)
        return BufferCheckpoint(state=state, manifest=manifest, path=str(source))

    def _apply_adapter(self, checkpoint: BufferCheckpoint) -> BufferCheckpoint:
        if checkpoint.schema_version == self.config.schema_version:
            return checkpoint
        adapter = self._adapters.get(checkpoint.schema_version)
        if adapter is not None:
            migrated = adapter(checkpoint)
            if not isinstance(migrated, BufferCheckpoint):
                raise BufferSerializationError("adapter", "checkpoint adapter must return BufferCheckpoint")
            return migrated
        if self.config.strict_schema_version:
            raise BufferLoadError(checkpoint.path or "<memory>", f"no adapter from {checkpoint.schema_version} to {self.config.schema_version}")
        return checkpoint

    def _load_legacy_npz(self, source: Path) -> BufferCheckpoint:
        try:
            data = np.load(str(source), allow_pickle=True)
            meta = data["meta"].item() if "meta" in data.files else {}
            state: Dict[str, Any] = {
                "buffer": data["buffer"].tolist() if "buffer" in data.files else [],
                "timestamps": data["timestamps"].tolist() if "timestamps" in data.files else [],
                "priorities": data["priorities"].tolist() if "priorities" in data.files else [],
                "meta": meta,
            }
            manifest = BufferCheckpointManifest(
                format_version="legacy.npz",
                schema_version="distributed_replay_buffer.v0.npz",
                component_name="distributed_replay_buffer",
                created_at=_utcnow_iso(),
                serializer="npz",
                compression="npz_compressed",
                encrypted=False,
                checksum_algorithm="sha256",
                payload_sha256="",
                stored_payload_sha256=_sha256(source.read_bytes()),
                payload_nbytes=0,
                stored_payload_nbytes=source.stat().st_size,
                metadata={"legacy": True, "source": "DistributedReplayBuffer.save(np.savez_compressed)"},
                telemetry_snapshot={},
            )
            return BufferCheckpoint(state=state, manifest=manifest, path=str(source))
        except Exception as exc:
            raise BufferLoadError(str(source), f"legacy npz adapter failed: {exc}") from exc

    def save_state(self, filepath: Union[str, Path], state: Any, **kwargs: Any) -> Path:
        """Backward-friendly alias around save_checkpoint."""
        return self.save_checkpoint(state=state, filepath=filepath, **kwargs)

    def load_state(self, filepath: Union[str, Path], **kwargs: Any) -> Any:
        """Load and return only the state payload."""
        return self.load_checkpoint(filepath, **kwargs).state


# ---------------------------------------------------------------------------
# Lightweight integration helpers
# ---------------------------------------------------------------------------
def build_checkpoint_io(
    user_config: Optional[Mapping[str, Any]] = None,
    *,
    telemetry: Optional[Any] = None,
    encrypt_hook: Optional[EncryptionHook] = None,
    decrypt_hook: Optional[DecryptionHook] = None,
) -> BufferCheckpointIO:
    return BufferCheckpointIO(
        user_config=user_config,
        telemetry=telemetry,
        encrypt_hook=encrypt_hook,
        decrypt_hook=decrypt_hook,
    )


def distributed_replay_state(buffer: Any) -> Dict[str, Any]:
    """Capture the known DistributedReplayBuffer persistence surface."""
    return {
        "buffer": list(getattr(buffer, "buffer", [])),
        "timestamps": list(getattr(buffer, "timestamps", [])),
        "priorities": [(-p[0], p[1]) for p in getattr(buffer, "priorities", [])],
        "meta": {
            "capacity": getattr(buffer, "capacity", None),
            "prioritization_alpha": getattr(buffer, "alpha", None),
            "staleness_threshold_seconds": getattr(getattr(buffer, "staleness_threshold", None), "total_seconds", lambda: None)(),
            "reward_stats": dict(getattr(buffer, "reward_stats", {}) or {}),
            "fairness_stats": dict(getattr(buffer, "fairness_stats", {}) or {}),
            "metric_provenance": dict(getattr(buffer, "metric_provenance", {}) or {}),
        },
    }


__all__ = [
    "build_checkpoint_io",
    "BufferCheckpointIO",
    "build_checkpoint_io",
]


if __name__ == "__main__":
    print("\n=== Running  Buffer Persistence ===\n")
    printer.status("TEST", " Buffer Persistence initialized", "info")

    import tempfile as _tempfile

    cfg = {"persistence": {"serializer": "pickle", "compression": "gzip", "strict_schema_version": True}}
    io = BufferCheckpointIO(user_config=cfg)
    state = {"buffer": [("agent", [1.0], 0, 1.5, [2.0], False)], "priorities": [1.0]}
    with _tempfile.TemporaryDirectory() as d:
        path = Path(d) / "buffer.slai-buffer"
        saved = io.save_checkpoint(state, path, component_name="test_buffer", metadata={"case": "roundtrip"})
        ckpt = io.load_checkpoint(saved, expected_component="test_buffer")
        assert ckpt.state == state
        assert ckpt.schema_version == io.config.schema_version
        assert ckpt.manifest.metadata["case"] == "roundtrip"
        printer.status("ROUNDTRIP", f"Saved and loaded {saved.name}", "success")

    print("\n=== Test ran successfully ===\n")
