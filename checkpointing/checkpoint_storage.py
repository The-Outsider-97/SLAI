"""Durable local-filesystem storage for SLAI checkpoints.

The storage layer owns path containment, file observation and hashing, atomic
writes, cooperative locking, staging/commit, archival, non-destructive restore,
and deletion.  It does not understand model objects, codec payloads, manifest
semantics, retention policy, or checkpoint selection.

Directory publication is atomic only when the staging directory and final
checkpoint directory reside on the same filesystem.  This implementation
guarantees that condition by creating all staging directories directly beneath
``base_dir``.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import socket
import stat
import tarfile
import time
import uuid

from contextlib import AbstractContextManager
from pathlib import Path, PurePosixPath
from typing import BinaryIO, IO, Iterable, Iterator, Mapping

from .checkpoint_errors import *
from .checkpoint_types import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]


logger = get_logger("Checkpoint Storage")
printer = PrettyPrinter()


_COPY_CHUNK_SIZE = 1024 * 1024
_LOCK_DIRECTORY = ".locks"
_GLOBAL_LOCK_NAME = "checkpointing.lock"
_STAGING_PREFIX = ".staging-"
_BACKUP_PREFIX = ".backup-"
_DELETE_PREFIX = ".deleting-"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _domain_version(value: str) -> str:
    try:
        return validate_version(value)
    except ValueError as exc:
        raise CheckpointVersionError(
            str(exc),
            operation=CheckpointOperation.LIST,
            stage=CheckpointStage.VALIDATION,
            version=value if isinstance(value, str) else None,
            details={"reason": "invalid_version"},
        ) from exc


def ensure_directory(
    path: str | os.PathLike[str],
    *,
    mode: int = 0o700,
) -> Path:
    """Create and return an absolute directory, rejecting non-directories."""

    candidate = Path(path).expanduser()
    try:
        candidate.mkdir(parents=True, exist_ok=True, mode=mode)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise CheckpointStorageError(
            f"failed to create checkpoint directory: {candidate}",
            operation=CheckpointOperation.SAVE,
            stage=CheckpointStage.STAGING,
            path=candidate,
            retryable=exc.errno in {errno.EAGAIN, errno.EBUSY, errno.ENOSPC},
        ) from exc
    if not resolved.is_dir():
        raise CheckpointPathError(
            "checkpoint base path is not a directory",
            stage=CheckpointStage.VALIDATION,
            path=resolved,
        )
    return resolved


def resolve_checkpoint_path(base_dir: Path, version: str) -> Path:
    """Resolve the direct child directory for a validated checkpoint version."""

    safe_version = _domain_version(version)
    root = Path(base_dir).expanduser().resolve(strict=False)
    candidate = root / safe_version
    resolved = candidate.resolve(strict=False)
    if resolved.parent != root:
        raise CheckpointPathError(
            "resolved checkpoint path is not a direct child of base_dir",
            stage=CheckpointStage.VALIDATION,
            path=resolved,
            version=safe_version,
        )
    if candidate.exists() and candidate.is_symlink():
        raise CheckpointPathError(
            "checkpoint directories cannot be symbolic links",
            stage=CheckpointStage.VALIDATION,
            path=candidate,
            version=safe_version,
        )
    return candidate


def resolve_artifact_path(
    checkpoint_dir: Path,
    relative_path: str,
    *,
    must_exist: bool = False,
    allow_manifest: bool = False,
) -> Path:
    """Resolve an artifact beneath ``checkpoint_dir`` without symlink escape."""

    try:
        canonical = validate_relative_path(
            relative_path,
            allow_manifest=allow_manifest,
        )
    except ValueError as exc:
        raise CheckpointPathError(
            str(exc),
            stage=CheckpointStage.VALIDATION,
            path=checkpoint_dir,
            details={"relative_path": relative_path},
        ) from exc

    root = Path(checkpoint_dir).resolve(strict=False)
    parts = PurePosixPath(canonical).parts
    candidate = root.joinpath(*parts)

    current = root
    for part in parts:
        current = current / part
        if current.exists() or current.is_symlink():
            try:
                mode = current.lstat().st_mode
            except OSError as exc:
                raise CheckpointPathError(
                    "failed to inspect artifact path",
                    stage=CheckpointStage.VALIDATION,
                    path=current,
                    details={"relative_path": canonical},
                ) from exc
            if stat.S_ISLNK(mode):
                raise CheckpointPathError(
                    "symbolic links are not permitted in checkpoint artifacts",
                    stage=CheckpointStage.VALIDATION,
                    path=current,
                    details={"relative_path": canonical},
                )

    resolved = candidate.resolve(strict=False)
    if not _is_relative_to(resolved, root):
        raise CheckpointPathError(
            "artifact path escapes the checkpoint directory",
            stage=CheckpointStage.VALIDATION,
            path=resolved,
            details={"relative_path": canonical},
        )
    if must_exist:
        if not candidate.exists():
            raise CheckpointNotFoundError(
                "checkpoint artifact does not exist",
                stage=CheckpointStage.INTEGRITY,
                path=candidate,
                details={"relative_path": canonical},
            )
        if not candidate.is_file():
            raise CheckpointPathError(
                "checkpoint artifact is not a regular file",
                stage=CheckpointStage.INTEGRITY,
                path=candidate,
                details={"relative_path": canonical},
            )
    return candidate


def fsync_directory(path: Path, *, strict: bool = True) -> None:
    """Synchronize directory entries on platforms that support it."""

    if os.name == "nt":
        # Windows does not expose a portable directory fsync through Python.
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(str(path), flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        if strict:
            raise CheckpointStorageError(
                "failed to synchronize checkpoint directory",
                stage=CheckpointStage.COMMIT,
                path=path,
                retryable=exc.errno in {errno.EAGAIN, errno.EBUSY},
            ) from exc


def _fsync_file(path: Path) -> None:
    if os.name == "nt":          # Windows does not support fsync on read-only handles
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def fsync_tree(root: Path, *, strict: bool = True) -> None:
    """Synchronize every regular file and directory beneath ``root``."""

    try:
        for directory, dir_names, file_names in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            for name in dir_names:
                path = directory_path / name
                if path.is_symlink():
                    raise CheckpointPathError(
                        "symbolic links are not permitted in a staged checkpoint",
                        stage=CheckpointStage.COMMIT,
                        path=path,
                    )
            for name in file_names:
                path = directory_path / name
                mode = path.lstat().st_mode
                if not stat.S_ISREG(mode):
                    raise CheckpointPathError(
                        "only regular files are permitted in a checkpoint",
                        stage=CheckpointStage.COMMIT,
                        path=path,
                    )
                _fsync_file(path)
            fsync_directory(directory_path, strict=strict)
    except CheckpointStorageError:
        raise
    except OSError as exc:
        if strict:
            raise CheckpointStorageError(
                "failed to synchronize staged checkpoint contents",
                stage=CheckpointStage.COMMIT,
                path=root,
                retryable=exc.errno in {errno.EAGAIN, errno.EBUSY, errno.ENOSPC},
            ) from exc


def atomic_write_bytes(
    data: bytes,
    path: Path,
    *,
    durable: bool = True,
    mode: int = 0o600,
) -> None:
    """Atomically replace one file using a same-directory temporary file."""

    if not isinstance(data, bytes):
        raise TypeError("atomic_write_bytes requires bytes")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise CheckpointPathError(
            "refusing to replace a symbolic link",
            stage=CheckpointStage.STAGING,
            path=target,
        )

    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(str(temporary), flags, mode)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(data)
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
        os.replace(temporary, target)
        if durable:
            fsync_directory(target.parent)
    except OSError as exc:
        raise CheckpointStorageError(
            "atomic checkpoint file write failed",
            stage=CheckpointStage.STAGING,
            path=target,
            retryable=exc.errno in {errno.EAGAIN, errno.EBUSY, errno.ENOSPC},
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_write_text(
    text: str,
    path: Path,
    *,
    durable: bool = True,
    mode: int = 0o600,
) -> None:
    if not isinstance(text, str):
        raise TypeError("atomic_write_text requires str")
    atomic_write_bytes(text.encode("utf-8"), path, durable=durable, mode=mode)


def read_limited_bytes(path: Path, *, max_bytes: int) -> bytes:
    """Read a regular file while enforcing an upper size bound."""

    candidate = Path(path)
    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero")
    try:
        mode = candidate.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise CheckpointPathError(
                "expected a regular file",
                stage=CheckpointStage.MANIFEST,
                path=candidate,
            )
        size = candidate.stat().st_size
        if size > max_bytes:
            raise CheckpointStorageError(
                f"file exceeds configured read limit ({size} > {max_bytes} bytes)",
                stage=CheckpointStage.MANIFEST,
                path=candidate,
                details={"size_bytes": size, "max_bytes": max_bytes},
            )
        with candidate.open("rb") as stream:
            data = stream.read(max_bytes + 1)
    except CheckpointStorageError:
        raise
    except OSError as exc:
        raise CheckpointStorageError(
            "failed to read checkpoint file",
            stage=CheckpointStage.MANIFEST,
            path=candidate,
            retryable=exc.errno in {errno.EAGAIN, errno.EBUSY},
        ) from exc
    if len(data) > max_bytes:
        raise CheckpointStorageError(
            "file grew beyond configured read limit while being read",
            stage=CheckpointStage.MANIFEST,
            path=candidate,
            details={"max_bytes": max_bytes},
        )
    return data


def sha256_file(path: Path, *, chunk_size: int = _COPY_CHUNK_SIZE) -> str:
    """Return the SHA-256 digest of a regular, non-symlink file."""

    candidate = Path(path)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    try:
        mode = candidate.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise CheckpointPathError(
                "only regular files can be hashed",
                stage=CheckpointStage.HASHING,
                path=candidate,
            )
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(chunk_size), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except CheckpointStorageError:
        raise
    except OSError as exc:
        raise CheckpointStorageError(
            "failed to hash checkpoint file",
            stage=CheckpointStage.HASHING,
            path=candidate,
            retryable=exc.errno in {errno.EAGAIN, errno.EBUSY},
        ) from exc


def iter_regular_files(root: Path) -> Iterator[Path]:
    """Yield files in deterministic order and reject special files/symlinks."""

    root = Path(root)
    for directory, dir_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        dir_names.sort()
        file_names.sort()
        for name in dir_names:
            path = directory_path / name
            if path.is_symlink():
                raise CheckpointPathError(
                    "symbolic-link directories are not permitted in checkpoints",
                    stage=CheckpointStage.INTEGRITY,
                    path=path,
                )
        for name in file_names:
            path = directory_path / name
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise CheckpointPathError(
                    "only regular files are permitted in checkpoints",
                    stage=CheckpointStage.INTEGRITY,
                    path=path,
                )
            yield path


def observe_checkpoint_files(
    checkpoint_dir: Path,
    *,
    exclude: Iterable[str] = (MANIFEST_NAME,),
) -> tuple[ObservedArtifact, ...]:
    """Observe size and digest of each file beneath a checkpoint directory."""

    try:
        root = Path(checkpoint_dir).resolve(strict=True)
    except OSError as exc:
        raise CheckpointStorageError(
            "checkpoint directory does not exist or cannot be resolved",
            operation=CheckpointOperation.VERIFY,
            stage=CheckpointStage.INTEGRITY,
            path=Path(checkpoint_dir),
            retryable=exc.errno in {errno.EAGAIN, errno.EBUSY},
        ) from exc
    if not root.is_dir():
        raise CheckpointPathError(
            "checkpoint path is not a directory",
            operation=CheckpointOperation.VERIFY,
            stage=CheckpointStage.INTEGRITY,
            path=root,
        )
    excluded = {validate_relative_path(name, allow_manifest=True) for name in exclude}
    observed: list[ObservedArtifact] = []
    for path in iter_regular_files(root):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        size = path.stat().st_size
        observed.append(
            ObservedArtifact(
                relative_path=relative,
                size_bytes=size,
                digest=ArtifactDigest("sha256", sha256_file(path)),
            )
        )
    return tuple(sorted(observed, key=lambda item: item.relative_path))


def hash_checkpoint_files(
    checkpoint_dir: Path,
    *,
    exclude: Iterable[str] = (MANIFEST_NAME,),
) -> dict[str, CheckpointFileInfo]:
    """Backward-compatible mapping view over ``observe_checkpoint_files``."""

    return {
        item.relative_path: CheckpointFileInfo(
            size_bytes=item.size_bytes,
            sha256=item.digest.value,
        )
        for item in observe_checkpoint_files(checkpoint_dir, exclude=exclude)
    }


def _pid_is_alive(pid: int) -> bool | None:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


class CheckpointFileLock(AbstractContextManager["CheckpointFileLock"]):
    """Cross-platform cooperative lock based on atomic file creation."""

    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 0.05,
        stale_after_seconds: float = 3600.0,
        durable: bool = True,
    ) -> None:
        if timeout_seconds < 0 or poll_interval_seconds <= 0 or stale_after_seconds <= 0:
            raise ValueError("invalid checkpoint lock timing configuration")
        self.path = Path(path)
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.stale_after_seconds = stale_after_seconds
        self.durable = durable
        self.token = uuid.uuid4().hex
        self._held = False

    def _metadata(self) -> dict[str, object]:
        return {
            "token": self.token,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_at": utc_now_iso(),
            "created_epoch": time.time(),
        }

    def _try_break_stale_lock(self) -> bool:
        try:
            stat_result = self.path.lstat()
            if not stat.S_ISREG(stat_result.st_mode):
                raise CheckpointLockError(
                    "checkpoint lock path is not a regular file",
                    operation=CheckpointOperation.LOCK,
                    stage=CheckpointStage.LOCKING,
                    path=self.path,
                )
            age = max(0.0, time.time() - stat_result.st_mtime)
            if age < self.stale_after_seconds:
                return False
            try:
                metadata = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                metadata = None

            if isinstance(metadata, Mapping):
                owner_host = metadata.get("hostname")
                owner_pid = metadata.get("pid")
                if owner_host != socket.gethostname():
                    # Never steal a lock owned by another host solely due to age.
                    return False
                if isinstance(owner_pid, int) and _pid_is_alive(owner_pid) is not False:
                    return False

            stale_path = self.path.with_name(f"{self.path.name}.stale-{uuid.uuid4().hex}")
            try:
                os.replace(self.path, stale_path)
            except FileNotFoundError:
                return True
            stale_path.unlink(missing_ok=True)
            if self.durable:
                fsync_directory(self.path.parent)
            return True
        except CheckpointLockError:
            raise
        except OSError:
            return False

    def acquire(self) -> "CheckpointFileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        payload = (json.dumps(self._metadata(), sort_keys=True) + "\n").encode("utf-8")

        while True:
            descriptor: int | None = None
            created = False
            try:
                descriptor = os.open(str(self.path), flags, 0o600)
                created = True
                with os.fdopen(descriptor, "wb") as stream:
                    descriptor = None
                    stream.write(payload)
                    stream.flush()
                    if self.durable:
                        os.fsync(stream.fileno())
                if self.durable:
                    fsync_directory(self.path.parent)
                self._held = True
                return self
            except FileExistsError:
                if self._try_break_stale_lock():
                    continue
                if time.monotonic() >= deadline:
                    raise CheckpointLockTimeoutError(
                        "timed out waiting for checkpoint storage lock",
                        operation=CheckpointOperation.LOCK,
                        stage=CheckpointStage.LOCKING,
                        path=self.path,
                        retryable=True,
                        details={"timeout_seconds": self.timeout_seconds},
                    )
                time.sleep(min(self.poll_interval_seconds, max(0.0, deadline - time.monotonic())))
            except CheckpointError:
                if created:
                    try:
                        self.path.unlink(missing_ok=True)
                    except OSError:
                        pass
                raise
            except OSError as exc:
                if created:
                    try:
                        self.path.unlink(missing_ok=True)
                    except OSError:
                        pass
                raise CheckpointLockError(
                    "failed to acquire checkpoint storage lock",
                    operation=CheckpointOperation.LOCK,
                    stage=CheckpointStage.LOCKING,
                    path=self.path,
                    retryable=exc.errno in {errno.EAGAIN, errno.EBUSY},
                ) from exc
            finally:
                if descriptor is not None:
                    os.close(descriptor)

    def release(self) -> None:
        if not self._held:
            return
        try:
            try:
                metadata = json.loads(self.path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                self._held = False
                return
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise CheckpointLockError(
                    "checkpoint lock metadata became unreadable before release",
                    operation=CheckpointOperation.LOCK,
                    stage=CheckpointStage.CLEANUP,
                    path=self.path,
                ) from exc
            if not isinstance(metadata, Mapping) or metadata.get("token") != self.token:
                raise CheckpointLockError(
                    "checkpoint lock ownership changed before release",
                    operation=CheckpointOperation.LOCK,
                    stage=CheckpointStage.LOCKING,
                    path=self.path,
                )
            self.path.unlink()
            if self.durable:
                fsync_directory(self.path.parent)
        except CheckpointLockError:
            raise
        except OSError as exc:
            raise CheckpointLockError(
                "failed to release checkpoint storage lock",
                operation=CheckpointOperation.LOCK,
                stage=CheckpointStage.CLEANUP,
                path=self.path,
                retryable=exc.errno in {errno.EAGAIN, errno.EBUSY},
            ) from exc
        finally:
            self._held = False

    def __enter__(self) -> "CheckpointFileLock":
        return self.acquire()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


def _remove_tree(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            path.unlink()
        else:
            shutil.rmtree(path)


class FileSystemCheckpointStorage:
    """Transactional checkpoint storage rooted at one local directory."""

    def __init__(
        self,
        base_dir: str | os.PathLike[str] | None = None,
        *,
        config: CheckpointConfig | None = None,
    ) -> None:
        if config is not None and base_dir is not None:
            raise ValueError("provide either base_dir or config, not both")
        self.config = config or CheckpointConfig(
            base_dir=Path(base_dir) if base_dir is not None else Path("src/checkpoints")
        )
        self._base_dir = ensure_directory(
            self.config.base_dir, mode=self.config.directory_mode
        )
        self._lock_dir = ensure_directory(
            self._base_dir / _LOCK_DIRECTORY, mode=self.config.directory_mode
        )

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def lock(self) -> CheckpointFileLock:
        return CheckpointFileLock(
            self._lock_dir / _GLOBAL_LOCK_NAME,
            timeout_seconds=self.config.lock_timeout_seconds,
            poll_interval_seconds=self.config.lock_poll_interval_seconds,
            stale_after_seconds=self.config.stale_lock_seconds,
            durable=self.config.durable_writes,
        )

    def checkpoint_path(self, version: str) -> Path:
        return resolve_checkpoint_path(self.base_dir, version)

    def begin(self, version: str, *, allow_overwrite: bool = False) -> StagingArea:
        safe_version = _domain_version(version)
        final_path = self.checkpoint_path(safe_version)
        transaction_id = uuid.uuid4().hex
        staging_path = self.base_dir / f"{_STAGING_PREFIX}{transaction_id}-{safe_version}"

        with self.lock():
            if final_path.exists() and not allow_overwrite:
                raise CheckpointConflictError(
                    "checkpoint version already exists",
                    operation=CheckpointOperation.SAVE,
                    stage=CheckpointStage.STAGING,
                    path=final_path,
                    version=safe_version,
                    committed=True,
                )
            try:
                staging_path.mkdir(mode=self.config.directory_mode, parents=False, exist_ok=False)
            except OSError as exc:
                raise CheckpointStorageError(
                    "failed to create checkpoint staging directory",
                    operation=CheckpointOperation.SAVE,
                    stage=CheckpointStage.STAGING,
                    path=staging_path,
                    version=safe_version,
                    retryable=exc.errno in {errno.EAGAIN, errno.EBUSY, errno.ENOSPC},
                    committed=False,
                ) from exc

        return StagingArea(
            version=safe_version,
            transaction_id=transaction_id,
            path=staging_path,
            final_path=final_path,
        )

    def _validate_staging(self, staging: StagingArea) -> None:
        expected_final = self.checkpoint_path(staging.version)
        if staging.final_path.resolve(strict=False) != expected_final.resolve(strict=False):
            raise CheckpointPathError(
                "staging final path does not match its version",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.VALIDATION,
                path=staging.final_path,
                version=staging.version,
                committed=False,
            )
        expected_prefix = f"{_STAGING_PREFIX}{staging.transaction_id}-"
        if (
            staging.path.parent.resolve(strict=False) != self.base_dir
            or not staging.path.name.startswith(expected_prefix)
            or not staging.path.is_dir()
            or staging.path.is_symlink()
        ):
            raise CheckpointPathError(
                "invalid or missing checkpoint staging directory",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.VALIDATION,
                path=staging.path,
                version=staging.version,
                committed=False,
            )

    def commit(self, staging: StagingArea, *, allow_overwrite: bool = False) -> Path:
        """Publish a complete staged checkpoint.

        The manifest must already exist and is expected to have been written
        after all payload artifacts.  Overwrite is recoverable but deliberately
        opt-in; immutable checkpoint versions remain the preferred mode.
        """

        self._validate_staging(staging)
        manifest_path = staging.path / MANIFEST_NAME
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise CheckpointCommitError(
                "staged checkpoint has no regular manifest file",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.COMMIT,
                path=manifest_path,
                version=staging.version,
                committed=False,
            )

        if self.config.durable_writes:
            try:
                fsync_tree(staging.path)
            except CheckpointStorageError as exc:
                raise CheckpointCommitError(
                    "failed to make staged checkpoint durable before publication",
                    operation=CheckpointOperation.SAVE,
                    stage=CheckpointStage.COMMIT,
                    path=staging.path,
                    version=staging.version,
                    retryable=exc.retryable,
                    committed=False,
                    details=exc.details,
                ) from exc

        backup: Path | None = None
        published = False
        with self.lock():
            try:
                if staging.final_path.exists():
                    if not allow_overwrite:
                        raise CheckpointConflictError(
                            "checkpoint version was committed by another writer",
                            operation=CheckpointOperation.SAVE,
                            stage=CheckpointStage.COMMIT,
                            path=staging.final_path,
                            version=staging.version,
                            committed=True,
                        )
                    backup = self.base_dir / (
                        f"{_BACKUP_PREFIX}{staging.transaction_id}-{staging.version}"
                    )
                    staging.final_path.rename(backup)

                staging.path.rename(staging.final_path)
                published = True
                if self.config.durable_writes:
                    fsync_directory(self.base_dir)
            except CheckpointConflictError:
                raise
            except Exception as exc:
                if not published and backup is not None and backup.exists() and not staging.final_path.exists():
                    try:
                        backup.rename(staging.final_path)
                        if self.config.durable_writes:
                            fsync_directory(self.base_dir)
                    except OSError:
                        pass
                raise CheckpointCommitError(
                    "failed to publish staged checkpoint",
                    operation=CheckpointOperation.SAVE,
                    stage=CheckpointStage.COMMIT,
                    path=staging.final_path,
                    version=staging.version,
                    retryable=(
                        exc.retryable
                        if isinstance(exc, CheckpointError)
                        else isinstance(exc, OSError)
                        and exc.errno in {errno.EAGAIN, errno.EBUSY, errno.ENOSPC}
                    ),
                    committed=published,
                ) from exc

            if backup is not None and backup.exists():
                try:
                    _remove_tree(backup)
                    if self.config.durable_writes:
                        fsync_directory(self.base_dir)
                except (OSError, CheckpointStorageError):
                    # A hidden backup does not invalidate the newly committed
                    # checkpoint. recover_incomplete_transactions removes it.
                    pass
        return staging.final_path

    def abort(self, staging: StagingArea) -> None:
        """Remove an uncommitted staging directory only."""

        expected_prefix = f"{_STAGING_PREFIX}{staging.transaction_id}-"
        if staging.path.parent.resolve(strict=False) != self.base_dir or not staging.path.name.startswith(
            expected_prefix
        ):
            raise CheckpointPathError(
                "refusing to remove a path that is not this transaction's staging directory",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.CLEANUP,
                path=staging.path,
                version=staging.version,
                committed=False,
            )
        try:
            _remove_tree(staging.path)
            if self.config.durable_writes:
                fsync_directory(self.base_dir)
        except OSError as exc:
            raise CheckpointStorageError(
                "failed to remove checkpoint staging directory",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.CLEANUP,
                path=staging.path,
                version=staging.version,
                retryable=True,
                committed=False,
            ) from exc

    def list_versions(self) -> tuple[str, ...]:
        versions: list[str] = []
        try:
            for path in self.base_dir.iterdir():
                if path.name.startswith(".") or not path.is_dir() or path.is_symlink():
                    continue
                try:
                    versions.append(validate_version(path.name))
                except ValueError:
                    continue
        except OSError as exc:
            raise CheckpointStorageError(
                "failed to enumerate checkpoint versions",
                operation=CheckpointOperation.LIST,
                stage=CheckpointStage.DISCOVERY,
                path=self.base_dir,
                retryable=exc.errno in {errno.EAGAIN, errno.EBUSY},
            ) from exc
        return tuple(sorted(versions))

    def delete(self, version: str, *, missing_ok: bool = False) -> bool:
        safe_version = _domain_version(version)
        target = self.checkpoint_path(safe_version)
        tombstone = self.base_dir / f"{_DELETE_PREFIX}{uuid.uuid4().hex}-{safe_version}"
        with self.lock():
            if not target.exists():
                if missing_ok:
                    return False
                raise CheckpointNotFoundError(
                    "checkpoint version does not exist",
                    operation=CheckpointOperation.DELETE,
                    stage=CheckpointStage.DISCOVERY,
                    path=target,
                    version=safe_version,
                )
            if target.is_symlink() or not target.is_dir():
                raise CheckpointPathError(
                    "refusing to delete a non-directory or symbolic-link checkpoint path",
                    operation=CheckpointOperation.DELETE,
                    stage=CheckpointStage.VALIDATION,
                    path=target,
                    version=safe_version,
                )
            try:
                target.rename(tombstone)
                if self.config.durable_writes:
                    fsync_directory(self.base_dir)
                _remove_tree(tombstone)
                if self.config.durable_writes:
                    fsync_directory(self.base_dir)
            except OSError as exc:
                raise CheckpointStorageError(
                    "failed to delete checkpoint",
                    operation=CheckpointOperation.DELETE,
                    stage=CheckpointStage.CLEANUP,
                    path=target,
                    version=safe_version,
                    retryable=exc.errno in {errno.EAGAIN, errno.EBUSY},
                ) from exc
        return True

    def archive_path(self, version: str) -> Path:
        safe_version = _domain_version(version)
        return self.base_dir / f"{safe_version}.tar.gz"

    def create_archive(self, version: str, *, overwrite: bool = False) -> Path:
        """Create an atomic tar.gz archive containing one immutable checkpoint."""

        safe_version = _domain_version(version)
        checkpoint_dir = self.checkpoint_path(safe_version)
        archive = self.archive_path(safe_version)
        sidecar = archive.with_name(f"{archive.name}.sha256")
        temporary = self.base_dir / f".{archive.name}.{uuid.uuid4().hex}.tmp"

        if not checkpoint_dir.is_dir() or checkpoint_dir.is_symlink():
            raise CheckpointNotFoundError(
                "checkpoint cannot be archived because it does not exist",
                operation=CheckpointOperation.ARCHIVE,
                stage=CheckpointStage.DISCOVERY,
                path=checkpoint_dir,
                version=safe_version,
            )
        if not (checkpoint_dir / MANIFEST_NAME).is_file():
            raise CheckpointArchiveError(
                "checkpoint cannot be archived without a manifest",
                operation=CheckpointOperation.ARCHIVE,
                stage=CheckpointStage.VALIDATION,
                path=checkpoint_dir,
                version=safe_version,
                details={"archive_format": "tar.gz"},
            )

        with self.lock():
            if archive.exists() and not overwrite:
                raise CheckpointConflictError(
                    "checkpoint archive already exists",
                    operation=CheckpointOperation.ARCHIVE,
                    stage=CheckpointStage.ARCHIVAL,
                    path=archive,
                    version=safe_version,
                )
            try:
                with tarfile.open(temporary, mode="w:gz", format=tarfile.PAX_FORMAT) as tar:
                    root_info = tar.gettarinfo(str(checkpoint_dir), arcname=safe_version)
                    tar.addfile(root_info)
                    for path in iter_regular_files(checkpoint_dir):
                        relative = path.relative_to(checkpoint_dir).as_posix()
                        info = tar.gettarinfo(
                            str(path), arcname=f"{safe_version}/{relative}"
                        )
                        with path.open("rb") as stream:
                            tar.addfile(info, stream)
                if self.config.durable_writes:
                    _fsync_file(temporary)
                os.replace(temporary, archive)
                if self.config.durable_writes:
                    fsync_directory(self.base_dir)
                digest = sha256_file(archive)
                atomic_write_text(
                    digest + "\n",
                    sidecar,
                    durable=self.config.durable_writes,
                    mode=self.config.file_mode,
                )
                return archive
            except (CheckpointStorageError, CheckpointConflictError):
                raise
            except (OSError, tarfile.TarError) as exc:
                raise CheckpointArchiveError(
                    "failed to create checkpoint archive",
                    operation=CheckpointOperation.ARCHIVE,
                    stage=CheckpointStage.ARCHIVAL,
                    path=archive,
                    version=safe_version,
                    retryable=isinstance(exc, OSError)
                    and exc.errno in {errno.EAGAIN, errno.EBUSY, errno.ENOSPC},
                    details={"archive_format": "tar.gz"},
                ) from exc
            finally:
                temporary.unlink(missing_ok=True)

    def _verify_archive_sidecar(self, archive: Path, *, required: bool) -> None:
        sidecar = archive.with_name(f"{archive.name}.sha256")
        if not sidecar.exists():
            if required:
                raise CheckpointArchiveError(
                    "checkpoint archive digest sidecar is missing",
                    operation=CheckpointOperation.RESTORE,
                    stage=CheckpointStage.INTEGRITY,
                    path=sidecar,
                )
            return
        try:
            expected = sidecar.read_text(encoding="ascii").strip().split()[0].lower()
        except (OSError, UnicodeError, IndexError) as exc:
            raise CheckpointArchiveError(
                "checkpoint archive digest sidecar is malformed",
                operation=CheckpointOperation.RESTORE,
                stage=CheckpointStage.INTEGRITY,
                path=sidecar,
            ) from exc
        actual = sha256_file(archive)
        if expected != actual:
            raise CheckpointArchiveError(
                "checkpoint archive digest does not match its sidecar",
                operation=CheckpointOperation.RESTORE,
                stage=CheckpointStage.INTEGRITY,
                path=archive,
                details={"expected_hash": expected, "actual_hash": actual},
            )

    def restore_archive(
        self,
        version: str,
        *,
        allow_overwrite: bool = False,
        require_digest: bool = True,
        max_members: int = 100_000,
        max_total_bytes: int | None = None,
    ) -> Path:
        """Safely restore an archive through staging, then atomically commit it."""

        safe_version = _domain_version(version)
        if max_members <= 0 or (max_total_bytes is not None and max_total_bytes < 0):
            raise ValueError("invalid archive extraction limits")
        archive = self.archive_path(safe_version)
        if not archive.is_file() or archive.is_symlink():
            raise CheckpointNotFoundError(
                "checkpoint archive does not exist",
                operation=CheckpointOperation.RESTORE,
                stage=CheckpointStage.DISCOVERY,
                path=archive,
                version=safe_version,
            )
        self._verify_archive_sidecar(archive, required=require_digest)
        staging = self.begin(safe_version, allow_overwrite=allow_overwrite)

        try:
            with tarfile.open(archive, mode="r:gz") as tar:
                total_size = 0
                member_count = 0
                for member in tar:
                    member_count += 1
                    if member_count > max_members:
                        raise CheckpointArchiveError(
                            "checkpoint archive exceeds the configured member limit",
                            operation=CheckpointOperation.RESTORE,
                            stage=CheckpointStage.RESTORATION,
                            path=archive,
                            version=safe_version,
                            details={
                                "members": member_count,
                                "max_members": max_members,
                            },
                        )
                    archive_path = PurePosixPath(member.name)
                    if archive_path.is_absolute() or any(
                        part in {"", ".", ".."} for part in archive_path.parts
                    ):
                        raise CheckpointArchiveError(
                            "checkpoint archive contains an unsafe member path",
                            operation=CheckpointOperation.RESTORE,
                            stage=CheckpointStage.RESTORATION,
                            path=archive,
                            version=safe_version,
                            details={"member": member.name},
                        )
                    if not archive_path.parts or archive_path.parts[0] != safe_version:
                        raise CheckpointArchiveError(
                            "checkpoint archive root does not match the requested version",
                            operation=CheckpointOperation.RESTORE,
                            stage=CheckpointStage.RESTORATION,
                            path=archive,
                            version=safe_version,
                            details={"member": member.name},
                        )
                    relative_parts = archive_path.parts[1:]
                    if not relative_parts:
                        if not member.isdir():
                            raise CheckpointArchiveError(
                                "checkpoint archive root must be a directory",
                                operation=CheckpointOperation.RESTORE,
                                stage=CheckpointStage.RESTORATION,
                                path=archive,
                                version=safe_version,
                            )
                        continue
                    relative = PurePosixPath(*relative_parts).as_posix()
                    destination = resolve_artifact_path(
                        staging.path,
                        relative,
                        allow_manifest=(relative == MANIFEST_NAME),
                    )

                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True, mode=self.config.directory_mode)
                        continue
                    if not member.isfile() or member.issym() or member.islnk():
                        raise CheckpointArchiveError(
                            "checkpoint archive contains a non-regular member",
                            operation=CheckpointOperation.RESTORE,
                            stage=CheckpointStage.RESTORATION,
                            path=archive,
                            version=safe_version,
                            details={"member": member.name, "type": member.type.decode(errors="replace")},
                        )
                    total_size += member.size
                    if max_total_bytes is not None and total_size > max_total_bytes:
                        raise CheckpointArchiveError(
                            "checkpoint archive exceeds the configured extraction size",
                            operation=CheckpointOperation.RESTORE,
                            stage=CheckpointStage.RESTORATION,
                            path=archive,
                            version=safe_version,
                            details={
                                "total_size": total_size,
                                "max_total_bytes": max_total_bytes,
                            },
                        )
                    source = tar.extractfile(member)
                    if source is None:
                        raise CheckpointArchiveError(
                            "failed to read checkpoint archive member",
                            operation=CheckpointOperation.RESTORE,
                            stage=CheckpointStage.RESTORATION,
                            path=archive,
                            version=safe_version,
                            details={"member": member.name},
                        )
                    destination.parent.mkdir(
                        parents=True, exist_ok=True, mode=self.config.directory_mode
                    )
                    self._copy_archive_member(source, destination, expected_size=member.size)

            if not (staging.path / MANIFEST_NAME).is_file():
                raise CheckpointArchiveError(
                    "restored archive does not contain a manifest",
                    operation=CheckpointOperation.RESTORE,
                    stage=CheckpointStage.VALIDATION,
                    path=archive,
                    version=safe_version,
                )
            return self.commit(staging, allow_overwrite=allow_overwrite)
        except Exception:
            try:
                self.abort(staging)
            except CheckpointStorageError:
                pass
            raise

    def _copy_archive_member(
        self,
        source: IO[bytes],
        destination: Path,
        *,
        expected_size: int,
    ) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        written = 0
        try:
            descriptor = os.open(str(destination), flags, self.config.file_mode)
            with os.fdopen(descriptor, "wb") as target:
                descriptor = None
                while True:
                    chunk = source.read(_COPY_CHUNK_SIZE)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > expected_size:
                        raise CheckpointArchiveError(
                            "archive member expanded beyond its declared size",
                            operation=CheckpointOperation.RESTORE,
                            stage=CheckpointStage.RESTORATION,
                            path=destination,
                        )
                    target.write(chunk)
                target.flush()
                if self.config.durable_writes:
                    os.fsync(target.fileno())
            if written != expected_size:
                raise CheckpointArchiveError(
                    "archive member size differs from its declaration",
                    operation=CheckpointOperation.RESTORE,
                    stage=CheckpointStage.RESTORATION,
                    path=destination,
                    details={"expected_size": expected_size, "actual_size": written},
                )
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def recover_incomplete_transactions(self) -> tuple[Path, ...]:
        """Recover or remove hidden overwrite backups left by process failure."""

        recovered: list[Path] = []
        with self.lock():
            for backup in sorted(self.base_dir.glob(f"{_BACKUP_PREFIX}*")):
                if not backup.is_dir() or backup.is_symlink():
                    continue
                suffix = backup.name[len(_BACKUP_PREFIX) :]
                transaction_id, separator, version = suffix.partition("-")
                if not separator or len(transaction_id) != 32:
                    continue
                try:
                    safe_version = validate_version(version)
                except ValueError:
                    continue
                final = self.checkpoint_path(safe_version)
                try:
                    if final.exists():
                        _remove_tree(backup)
                    else:
                        backup.rename(final)
                        recovered.append(final)
                    if self.config.durable_writes:
                        fsync_directory(self.base_dir)
                except OSError as exc:
                    raise CheckpointStorageError(
                        "failed to recover an interrupted checkpoint overwrite",
                        operation=CheckpointOperation.RESTORE,
                        stage=CheckpointStage.RESTORATION,
                        path=backup,
                        version=safe_version,
                        retryable=True,
                    ) from exc
        return tuple(recovered)


def prepare_staging_dir(
    base_dir: Path,
    version: str,
    *,
    allow_overwrite: bool,
) -> tuple[Path, Path]:
    """Compatibility wrapper returning ``(final_dir, staging_dir)``."""

    storage = FileSystemCheckpointStorage(base_dir)
    staging = storage.begin(version, allow_overwrite=allow_overwrite)
    return staging.final_path, staging.path


def commit_staging_dir(
    staging_dir: Path,
    final_dir: Path,
    *,
    allow_overwrite: bool,
) -> None:
    """Compatibility wrapper for callers not yet using ``StagingArea``."""

    staging_path = Path(staging_dir)
    final_path = Path(final_dir)
    if staging_path.parent.resolve(strict=False) != final_path.parent.resolve(strict=False):
        raise CheckpointPathError(
            "staging and final directories must share a parent filesystem",
            stage=CheckpointStage.COMMIT,
            path=staging_path,
        )
    prefix = staging_path.name
    if prefix.startswith(_STAGING_PREFIX):
        remainder = prefix[len(_STAGING_PREFIX) :]
        transaction_id, separator, embedded_version = remainder.partition("-")
    elif prefix.startswith(".tmp-"):
        transaction_id = uuid.uuid4().hex
        separator = "-"
        embedded_version = final_path.name
    else:
        transaction_id = uuid.uuid4().hex
        separator = "-"
        embedded_version = final_path.name
    if not separator:
        raise CheckpointPathError("cannot identify checkpoint staging transaction", path=staging_path)

    storage = FileSystemCheckpointStorage(final_path.parent)
    # Rename legacy staging paths to the canonical transaction name before
    # applying the same validation and commit logic.
    canonical_staging = storage.base_dir / f"{_STAGING_PREFIX}{transaction_id}-{embedded_version}"
    if staging_path.resolve(strict=False) != canonical_staging.resolve(strict=False):
        staging_path.rename(canonical_staging)
    area = StagingArea(
        version=final_path.name,
        transaction_id=transaction_id,
        path=canonical_staging,
        final_path=storage.checkpoint_path(final_path.name),
    )
    storage.commit(area, allow_overwrite=allow_overwrite)


def archive_checkpoint_dir(
    base_dir: Path,
    version: str,
    *,
    overwrite: bool = True,
) -> Path:
    return FileSystemCheckpointStorage(base_dir).create_archive(
        version, overwrite=overwrite
    )


__all__ = [
    "CheckpointFileLock",
    "FileSystemCheckpointStorage",
    "archive_checkpoint_dir",
    "atomic_write_bytes",
    "atomic_write_text",
    "commit_staging_dir",
    "ensure_directory",
    "fsync_directory",
    "fsync_tree",
    "hash_checkpoint_files",
    "iter_regular_files",
    "observe_checkpoint_files",
    "prepare_staging_dir",
    "read_limited_bytes",
    "resolve_artifact_path",
    "resolve_checkpoint_path",
    "sha256_file",
]