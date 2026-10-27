"""Crash-safe publication of generated files.

Every writer in this package produces its bytes first (in memory or in a
private temporary file beside the destination), flushes them to stable
storage, and only then makes them visible under the final name:

* ``overwrite=True``  -> ``os.replace`` (atomic on POSIX and Windows);
* ``overwrite=False`` -> exclusive creation via a hard link (POSIX) or an
  atomic no-clobber rename (Windows), so a concurrent writer can never be
  silently clobbered and a reader never observes a partial file.

If anything fails, the destination is untouched and the temporary file is
removed.
"""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

from .errors import ConflictError, ValidationError

__all__ = ["atomic_output", "write_bytes_atomic", "file_sha256"]


def file_sha256(path, *, chunk: int = 1024 * 1024) -> str:
    """SHA-256 hex digest of a file, streamed."""
    hasher = sha256()
    with open(path, "rb") as stream:
        while block := stream.read(chunk):
            hasher.update(block)
    return hasher.hexdigest()


def _fsync_file(name) -> None:
    # Windows' _commit() needs a writable descriptor, so reopen read/write.
    with open(name, "r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(directory: Path) -> None:
    """Best effort: persist the rename itself (not meaningful on Windows)."""
    if os.name == "nt":
        return
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _publish(temporary: str, destination: Path, overwrite: bool) -> None:
    if overwrite:
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        return
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise ConflictError("output file already exists", path=str(destination)) from exc
    except OSError as exc:
        if os.name == "nt":
            # Some Windows filesystems have no hard links, but os.rename()
            # there refuses to replace an existing file: still no-clobber.
            try:
                os.rename(temporary, destination)
                return
            except FileExistsError as inner:
                raise ConflictError("output file already exists", path=str(destination)) from inner
            except OSError as inner:
                raise ConflictError("filesystem does not support atomic no-clobber publication",
                                    path=str(destination)) from inner
        # A streamed copy would expose partial data: fail without publishing.
        raise ConflictError("filesystem does not support atomic no-clobber publication",
                            path=str(destination)) from exc
    _fsync_directory(destination.parent)


@contextmanager
def atomic_output(path, *, overwrite: bool = True, suffix: str = ".tmp"):
    """Yield a private temporary path; publish it as ``path`` on clean exit.

    Use for backends that insist on writing to a filename (``ezdxf.saveas``,
    external converters). The temporary file lives in the destination
    directory so publication never crosses filesystems.
    """
    destination = Path(path)
    if destination.is_dir():
        raise ValidationError("output path is a directory", path=str(destination))
    if destination.exists() and not overwrite:
        raise ConflictError("output file already exists", path=str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=".slai-spatial-", suffix=suffix, dir=destination.parent)
    os.close(handle)
    try:
        yield Path(name)
        _fsync_file(name)
        _publish(name, destination, overwrite)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_bytes_atomic(path, data: bytes, *, overwrite: bool = False) -> str:
    """Atomically publish ``data``; return its SHA-256 hex digest."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValidationError("data must be bytes")
    raw = bytes(data)
    with atomic_output(path, overwrite=overwrite) as temporary:
        temporary.write_bytes(raw)
    return sha256(raw).hexdigest()
