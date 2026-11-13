"""Typed errors shared by every SLAI application.

Every error carries an HTTP ``status``, a stable ``code`` for the client and a message a person can act on. The HTTP layer
(:mod:`applications.tools.server.http`) turns any :class:`AppError` into ``{"ok": false, "error": {...}}``; applications raise
these instead of defining their own base classes.

Storage failures are classified once, here, by :func:`classify_oserror` and :func:`classify_sqlite`, so every application reports
"the disk is full" or "another program holds the file" the same way instead of guessing at one cause.
"""
from __future__ import annotations

import errno
import os
import sqlite3
from typing import Any


class AppError(Exception):
    """Base class. ``status`` is the HTTP status, ``code`` a stable machine-readable name."""

    status = 500
    code = "error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.hint = hint

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.hint:
            out["hint"] = self.hint
        if self.details:
            out["details"] = self.details
        return out


class BadRequest(AppError):
    """The request itself is wrong: missing, malformed or out-of-range input."""

    status, code = 400, "bad_request"


class Unauthorized(AppError):
    status, code = 401, "unauthorized"


class Forbidden(AppError):
    status, code = 403, "forbidden"


class NotFound(AppError):
    status, code = 404, "not_found"


class MethodNotAllowed(AppError):
    status, code = 405, "method_not_allowed"


class Conflict(AppError):
    """The stored object changed since the client last saw it (version mismatch) or the operation clashes with its state."""

    status, code = 409, "conflict"


class LimitExceeded(AppError):
    status, code = 413, "limit_exceeded"


class DependencyMissing(AppError):
    """Something optional that the request needs (the SLAI checkout, an agent, a module) is not available."""

    status, code = 503, "dependency_missing"


# --------------------------------------------------------------------------------------------------------------- storage
STORAGE_REASONS: dict[str, tuple[int, str]] = {
    # reason: (HTTP status, what a person can do about it)
    "disk_full": (507, "There is no space left on that drive. Free some space and try again."),
    "permission_denied": (403, "The application is not allowed to write there. Check that the folder and the file are not read-only "
                               "and that your account may change them."),
    "locked": (423, "Another program has the file open (a backup, sync or anti-virus tool, for example). Wait a moment and try again."),
    "cross_volume": (500, "The data folder is on a redirected, mounted or virtualised location that the system will not rename files within."),
    "invalid_path": (500, "The data folder path is not usable (missing, too long, or containing characters the system does not allow)."),
    "corrupt": (500, "The database file is damaged. A copy was kept; restore one of the backups in the data folder."),
    "unexpected": (500, "The application could not write its files. The technical details are in its console window."),
}


class StorageError(AppError, OSError):
    """A write to the application's own storage failed. ``reason`` is one of :data:`STORAGE_REASONS`.

    ``details`` carries the failing operation, errno / winerror and the exception type - never file contents or tokens. It is also an
    :class:`OSError` so older ``except OSError`` guards around best-effort writes keep working.
    """

    status, code = 500, "storage_error"

    def __init__(self, message: str, *, reason: str = "unexpected", details: dict[str, Any] | None = None) -> None:
        AppError.__init__(self, message, details=details, hint=STORAGE_REASONS.get(reason, STORAGE_REASONS["unexpected"])[1])
        self.reason = reason
        self.code = f"storage_{reason}"
        self.status = STORAGE_REASONS.get(reason, STORAGE_REASONS["unexpected"])[0]


_WIN_LOCKED = {32, 33}
_WIN_DENIED = {5}
_WIN_FULL = {39, 112}
_WIN_BAD_PATH = {2, 3, 123, 161, 206}
_WIN_CROSS = {17}


def classify_oserror(exc: OSError, operation: str = "storage") -> StorageError:
    """Map an :class:`OSError` to a :class:`StorageError` whose reason names the real cause."""
    winerror = getattr(exc, "winerror", None)
    number = exc.errno
    if winerror in _WIN_LOCKED:
        reason = "locked"
    elif winerror in _WIN_DENIED or number in (errno.EACCES, errno.EPERM, errno.EROFS):
        reason = "permission_denied"
    elif winerror in _WIN_FULL or number in (errno.ENOSPC, getattr(errno, "EDQUOT", -1)):
        reason = "disk_full"
    elif winerror in _WIN_CROSS or number == errno.EXDEV:
        reason = "cross_volume"
    elif winerror in _WIN_BAD_PATH or number in (errno.ENOENT, errno.ENAMETOOLONG, errno.ENOTDIR, errno.EINVAL):
        reason = "invalid_path"
    elif number in (errno.EBUSY, errno.ETXTBSY):
        reason = "locked"
    else:
        reason = "unexpected"
    return StorageError(f"{operation} failed: {STORAGE_REASONS[reason][1]}", reason=reason,
                        details={"operation": operation, "errno": number, "winerror": winerror, "type": type(exc).__name__,
                                 "file": os.path.basename(str(exc.filename)) if exc.filename else None})


def classify_sqlite(exc: sqlite3.Error, operation: str = "database") -> StorageError:
    """Map an :class:`sqlite3.Error` to a :class:`StorageError` (locked, full, damaged or unexpected)."""
    text = str(exc).lower()
    if "locked" in text or "busy" in text:
        reason = "locked"
    elif "full" in text:
        reason = "disk_full"
    elif "readonly" in text or "read-only" in text:
        reason = "permission_denied"
    elif isinstance(exc, sqlite3.DatabaseError) and ("malformed" in text or "not a database" in text or "corrupt" in text):
        reason = "corrupt"
    elif "unable to open" in text:
        reason = "invalid_path"
    else:
        reason = "unexpected"
    return StorageError(f"{operation} failed: {STORAGE_REASONS[reason][1]}", reason=reason,
                        details={"operation": operation, "type": type(exc).__name__})
