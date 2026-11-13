"""Where an application keeps its files, and how it writes them.

* **Folders** - every SLAI application keeps its data in ``%LOCALAPPDATA%\\SLAI\\<Title>`` (Windows) or
  ``$XDG_DATA_HOME/slai/<slug>`` (everywhere else), and its exports in ``Documents/SLAI <Title>``. The hand-off folder shared by the
  applications is ``<SLAI data>/handoff``, next to the application folders.
* **Atomic writes** - write a temporary file next to the target, flush it to disk, then rename it over the target in one step, so a
  crash or a full disk never leaves half a file behind. On some Windows machines that rename fails even inside one folder
  (``WinError 17`` in a redirected AppData folder, ``WinError 5``/``32`` while an anti-virus scanner holds the new file): those are
  retried a few times and, as a last resort, the data is written straight into the target so the person's work is saved either way.
* **Preferences** - a small JSON file with a declared schema (:class:`Prefs`). Unknown keys and invalid values are refused on write
  and dropped on read, so a hand-edited or damaged file can never put the interface into a state it cannot render.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import BadRequest, StorageError, classify_oserror

# --------------------------------------------------------------------------------------------------------------- folders


def default_data_dir(slug: str, title: str) -> Path:
    """``%LOCALAPPDATA%\\SLAI\\<title>`` on Windows, ``$XDG_DATA_HOME/slai/<slug>`` elsewhere."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "SLAI" / title
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "slai" / slug


def default_output_dir(title: str) -> Path:
    docs = Path.home() / "Documents"
    return (docs if docs.exists() else Path.home()) / f"SLAI {title}"


def handoff_dir(data_dir: Path) -> Path:
    """Shared by the SLAI applications: ``<SLAI data>/handoff``."""
    return data_dir.parent / "handoff"


# ---------------------------------------------------------------------------------------------------------- atomic writes
_RETRIES = 5


def atomic_write(path: Path | str, data: bytes, *, exclusive: bool = False) -> None:
    """Replace ``path`` with ``data`` without ever exposing a partial file. Raises :class:`StorageError` when nothing works.

    With ``exclusive=True`` the file must not exist yet (``FileExistsError`` otherwise); the name is claimed first, so two writers
    can never both create it.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".tmp")
    except OSError as exc:
        raise classify_oserror(exc, "create a temporary file") from exc
    claimed = False
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            try:
                os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))      # FileExistsError propagates
            except FileExistsError:
                raise
            except OSError as exc:
                raise classify_oserror(exc, "save a file") from exc
            claimed = True
        last: OSError | None = None
        for attempt in range(_RETRIES):
            try:
                os.replace(tmp, path)
                claimed = False
                return
            except OSError as exc:
                last = exc
                time.sleep(0.05 * (attempt + 1))
        # The rename will not work here (WinError 17 in a redirected AppData folder, a scanner holding the file...).
        # Write in place instead: not atomic, but the person's work is saved.
        try:
            with open(path, "wb") as out:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
            claimed = False
        except OSError as exc:
            raise classify_oserror(last or exc, "save a file") from exc
    except OSError as exc:
        if isinstance(exc, (StorageError, FileExistsError)):
            raise
        raise classify_oserror(exc, "save a file") from exc
    finally:
        if claimed:                                                 # we created an empty file and could not fill it
            try:
                os.unlink(path)
            except OSError:
                pass
        try:
            os.unlink(tmp)
        except OSError:
            pass


def atomic_write_text(path: Path | str, text: str) -> None:
    atomic_write(path, text.encode("utf-8"))


def atomic_write_json(path: Path | str, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


# -------------------------------------------------------------------------------------------------------------- preferences


@dataclass(frozen=True)
class Pref:
    """One preference: its type and what values it may take."""

    kind: type                                   # bool, int, float or str
    allowed: tuple[Any, ...] | None = None       # enumerated values (str or int)
    low: float | None = None                     # numeric range, inclusive
    high: float | None = None
    max_len: int = 64                            # strings without ``allowed``
    pattern: str | None = None                   # strings without ``allowed`` must match this regular expression

    def check(self, value: Any) -> Any:
        """Return the cleaned value or raise ``ValueError``."""
        if self.kind is bool:
            if type(value) is not bool:
                raise ValueError("must be true or false")
            return value
        if self.kind in (int, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
                raise ValueError("must be a number")
            if self.kind is int:
                if int(value) != value:
                    raise ValueError("must be a whole number")
                value = int(value)
            else:
                value = float(value)
            if self.allowed is not None and value not in self.allowed:
                raise ValueError(f"must be one of {', '.join(str(a) for a in self.allowed)}")
            if (self.low is not None and value < self.low) or (self.high is not None and value > self.high):
                raise ValueError(f"must be between {self.low} and {self.high}")
            return value
        if not isinstance(value, str):
            raise ValueError("must be text")
        if self.allowed is not None:
            if value not in self.allowed:
                raise ValueError(f"must be one of {', '.join(str(a) for a in self.allowed)}")
            return value
        if len(value) > self.max_len or any(ord(c) < 32 for c in value):
            raise ValueError(f"must be at most {self.max_len} characters")
        if self.pattern is not None and not re.fullmatch(self.pattern, value):
            raise ValueError("has an unsupported format")
        return value


class Prefs:
    """A schema-checked JSON preferences file. Thread-safe; written atomically after every change."""

    def __init__(self, path: Path | str, schema: Mapping[str, Pref]) -> None:
        self.path = Path(path)
        self.schema = dict(schema)
        self.data: dict[str, Any] = {}
        self.recovered: str | None = None
        self._lock = threading.RLock()
        if self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError("not an object")
                self.data = self.clean(raw, strict=False)
            except (OSError, ValueError):
                self.recovered = "The preferences file could not be read and was reset."

    def clean(self, raw: Mapping[str, Any], strict: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in raw.items():
            spec = self.schema.get(key)
            if spec is None:
                if strict:
                    raise BadRequest(f"Unknown preference '{key}'")
                continue
            try:
                out[key] = spec.check(value)
            except ValueError as exc:
                if strict:
                    raise BadRequest(f"Preference '{key}' {exc}") from exc
        return out

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.data)

    def update(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            cleaned = self.clean(raw)
            merged = {**self.data, **cleaned}
            atomic_write_json(self.path, merged)          # only replace the in-memory copy once the file is safely written
            self.data = merged
            return dict(self.data)
