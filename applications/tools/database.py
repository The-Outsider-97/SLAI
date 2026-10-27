"""A small SQLite wrapper for application data: WAL, foreign keys, explicit transactions, versioned migrations and backups.

* **One connection, one lock.** The HTTP server is threaded; every statement runs under a re-entrant lock, so store methods can call
  each other inside one transaction (``with db.transaction():`` is re-entrant: only the outermost block begins and commits).
* **Migrations** are numbered and applied in order, each in its own transaction, recorded in ``PRAGMA user_version``. A database
  that is *newer* than the code is refused rather than guessed at. Before the first migration of an existing database a coherent copy
  is written to the backups folder, so a bad upgrade can always be undone.
* **Backups** use SQLite's online backup API, which is coherent even while the database is in use (copying only the ``.sqlite3``
  file of a running WAL database is not).
"""
from __future__ import annotations

import contextlib
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence, Union

from .errors import Conflict, StorageError, classify_oserror, classify_sqlite

Step = Union[str, Callable[[sqlite3.Connection], None]]


@dataclass(frozen=True)
class Migration:
    version: int
    description: str
    steps: tuple[Step, ...]


class Database:
    def __init__(self, path: Path | str, migrations: Sequence[Migration] = (), *, backups_dir: Path | str | None = None,
                 busy_timeout_ms: int = 5000) -> None:
        self.path = Path(path)
        self.backups_dir = Path(backups_dir) if backups_dir else None
        self.lock = threading.RLock()
        self._depth = 0
        self.migrated: list[int] = []
        self.upgraded = False                    # True when existing data (not a brand-new file) was migrated
        self.migration_backup: Path | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None, timeout=busy_timeout_ms / 1000)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
            self.conn.execute("PRAGMA journal_mode=WAL")
            self._migrate(sorted(migrations, key=lambda m: m.version))
        except sqlite3.Error as exc:
            self._discard()
            raise classify_sqlite(exc, "open the database") from exc
        except OSError as exc:
            self._discard()
            raise classify_oserror(exc, "open the database") from exc
        except Exception:
            self._discard()
            raise

    def _discard(self) -> None:
        conn = getattr(self, "conn", None)
        if conn is not None:
            with contextlib.suppress(sqlite3.Error):
                conn.close()

    # --------------------------------------------------------------------------------------------------- migrations
    @property
    def schema_version(self) -> int:
        return int(self.conn.execute("PRAGMA user_version").fetchone()[0])

    def _has_tables(self) -> bool:
        return self.conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchone()[0] > 0

    def _migrate(self, migrations: Sequence[Migration]) -> None:
        current = self.schema_version
        latest = migrations[-1].version if migrations else current
        if current > latest:
            raise Conflict(f"This data was written by a newer version of the application (database schema {current}, this version "
                           f"understands {latest}). Update the application; nothing was changed.")
        pending = [m for m in migrations if m.version > current]
        if not pending:
            return
        self.upgraded = self._has_tables()
        if self.upgraded and self.backups_dir is not None:
            self.migration_backup = self.backup(f"pre-migration-v{current}")
        for migration in pending:
            with self.transaction():
                for step in migration.steps:
                    if callable(step):
                        step(self.conn)
                    else:
                        self.conn.execute(step)
                self.conn.execute(f"PRAGMA user_version = {int(migration.version)}")
            self.migrated.append(migration.version)

    # --------------------------------------------------------------------------------------------------- statements
    @contextlib.contextmanager
    def transaction(self) -> Iterator["Database"]:
        """``BEGIN IMMEDIATE`` ... ``COMMIT`` (or ``ROLLBACK`` on any exception). Re-entrant."""
        with self.lock:
            outermost = self._depth == 0
            if outermost:
                try:
                    self.conn.execute("BEGIN IMMEDIATE")
                except sqlite3.Error as exc:
                    raise classify_sqlite(exc, "save to the database") from exc
            self._depth += 1
            try:
                yield self
            except BaseException:
                self._depth -= 1
                if outermost:
                    with contextlib.suppress(sqlite3.Error):
                        self.conn.execute("ROLLBACK")
                raise                                   # errors from the body are the caller's to classify
            self._depth -= 1
            if outermost:
                try:
                    self.conn.execute("COMMIT")
                except sqlite3.Error as exc:
                    with contextlib.suppress(sqlite3.Error):
                        self.conn.execute("ROLLBACK")
                    raise classify_sqlite(exc, "save to the database") from exc

    def execute(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> sqlite3.Cursor:
        with self.lock:
            return self.conn.execute(sql, params)

    def rows(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> list[dict[str, Any]]:
        with self.lock:
            return [dict(row) for row in self.conn.execute(sql, params).fetchall()]

    def row(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> dict[str, Any] | None:
        with self.lock:
            found = self.conn.execute(sql, params).fetchone()
            return dict(found) if found is not None else None

    def scalar(self, sql: str, params: Sequence[Any] | dict[str, Any] = ()) -> Any:
        with self.lock:
            found = self.conn.execute(sql, params).fetchone()
            return found[0] if found is not None else None

    # --------------------------------------------------------------------------------------------------- upkeep
    def backup(self, label: str = "backup", directory: Path | None = None) -> Path:
        """A coherent copy of the database (SQLite online backup) named ``<label>-<UTC time>.sqlite3``."""
        target_dir = directory or self.backups_dir
        if target_dir is None:
            raise StorageError("No backup folder is configured", reason="invalid_path")
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        target = target_dir / f"{label}-{stamp}.sqlite3"
        counter = 1
        while target.exists():
            counter += 1
            target = target_dir / f"{label}-{stamp}-{counter}.sqlite3"
        try:
            with self.lock:
                copy = sqlite3.connect(str(target))
                try:
                    self.conn.backup(copy)
                finally:
                    copy.close()
        except sqlite3.Error as exc:
            raise classify_sqlite(exc, "back up the database") from exc
        except OSError as exc:
            raise classify_oserror(exc, "back up the database") from exc
        return target

    def integrity(self) -> str:
        """``"ok"`` or the first problem SQLite reports."""
        with self.lock:
            return str(self.conn.execute("PRAGMA integrity_check").fetchone()[0])

    def close(self) -> None:
        with self.lock:
            with contextlib.suppress(sqlite3.Error):
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            with contextlib.suppress(sqlite3.Error):
                self.conn.close()
