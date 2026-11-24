"""Shared SQLite opener for governance state (durable, WAL, fail-safe defaults)."""

from __future__ import annotations

import sqlite3

from pathlib import Path
from typing import Union

__all__ = ["open_db"]


def open_db(path: Union[str, Path]) -> sqlite3.Connection:
    """Open a governance database in autocommit mode; callers use explicit BEGIN IMMEDIATE."""
    in_memory = str(path) == ":memory:"
    if not in_memory:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None, timeout=30.0)
    if not in_memory:
        db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA foreign_keys=ON")
    return db
