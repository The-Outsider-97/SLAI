"""Evaluation issue persistence with explicit resource lifecycle and safe fallback."""
from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Dict

from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Database")
printer = PrettyPrinter()


class IssueDBConnector:
    """Connection-scoped PostgreSQL issue tracker; serializes cursor operations."""

    def __init__(
        self, host: str = "localhost", port: int = 5432,
        database: str = "safety_issues", user: str | None = None,
        password: str | None = None, **kwargs: Any,
    ) -> None:
        # Optional dependency is loaded only when a database is actually requested.
        import psycopg2  # type: ignore

        self._lock = RLock()
        self._closed = False
        params: Dict[str, Any] = {"host": host, "port": port, "database": database}
        if user is not None:
            params["user"] = user
        if password is not None:
            params["password"] = password
        for key in ("connect_timeout", "sslmode"):
            if key in kwargs:
                params[key] = kwargs[key]
        self.conn = psycopg2.connect(**params)
        self.cursor = self.conn.cursor()
        logger.info("Evaluation database connection established")

    def initialize_schema(self, tables: Dict[str, str]) -> None:
        """Apply trusted internal DDL only; never accept model-generated SQL."""
        if not isinstance(tables, dict):
            raise TypeError("tables must be a mapping of trusted schema DDL")
        with self._lock:
            if self._closed:
                raise RuntimeError("Database connector already closed")
            try:
                for _, ddl in tables.items():
                    if not isinstance(ddl, str) or not ddl.strip():
                        raise ValueError("schema DDL must be non-empty text")
                    self.cursor.execute(ddl)
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                logger.exception("Evaluation database schema initialization failed")
                raise

    def log_issue(self, issue_data: Dict[str, Any]) -> bool:
        """Write a structured issue with parameterized values."""
        query = """
            INSERT INTO evaluation_issues
            (id, timestamp, issue_type, severity, context, metrics)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        if not isinstance(issue_data, dict):
            raise TypeError("issue_data must be a dictionary")
        with self._lock:
            if self._closed:
                return False
            try:
                self.cursor.execute(query, (
                    issue_data["id"],
                    issue_data["timestamp"],
                    issue_data["type"],
                    issue_data["severity"],
                    json.dumps(issue_data.get("context", {}), allow_nan=False),
                    json.dumps(issue_data.get("metrics", {}), allow_nan=False),
                ))
                self.conn.commit()
                return True
            except Exception:
                self.conn.rollback()
                logger.exception("Evaluation issue write failed")
                return False

    def close(self) -> None:
        """Idempotent shutdown; safe under repeated EvaluationAgent.shutdown calls."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self.cursor.close()
            finally:
                self.conn.close()


class FallbackIssueTracker:
    """Bounded, thread-safe in-memory issue tracker for database outages."""

    def __init__(self, max_entries: int = 5000) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self.max_entries = max_entries
        self.issues: list[Dict[str, Any]] = []
        self._lock = RLock()
        logger.warning("Using fallback evaluation issue tracking (non-persistent)")

    def log_issue(self, issue_data: Dict[str, Any]) -> bool:
        if not isinstance(issue_data, dict):
            raise TypeError("issue_data must be a dictionary")
        copy = deepcopy(issue_data)
        copy.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with self._lock:
            self.issues.append(copy)
            if len(self.issues) > self.max_entries:
                del self.issues[:len(self.issues) - self.max_entries]
        return True

    def get_issues(self) -> list[Dict[str, Any]]:
        with self._lock:
            return deepcopy(self.issues)

    def close(self) -> None:
        """No persistent connection to release."""


__all__ = ["IssueDBConnector", "FallbackIssueTracker"]
