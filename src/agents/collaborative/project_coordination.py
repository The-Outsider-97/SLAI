"""Durable, application-neutral participant sessions and *coordination* leases.

This is not an application revision store or mutation engine. Host adapters
remain authoritative for documents, model revisions, permissions, and commits.
Leases carry monotonically increasing fencing tokens; a host that requires
exclusive writes MUST validate the token at its own commit boundary. A lease
checked only in this store is advisory, not a distributed write guarantee.

The SQLite transaction serializes short metadata changes, never agent work.
Independent projects/resources are not locked during long computations.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from src.agents.planning.workflow_runtime import AuthorizationContext


class ProjectCoordinationError(ValueError):
    """Invalid or conflicting project collaboration operation."""


def _id(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectCoordinationError(f"{label} must be a nonempty string")
    return value.strip()


class ProjectCoordinationStore:
    """SQLite-backed project presence, lease fencing, and replayable metadata.

    A trusted host constructs ``AuthorizationContext`` from verified identity
    and policy. The store checks scopes on every operation but cannot authenticate
    a caller that is allowed to construct arbitrary AuthorizationContext objects.
    """

    def __init__(self, path: str | Path, *, clock: Callable[[], float] = time.time) -> None:
        self._lock = threading.RLock()
        self._clock = clock
        selected = ":memory:" if str(path) == ":memory:" else str(Path(path).expanduser().resolve())
        if selected != ":memory:":
            Path(selected).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(selected, timeout=15, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA busy_timeout=15000")
        with self._db:
            self._db.execute("CREATE TABLE IF NOT EXISTS project_participants ("
                             "project_id TEXT NOT NULL, actor_id TEXT NOT NULL, kind TEXT NOT NULL, "
                             "active INTEGER NOT NULL, updated_at REAL NOT NULL, "
                             "PRIMARY KEY(project_id,actor_id))")
            self._db.execute("CREATE TABLE IF NOT EXISTS resource_fences ("
                             "project_id TEXT NOT NULL, resource_id TEXT NOT NULL, token INTEGER NOT NULL, "
                             "PRIMARY KEY(project_id,resource_id))")
            self._db.execute("CREATE TABLE IF NOT EXISTS resource_leases ("
                             "project_id TEXT NOT NULL, resource_id TEXT NOT NULL, actor_id TEXT NOT NULL, "
                             "lease_id TEXT NOT NULL, token INTEGER NOT NULL, expires_at REAL NOT NULL, "
                             "PRIMARY KEY(project_id,resource_id))")
            self._db.execute("CREATE TABLE IF NOT EXISTS project_coordination_events ("
                             "sequence INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL, "
                             "event_type TEXT NOT NULL, payload TEXT NOT NULL)")
            self._db.execute("CREATE INDEX IF NOT EXISTS ix_project_coordination_events "
                             "ON project_coordination_events(project_id,sequence)")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _event(self, project_id: str, name: str, detail: dict[str, Any]) -> None:
        self._db.execute("INSERT INTO project_coordination_events(project_id,event_type,payload) VALUES (?,?,?)",
                         (project_id, name, json.dumps({"time": self._clock(), **detail}, sort_keys=True)))

    def _begin(self) -> None:
        self._db.execute("BEGIN IMMEDIATE")

    def _joined(self, auth: AuthorizationContext) -> None:
        row = self._db.execute("SELECT active FROM project_participants WHERE project_id=? AND actor_id=?",
                               (auth.project_id, auth.actor_id)).fetchone()
        if row is None or not row["active"]:
            raise ProjectCoordinationError("participant must join this project first")

    def join(self, *, auth: AuthorizationContext, kind: str) -> dict[str, Any]:
        auth.require(auth.project_id, "collaboration:join")
        if kind not in {"human", "agent", "service"}:
            raise ProjectCoordinationError("kind must be human, agent, or service")
        with self._lock:
            self._begin()
            try:
                current = self._db.execute("SELECT kind,active FROM project_participants "
                                           "WHERE project_id=? AND actor_id=?",
                                           (auth.project_id, auth.actor_id)).fetchone()
                if current is not None and current["active"] and current["kind"] == kind:
                    self._db.rollback()  # Repeated joins are idempotent, including event publication.
                    return {"project_id": auth.project_id, "actor_id": auth.actor_id,
                            "kind": kind, "active": True}
                self._db.execute("INSERT INTO project_participants(project_id,actor_id,kind,active,updated_at) "
                                 "VALUES (?,?,?,?,?) ON CONFLICT(project_id,actor_id) DO UPDATE SET "
                                 "kind=excluded.kind,active=1,updated_at=excluded.updated_at",
                                 (auth.project_id, auth.actor_id, kind, 1, self._clock()))
                self._event(auth.project_id, "participant.joined", {"actor_id": auth.actor_id, "kind": kind})
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return {"project_id": auth.project_id, "actor_id": auth.actor_id, "kind": kind, "active": True}

    def leave(self, *, auth: AuthorizationContext) -> None:
        auth.require(auth.project_id, "collaboration:join")
        with self._lock:
            self._begin()
            try:
                self._joined(auth)
                self._db.execute("UPDATE project_participants SET active=0,updated_at=? "
                                 "WHERE project_id=? AND actor_id=?",
                                 (self._clock(), auth.project_id, auth.actor_id))
                self._db.execute("DELETE FROM resource_leases WHERE project_id=? AND actor_id=?",
                                 (auth.project_id, auth.actor_id))
                self._event(auth.project_id, "participant.left", {"actor_id": auth.actor_id})
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

    def participants(self, *, auth: AuthorizationContext) -> list[dict[str, Any]]:
        auth.require(auth.project_id, "collaboration:read")
        with self._lock:
            self._joined(auth)
            rows = self._db.execute("SELECT actor_id,kind FROM project_participants "
                                    "WHERE project_id=? AND active=1 ORDER BY actor_id",
                                    (auth.project_id,)).fetchall()
            return [dict(row) for row in rows]

    def snapshot(self, *, auth: AuthorizationContext) -> dict[str, Any]:
        """Consistent reconnect snapshot; resume ``events(after=cursor)`` after it."""
        auth.require(auth.project_id, "collaboration:read")
        with self._lock:
            self._db.execute("BEGIN")
            try:
                self._joined(auth)
                participants = [dict(row) for row in self._db.execute(
                    "SELECT actor_id,kind FROM project_participants "
                    "WHERE project_id=? AND active=1 ORDER BY actor_id", (auth.project_id,))]
                leases = [dict(row) for row in self._db.execute(
                    "SELECT resource_id,actor_id,token,expires_at FROM resource_leases "
                    "WHERE project_id=? AND expires_at>? ORDER BY resource_id",
                    (auth.project_id, self._clock()))]
                cursor = self._db.execute(
                    "SELECT COALESCE(MAX(sequence),0) FROM project_coordination_events WHERE project_id=?",
                    (auth.project_id,)).fetchone()[0]
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return {"project_id": auth.project_id, "participants": participants,
                "leases": leases, "event_cursor": cursor}

    def acquire(self, resource_id: str, *, auth: AuthorizationContext, ttl: float = 30.0) -> dict[str, Any]:
        auth.require(auth.project_id, "collaboration:lock")
        resource = _id(resource_id, "resource_id")
        if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not math.isfinite(ttl) or not 0 < ttl <= 3600:
            raise ProjectCoordinationError("lease TTL must be between 0 and 3600 seconds")
        with self._lock:
            self._begin()
            try:
                self._joined(auth)
                now = self._clock()
                row = self._db.execute("SELECT * FROM resource_leases WHERE project_id=? AND resource_id=?",
                                       (auth.project_id, resource)).fetchone()
                if row is not None and row["expires_at"] > now:
                    raise ProjectCoordinationError("resource already has an active exclusive lease")
                previous = self._db.execute("SELECT token FROM resource_fences WHERE project_id=? AND resource_id=?",
                                            (auth.project_id, resource)).fetchone()
                token = (previous["token"] if previous else 0) + 1
                self._db.execute("INSERT INTO resource_fences VALUES (?,?,?) ON CONFLICT(project_id,resource_id) "
                                 "DO UPDATE SET token=excluded.token", (auth.project_id, resource, token))
                lease_id = uuid.uuid4().hex
                expires_at = now + ttl
                self._db.execute("INSERT INTO resource_leases VALUES (?,?,?,?,?,?) "
                                 "ON CONFLICT(project_id,resource_id) DO UPDATE SET "
                                 "actor_id=excluded.actor_id,lease_id=excluded.lease_id,token=excluded.token,"
                                 "expires_at=excluded.expires_at",
                                 (auth.project_id, resource, auth.actor_id, lease_id, token, expires_at))
                self._event(auth.project_id, "lease.acquired", {"resource_id": resource,
                            "actor_id": auth.actor_id, "token": token, "expires_at": expires_at})
                self._db.commit()
                return {"project_id": auth.project_id, "resource_id": resource, "actor_id": auth.actor_id,
                        "lease_id": lease_id, "fencing_token": token, "expires_at": expires_at}
            except Exception:
                self._db.rollback()
                raise

    def _owned(self, resource: str, lease_id: str, token: int, auth: AuthorizationContext) -> sqlite3.Row:
        self._joined(auth)
        row = self._db.execute("SELECT * FROM resource_leases WHERE project_id=? AND resource_id=?",
                               (auth.project_id, resource)).fetchone()
        if (row is None or row["actor_id"] != auth.actor_id or row["lease_id"] != lease_id
                or row["token"] != token or row["expires_at"] <= self._clock()):
            raise ProjectCoordinationError("lease is expired, invalid, or not owned by actor")
        return row

    def renew(self, resource_id: str, lease_id: str, fencing_token: int, *,
              auth: AuthorizationContext, ttl: float = 30.0) -> dict[str, Any]:
        auth.require(auth.project_id, "collaboration:lock")
        if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not math.isfinite(ttl) or not 0 < ttl <= 3600:
            raise ProjectCoordinationError("lease TTL must be between 0 and 3600 seconds")
        resource = _id(resource_id, "resource_id")
        with self._lock:
            self._begin()
            try:
                self._owned(resource, lease_id, fencing_token, auth)
                expires_at = self._clock() + ttl
                self._db.execute("UPDATE resource_leases SET expires_at=? WHERE project_id=? AND resource_id=?",
                                 (expires_at, auth.project_id, resource))
                self._event(auth.project_id, "lease.renewed", {"resource_id": resource,
                            "actor_id": auth.actor_id, "token": fencing_token, "expires_at": expires_at})
                self._db.commit()
                return {"fencing_token": fencing_token, "lease_id": lease_id, "expires_at": expires_at}
            except Exception:
                self._db.rollback()
                raise

    def release(self, resource_id: str, lease_id: str, fencing_token: int, *, auth: AuthorizationContext) -> None:
        auth.require(auth.project_id, "collaboration:lock")
        resource = _id(resource_id, "resource_id")
        with self._lock:
            self._begin()
            try:
                self._owned(resource, lease_id, fencing_token, auth)
                self._db.execute("DELETE FROM resource_leases WHERE project_id=? AND resource_id=?",
                                 (auth.project_id, resource))
                self._event(auth.project_id, "lease.released", {"resource_id": resource,
                            "actor_id": auth.actor_id, "token": fencing_token})
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

    def check_lease(self, resource_id: str, lease_id: str, fencing_token: int, *,
                    auth: AuthorizationContext) -> bool:
        auth.require(auth.project_id, "collaboration:lock")
        with self._lock:
            try:
                self._owned(_id(resource_id, "resource_id"), lease_id, fencing_token, auth)
            except ProjectCoordinationError:
                return False
            return True

    def events(self, *, auth: AuthorizationContext, after: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        auth.require(auth.project_id, "collaboration:read")
        if (isinstance(after, bool) or not isinstance(after, int) or after < 0
                or isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000):
            raise ProjectCoordinationError("invalid event cursor or limit")
        with self._lock:
            self._joined(auth)
            rows = self._db.execute("SELECT sequence,event_type,payload FROM project_coordination_events "
                                    "WHERE project_id=? AND sequence>? ORDER BY sequence LIMIT ?",
                                    (auth.project_id, after, limit)).fetchall()
            return [{"sequence": row["sequence"], "event_type": row["event_type"],
                     "payload": json.loads(row["payload"])} for row in rows]


__all__ = ["ProjectCoordinationError", "ProjectCoordinationStore"]
