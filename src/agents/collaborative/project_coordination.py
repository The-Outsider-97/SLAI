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

import hashlib
import json
import math
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.agents.planning.workflow_runtime import AuthorizationContext


class ProjectCoordinationError(ValueError):
    """Invalid or conflicting project collaboration operation."""


def _id(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectCoordinationError(f"{label} must be a nonempty string")
    return value.strip()


def _json_record(value: Any, label: str, *, max_bytes: int = 65536) -> str:
    """Validate a bounded, portable JSON record (no pickle or NaN values)."""
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProjectCoordinationError(f"{label} must be JSON serializable") from exc
    if len(encoded.encode("utf-8")) > max_bytes:
        raise ProjectCoordinationError(f"{label} exceeds {max_bytes} UTF-8 bytes")
    return encoded


def _references(values: Any, label: str) -> list[str]:
    if not isinstance(values, (tuple, list)) or len(values) > 128:
        raise ProjectCoordinationError(f"{label} must be a list of up to 128 references")
    result = [_id(value, label) for value in values]
    if len(result) != len(set(result)):
        raise ProjectCoordinationError(f"{label} must contain unique references")
    return result


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
            self._db.execute("CREATE TABLE IF NOT EXISTS inter_agent_messages ("
                             "message_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, sender_id TEXT NOT NULL, "
                             "recipient_id TEXT NOT NULL, message_type TEXT NOT NULL, task_id TEXT, "
                             "correlation_id TEXT, causation_id TEXT, payload TEXT NOT NULL, "
                             "context_refs TEXT NOT NULL, fingerprint TEXT NOT NULL, "
                             "idempotency_key TEXT, created_at REAL NOT NULL, expires_at REAL, acked_at REAL, "
                             "UNIQUE(project_id,sender_id,idempotency_key))")
            self._db.execute("CREATE INDEX IF NOT EXISTS ix_inter_agent_inbox "
                             "ON inter_agent_messages(project_id,recipient_id,created_at)")
            self._db.execute("CREATE TABLE IF NOT EXISTS inter_agent_handoffs ("
                             "handoff_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, task_id TEXT NOT NULL, "
                             "sender_id TEXT NOT NULL, recipient_id TEXT NOT NULL, capability TEXT NOT NULL, "
                             "objective TEXT NOT NULL, details TEXT NOT NULL, context_refs TEXT NOT NULL, "
                             "fingerprint TEXT NOT NULL, idempotency_key TEXT, "
                             "status TEXT NOT NULL, owner_id TEXT NOT NULL, created_at REAL NOT NULL, "
                             "expires_at REAL, accepted_at REAL, finished_at REAL, result_refs TEXT, failure TEXT, "
                             "UNIQUE(project_id,sender_id,idempotency_key))")
            self._db.execute("CREATE INDEX IF NOT EXISTS ix_inter_agent_handoffs "
                             "ON inter_agent_handoffs(project_id,recipient_id,status)")
            self._db.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_inter_agent_active_task "
                             "ON inter_agent_handoffs(project_id,task_id) "
                             "WHERE status IN ('pending','accepted')")

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

    # Durable communication metadata belongs beside project participation and
    # coordination events. SharedMemory is an optional transient notification/
    # context cache; its pub/sub delivery is NOT a durable task queue.
    def _active_recipient(self, project_id: str, recipient_id: str) -> None:
        row = self._db.execute(
            "SELECT active FROM project_participants WHERE project_id=? AND actor_id=?",
            (project_id, recipient_id),
        ).fetchone()
        if row is None or not row["active"]:
            raise ProjectCoordinationError("recipient must be an active project participant")

    def _deadline(self, ttl: float | None) -> float | None:
        if ttl is None:
            return None
        if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not math.isfinite(ttl) or not 0 < ttl <= 86400:
            raise ProjectCoordinationError("TTL must be a finite number in (0, 86400] seconds")
        return self._clock() + ttl

    @staticmethod
    def _message(row: sqlite3.Row) -> dict[str, Any]:
        return {**dict(row), "schema_version": "1.0", "payload": json.loads(row["payload"]),
                "context_refs": json.loads(row["context_refs"])}

    def send_message(self, recipient_id: str, message_type: str, payload: dict[str, Any], *,
                     auth: AuthorizationContext, task_id: str | None = None,
                     correlation_id: str | None = None, causation_id: str | None = None,
                     context_refs: tuple[str, ...] | list[str] = (),
                     idempotency_key: str | None = None, ttl: float | None = None) -> dict[str, Any]:
        """Store a directed message before notifying anyone; the sender is host-derived.

        Acknowledgment only means receipt, never successful task execution.
        Consumers must apply their own side-effect idempotency controls.
        """
        auth.require(auth.project_id, "collaboration:message")
        recipient = _id(recipient_id, "recipient_id")
        kind = _id(message_type, "message_type").upper()
        if kind not in {"REQUEST", "RESPONSE", "EVENT", "PROGRESS", "RESULT", "ERROR", "CANCEL", "STATUS", "CONTEXT_UPDATE"}:
            raise ProjectCoordinationError("unsupported message_type")
        if not isinstance(payload, dict):
            raise ProjectCoordinationError("message payload must be a JSON object")
        data = _json_record(payload, "payload")
        refs = _json_record(_references(context_refs, "context_refs"), "context_refs")
        task = _id(task_id, "task_id") if task_id is not None else None
        correlation = _id(correlation_id, "correlation_id") if correlation_id is not None else None
        causation = _id(causation_id, "causation_id") if causation_id is not None else None
        idem = _id(idempotency_key, "idempotency_key") if idempotency_key is not None else None
        deadline = self._deadline(ttl)
        fingerprint = hashlib.sha256(_json_record(
            [recipient, kind, task, correlation, causation, data, refs, ttl], "message identity"
        ).encode()).hexdigest()
        with self._lock:
            self._begin()
            try:
                self._joined(auth)
                self._active_recipient(auth.project_id, recipient)
                if idem is not None:
                    prior = self._db.execute("SELECT * FROM inter_agent_messages WHERE "
                                             "project_id=? AND sender_id=? AND idempotency_key=?",
                                             (auth.project_id, auth.actor_id, idem)).fetchone()
                    if prior is not None:
                        if prior["fingerprint"] != fingerprint:
                            raise ProjectCoordinationError("idempotency key reused with different message")
                        self._db.rollback()
                        return self._message(prior)
                message_id = uuid.uuid4().hex
                self._db.execute("INSERT INTO inter_agent_messages VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                 (message_id, auth.project_id, auth.actor_id, recipient, kind, task,
                                  correlation, causation, data, refs, fingerprint, idem, self._clock(),
                                  deadline, None))
                self._event(auth.project_id, "message.sent", {"message_id": message_id,
                            "sender_id": auth.actor_id, "recipient_id": recipient, "message_type": kind})
                row = self._db.execute("SELECT * FROM inter_agent_messages WHERE message_id=?", (message_id,)).fetchone()
                self._db.commit()
                return self._message(row)
            except Exception:
                self._db.rollback()
                raise

    def inbox(self, *, auth: AuthorizationContext, after: float = 0.0,
              limit: int = 100, unacknowledged_only: bool = True) -> list[dict[str, Any]]:
        auth.require(auth.project_id, "collaboration:message")
        if not isinstance(after, (float, int)) or isinstance(after, bool) or not math.isfinite(after) or after < 0:
            raise ProjectCoordinationError("invalid inbox cursor")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ProjectCoordinationError("invalid inbox limit")
        with self._lock:
            self._joined(auth)
            rows = self._db.execute(
                "SELECT * FROM inter_agent_messages WHERE project_id=? AND recipient_id=? AND "
                "created_at>=? AND (expires_at IS NULL OR expires_at>?) AND "
                "(?=0 OR acked_at IS NULL) ORDER BY created_at,message_id LIMIT ?",
                (auth.project_id, auth.actor_id, after, self._clock(), int(unacknowledged_only), limit),
            ).fetchall()
            return [self._message(row) for row in rows]

    def acknowledge_message(self, message_id: str, *, auth: AuthorizationContext) -> dict[str, Any]:
        auth.require(auth.project_id, "collaboration:message")
        with self._lock:
            self._begin()
            try:
                self._joined(auth)
                row = self._db.execute("SELECT * FROM inter_agent_messages WHERE message_id=? AND project_id=?",
                                       (_id(message_id, "message_id"), auth.project_id)).fetchone()
                if row is None or row["recipient_id"] != auth.actor_id:
                    raise ProjectCoordinationError("message not found in recipient inbox")
                if row["expires_at"] is not None and row["expires_at"] <= self._clock():
                    raise ProjectCoordinationError("message has expired")
                if row["acked_at"] is None:
                    self._db.execute("UPDATE inter_agent_messages SET acked_at=? WHERE message_id=?",
                                     (self._clock(), message_id))
                    self._event(auth.project_id, "message.acknowledged", {"message_id": message_id,
                                "recipient_id": auth.actor_id})
                row = self._db.execute("SELECT * FROM inter_agent_messages WHERE message_id=?", (message_id,)).fetchone()
                self._db.commit()
                return self._message(row)
            except Exception:
                self._db.rollback()
                raise

    @staticmethod
    def _handoff(row: sqlite3.Row) -> dict[str, Any]:
        return {**dict(row), "schema_version": "1.0", "details": json.loads(row["details"]),
                "context_refs": json.loads(row["context_refs"]),
                "result_refs": json.loads(row["result_refs"]) if row["result_refs"] else None}

    def _expire_pending(self, project_id: str) -> None:
        rows = self._db.execute("SELECT handoff_id FROM inter_agent_handoffs WHERE project_id=? "
                                "AND status='pending' AND expires_at IS NOT NULL AND expires_at<=?",
                                (project_id, self._clock())).fetchall()
        for row in rows:
            self._db.execute("UPDATE inter_agent_handoffs SET status='expired',finished_at=? WHERE handoff_id=?",
                             (self._clock(), row["handoff_id"]))
            self._event(project_id, "handoff.expired", {"handoff_id": row["handoff_id"]})

    def prepare_handoff(self, recipient_id: str, task_id: str, capability: str, objective: str, *,
                        auth: AuthorizationContext, details: dict[str, Any] | None = None,
                        context_refs: tuple[str, ...] | list[str] = (),
                        idempotency_key: str | None = None, ttl: float | None = 3600.0) -> dict[str, Any]:
        """Prepare an exclusive handoff. Sender remains owner until acceptance.

        This coordinates the *handoff*, not an external workflow/host commit;
        owners of authoritative task state must validate their own revisions.
        """
        auth.require(auth.project_id, "collaboration:handoff")
        recipient, task = _id(recipient_id, "recipient_id"), _id(task_id, "task_id")
        capability, objective = _id(capability, "capability"), _id(objective, "objective")
        if recipient == auth.actor_id:
            raise ProjectCoordinationError("handoff recipient must differ from sender")
        if details is None:
            details = {}
        if not isinstance(details, dict):
            raise ProjectCoordinationError("handoff details must be a JSON object")
        data = _json_record(details, "handoff details")
        refs = _json_record(_references(context_refs, "context_refs"), "context_refs")
        idem = _id(idempotency_key, "idempotency_key") if idempotency_key is not None else None
        deadline = self._deadline(ttl)
        fingerprint = hashlib.sha256(_json_record(
            [recipient, task, capability, objective, data, refs, ttl], "handoff identity"
        ).encode()).hexdigest()
        with self._lock:
            self._begin()
            try:
                self._joined(auth)
                self._active_recipient(auth.project_id, recipient)
                self._expire_pending(auth.project_id)
                if idem is not None:
                    prior = self._db.execute("SELECT * FROM inter_agent_handoffs WHERE "
                                             "project_id=? AND sender_id=? AND idempotency_key=?",
                                             (auth.project_id, auth.actor_id, idem)).fetchone()
                    if prior is not None:
                        if prior["fingerprint"] != fingerprint:
                            raise ProjectCoordinationError("idempotency key reused with different handoff")
                        self._db.commit()  # Commit expiry cleanup, not a duplicate event.
                        return self._handoff(prior)
                handoff_id = uuid.uuid4().hex
                try:
                    self._db.execute("INSERT INTO inter_agent_handoffs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                     (handoff_id, auth.project_id, task, auth.actor_id, recipient,
                                      capability, objective, data, refs, fingerprint, idem, "pending",
                                      auth.actor_id, self._clock(), deadline, None, None, None, None))
                except sqlite3.IntegrityError as exc:
                    raise ProjectCoordinationError("task already has an active exclusive handoff") from exc
                self._event(auth.project_id, "handoff.prepared", {"handoff_id": handoff_id,
                            "task_id": task, "recipient_id": recipient})
                row = self._db.execute("SELECT * FROM inter_agent_handoffs WHERE handoff_id=?", (handoff_id,)).fetchone()
                self._db.commit()
                return self._handoff(row)
            except Exception:
                self._db.rollback()
                raise

    def _get_handoff(self, handoff_id: str, auth: AuthorizationContext) -> sqlite3.Row:
        self._joined(auth)
        row = self._db.execute("SELECT * FROM inter_agent_handoffs WHERE handoff_id=? AND project_id=?",
                               (_id(handoff_id, "handoff_id"), auth.project_id)).fetchone()
        if row is None or auth.actor_id not in {row["sender_id"], row["recipient_id"]}:
            raise ProjectCoordinationError("handoff not found for participant")
        return row

    def get_handoff(self, handoff_id: str, *, auth: AuthorizationContext) -> dict[str, Any]:
        auth.require(auth.project_id, "collaboration:handoff")
        with self._lock:
            self._begin()
            try:
                self._joined(auth)
                self._expire_pending(auth.project_id)
                row = self._get_handoff(handoff_id, auth)
                self._db.commit()
                return self._handoff(row)
            except Exception:
                self._db.rollback()
                raise

    def handoff_inbox(self, *, auth: AuthorizationContext, status: str = "pending",
                      limit: int = 100) -> list[dict[str, Any]]:
        """Discover assigned handoffs after restart or missed transient hints."""
        auth.require(auth.project_id, "collaboration:handoff")
        if status not in {"pending", "accepted", "completed", "rejected", "failed", "cancelled", "expired"}:
            raise ProjectCoordinationError("invalid handoff inbox status")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ProjectCoordinationError("invalid handoff inbox limit")
        with self._lock:
            self._begin()
            try:
                self._joined(auth)
                self._expire_pending(auth.project_id)
                rows = self._db.execute(
                    "SELECT * FROM inter_agent_handoffs WHERE project_id=? AND recipient_id=? "
                    "AND status=? ORDER BY created_at,handoff_id LIMIT ?",
                    (auth.project_id, auth.actor_id, status, limit),
                ).fetchall()
                self._db.commit()
                return [self._handoff(row) for row in rows]
            except Exception:
                self._db.rollback()
                raise

    def transition_handoff(self, handoff_id: str, action: str, *, auth: AuthorizationContext,
                           result_refs: dict[str, str] | None = None, reason: str | None = None) -> dict[str, Any]:
        """Accept, reject, cancel, complete or fail a handoff transactionally.

        An accepted recipient owns the handoff record, but the external
        workflow must still validate its own revision/commit authorization.
        """
        auth.require(auth.project_id, "collaboration:handoff")
        transitions = {"accept": ("pending", "accepted"), "reject": ("pending", "rejected"),
                       "cancel": ("pending", "cancelled"), "complete": ("accepted", "completed"),
                       "fail": ("accepted", "failed")}
        if action not in transitions:
            raise ProjectCoordinationError("unsupported handoff action")
        if result_refs is not None and (not isinstance(result_refs, dict) or any(
                not isinstance(k, str) or not k or not isinstance(v, str) or not v
                for k, v in result_refs.items())):
            raise ProjectCoordinationError("result_refs must map nonempty strings to nonempty strings")
        result_data = _json_record(result_refs or {}, "result_refs") if action == "complete" else None
        if reason is not None:
            reason = _id(reason, "reason")
            if len(reason) > 500:
                raise ProjectCoordinationError("reason is too long")
        with self._lock:
            self._begin()
            try:
                self._joined(auth)
                self._expire_pending(auth.project_id)
                row = self._get_handoff(handoff_id, auth)
                before, after = transitions[action]
                authorized_actor = row["sender_id"] if action == "cancel" else row["recipient_id"]
                if auth.actor_id != authorized_actor:
                    raise ProjectCoordinationError("actor cannot perform this handoff transition")
                if row["status"] != before:
                    raise ProjectCoordinationError(f"invalid handoff transition from {row['status']} to {after}")
                owner = row["recipient_id"] if action == "accept" else row["owner_id"]
                self._db.execute("UPDATE inter_agent_handoffs SET status=?,owner_id=?,accepted_at=?, "
                                 "finished_at=?,result_refs=?,failure=? WHERE handoff_id=?",
                                 (after, owner, self._clock() if action == "accept" else row["accepted_at"],
                                  self._clock() if action in {"reject", "cancel", "complete", "fail"} else None,
                                  result_data, reason if action in {"reject", "fail"} else None, handoff_id))
                self._event(auth.project_id, f"handoff.{after}", {"handoff_id": handoff_id,
                            "task_id": row["task_id"], "actor_id": auth.actor_id})
                updated = self._db.execute("SELECT * FROM inter_agent_handoffs WHERE handoff_id=?", (handoff_id,)).fetchone()
                self._db.commit()
                return self._handoff(updated)
            except Exception:
                self._db.rollback()
                raise


__all__ = ["ProjectCoordinationError", "ProjectCoordinationStore"]
