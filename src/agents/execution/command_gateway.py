"""Fail-closed, capability-bound command gateway for SLAI v2.4.

This module intentionally uses only the Python standard library and contains no
application-specific logic. Identity/authorization are supplied by trusted hosts;
commands never carry self-asserted permissions. SQLite is a durable *journal*,
not a distributed transaction coordinator or an authorization service.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
import math
import sqlite3
from pathlib import Path
from threading import RLock
from time import time
from typing import Any, Callable, Mapping, Protocol


class CommandRejected(ValueError):
    """Invalid command, capability, idempotency conflict, or permission failure."""


class Outcome(str, Enum):
    DENIED = "DENIED"
    REJECTED = "REJECTED"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCEEDED = "SUCCEEDED"
    COMMITTED = "COMMITTED"  # reserved for actual atomic resource transactions
    FAILED = "FAILED"
    INDETERMINATE = "INDETERMINATE"


def _json_map(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        raise CommandRejected("parameters must be a string-keyed mapping")
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode("utf-8")) > 262_144:
            raise CommandRejected("command parameters exceed 256 KiB")
        return encoded
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise CommandRejected("parameters must be bounded JSON-compatible values") from exc


@dataclass(frozen=True)
class Command:
    command_id: str
    idempotency_key: str
    operation: str
    resource: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    expected_version: str | None = None
    deadline: float | None = None
    trace_id: str = ""

    def __post_init__(self) -> None:
        for name in ("command_id", "idempotency_key", "operation", "resource"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 256:
                raise CommandRejected(f"{name} must be a nonempty string (up to 256 chars)")
        if self.deadline is not None and (not isinstance(self.deadline, (int, float)) or not math.isfinite(self.deadline)):
            raise CommandRejected("deadline must be a finite unix timestamp")
        if self.expected_version is not None and not isinstance(self.expected_version, str):
            raise CommandRejected("expected_version must be a string")
        if not isinstance(self.trace_id, str) or len(self.trace_id) > 256:
            raise CommandRejected("trace_id must be a string up to 256 chars")
        object.__setattr__(self, "parameters", json.loads(_json_map(self.parameters)))

    def fingerprint(self) -> str:
        """Stable intent digest; does not record the sensitive parameters in the journal."""
        payload = {"operation": self.operation, "resource": self.resource,
                   "parameters": self.parameters, "expected_version": self.expected_version}
        return sha256(_json_map(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Decision:
    allowed: bool
    principal_id: str
    reason: str = ""


class Authority(Protocol):
    def authorize(self, request_context: Any, command: Command, capability: "Capability") -> Decision:
        """Resolve trustworthy identity and current permissions; deny by default."""


class CapabilityHandler(Protocol):
    def __call__(self, command: Command) -> Any: ...


@dataclass(frozen=True)
class Capability:
    """Only trusted application bootstrap code may register capabilities."""
    name: str
    handler: CapabilityHandler
    effect: str = "external"  # read, external, or same-SQLite-connection transaction
    risk: str = "high"
    version_reader: Callable[[str], str] | None = None
    verifier: Callable[[Command, Any], bool] | None = None

    def __post_init__(self) -> None:
        if not self.name or self.effect not in {"read", "external", "transactional"} or self.risk not in {"low", "moderate", "high", "critical"} or not callable(self.handler):
            raise ValueError("invalid registered capability")


@dataclass(frozen=True)
class Result:
    command_id: str
    status: Outcome
    reason: str = ""
    result: Any = None  # returned live, never persisted in the journal
    replayed: bool = False


class CommandGateway:
    """Durable at-most-one-dispatch gateway for registered operations.

    The process must ensure all relevant side-effect adapters are accessible only
    through an equally enforced boundary. Crash after intent journal but before
    acknowledgment yields INDETERMINATE; it is NOT automatically retried.
    """

    def __init__(self, journal_path: str | Path, authority: Authority):
        if authority is None or not callable(getattr(authority, "authorize", None)):
            raise ValueError("trusted authorization authority is required")
        self.authority = authority
        self._capabilities: dict[str, Capability] = {}
        self._lock = RLock()
        self._db = sqlite3.connect(str(journal_path), check_same_thread=False, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("""CREATE TABLE IF NOT EXISTS command_journal (
            principal TEXT NOT NULL, resource TEXT NOT NULL, idempotency TEXT NOT NULL,
            fingerprint TEXT NOT NULL, command_id TEXT NOT NULL, status TEXT NOT NULL,
            updated REAL NOT NULL, PRIMARY KEY(principal,resource,idempotency))""")
        self._db.commit()

    def register(self, capability: Capability) -> None:
        with self._lock:
            if capability.name in self._capabilities:
                raise CommandRejected("capability already registered")
            self._capabilities[capability.name] = capability

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def execute(self, command: Command, *, request_context: Any) -> Result:
        if not isinstance(command, Command):
            raise CommandRejected("structured Command required")
        with self._lock:
            cap = self._capabilities.get(command.operation)
        if cap is None:
            return Result(command.command_id, Outcome.REJECTED, "unregistered capability")
        # Authorization occurs before both first dispatch and replay; stored results
        # must never become reusable grants after revocation.
        decision = self.authority.authorize(request_context, command, cap)
        if not isinstance(decision, Decision) or not decision.allowed or not decision.principal_id:
            return Result(command.command_id, Outcome.DENIED, "authorization denied")
        principal = decision.principal_id
        key = (principal, command.resource, command.idempotency_key)
        fingerprint = command.fingerprint()
        with self._lock:
            # SQLite BEGIN IMMEDIATE serializes durable claims across processes.
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute("""SELECT fingerprint,command_id,status FROM command_journal
                    WHERE principal=? AND resource=? AND idempotency=?""", key).fetchone()
                if row is not None:
                    self._db.commit()
                    if row[0] != fingerprint:
                        return Result(command.command_id, Outcome.REJECTED, "idempotency key conflict")
                    outcome = Outcome(row[2])
                    if outcome == Outcome.IN_PROGRESS:
                        # Another process may still be acting; don't retry.
                        return Result(row[1], Outcome.IN_PROGRESS, "already dispatched or awaiting reconciliation", replayed=True)
                    return Result(row[1], outcome, "replayed durable outcome; details not persisted", replayed=True)
                if command.deadline is not None and time() >= command.deadline:
                    self._db.commit()
                    return Result(command.command_id, Outcome.REJECTED, "deadline expired")
                if command.expected_version is not None:
                    if cap.version_reader is None:
                        self._db.commit()
                        return Result(command.command_id, Outcome.REJECTED, "version check unsupported")
                    if cap.version_reader(command.resource) != command.expected_version:
                        self._db.commit()
                        return Result(command.command_id, Outcome.REJECTED, "resource version conflict")
                if cap.effect == "transactional":
                    # This handler must use ONLY the supplied SQLite connection for
                    # transactional writes; its result and journal share one COMMIT.
                    try:
                        value = cap.handler(command, self._db)
                        if cap.verifier is not None and not cap.verifier(command, value):
                            raise CommandRejected("transaction postcondition failed")
                        self._db.execute("""INSERT INTO command_journal
                            (principal,resource,idempotency,fingerprint,command_id,status,updated)
                            VALUES (?,?,?,?,?,?,?)""", (*key, fingerprint, command.command_id, Outcome.COMMITTED.value, time()))
                        self._db.commit()
                        return Result(command.command_id, Outcome.COMMITTED, result=value)
                    except Exception:
                        self._db.rollback()
                        return Result(command.command_id, Outcome.FAILED, "transaction rolled back")
                self._db.execute("""INSERT INTO command_journal
                    (principal,resource,idempotency,fingerprint,command_id,status,updated)
                    VALUES (?,?,?,?,?,?,?)""", (*key, fingerprint, command.command_id, Outcome.IN_PROGRESS.value, time()))
                self._db.commit()  # durable *before* invoking irreversible effects
            except BaseException:
                self._db.rollback()
                raise
        try:
            value = cap.handler(command)
            if cap.verifier is not None and not cap.verifier(command, value):
                status, reason = Outcome.INDETERMINATE, "postcondition not verified"
            else:
                status, reason = Outcome.SUCCEEDED, ""
        except Exception:
            # Even an exception can occur AFTER an external side effect.
            status = Outcome.FAILED if cap.effect == "read" else Outcome.INDETERMINATE
            value, reason = None, "handler failed; effect may have occurred" if cap.effect != "read" else "read handler failed"
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute("""UPDATE command_journal SET status=?,updated=?
                    WHERE principal=? AND resource=? AND idempotency=?""", (status.value, time(), *key))
                self._db.commit()
            except BaseException:
                self._db.rollback()
                # Never advertise success when durable outcome recording failed.
                return Result(command.command_id, Outcome.INDETERMINATE, "durable outcome update failed")
        return Result(command.command_id, status, reason, value if status in {Outcome.COMMITTED, Outcome.SUCCEEDED} else None)

    def reconcile(self, principal: str, resource: str, key: str, *,
                  observed_outcome: Outcome, authoritative_evidence: bool = False) -> bool:
        """Administrative reconciliation, gated by a separate trusted caller.

        This method MUST NOT be exposed as an untrusted command capability.
        It cannot turn an uncertain outcome into success without external evidence.
        """
        # The caller's boolean is not authority. A separately trusted recovery
        # policy must authenticate the requesting administrator and evidence.
        check = getattr(self.authority, "authorize_reconciliation", None)
        if (not authoritative_evidence or not callable(check)
                or not check(principal, resource, key, observed_outcome)
                or observed_outcome not in {Outcome.SUCCEEDED, Outcome.COMMITTED, Outcome.FAILED}):
            raise CommandRejected("trusted reconciliation authorization and evidence required")
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cur = self._db.execute("""UPDATE command_journal SET status=?,updated=?
                    WHERE principal=? AND resource=? AND idempotency=? AND status IN (?,?)""",
                    (observed_outcome.value,time(),principal,resource,key,Outcome.IN_PROGRESS.value,Outcome.INDETERMINATE.value))
                self._db.commit()
                return bool(cur.rowcount)
            except BaseException:
                self._db.rollback()
                raise

    def inspect(self, principal: str, resource: str, key: str) -> Outcome | None:
        with self._lock:
            row = self._db.execute("""SELECT status FROM command_journal WHERE
                principal=? AND resource=? AND idempotency=?""", (principal,resource,key)).fetchone()
        return Outcome(row[0]) if row else None
