"""Fail-closed, capability-bound command gateway for SLAI v2.4.

This module intentionally uses only the Python standard library and contains no
application-specific logic. Identity/authorization are supplied by trusted hosts;
commands never carry self-asserted permissions. SQLite is a durable *journal*,
not a distributed transaction coordinator or an authorization service.
"""
from __future__ import annotations

import json
import sqlite3

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from time import time
from typing import Any, Callable, Mapping, Optional, Protocol, cast

from .utils.execution_error import *
from .utils.execution_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Command Gateway")
printer = PrettyPrinter()


# ---------------------------------------------------------------------------
# Outcome and Result
# ---------------------------------------------------------------------------

class Outcome(str, Enum):
    DENIED = "DENIED"
    REJECTED = "REJECTED"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCEEDED = "SUCCEEDED"
    COMMITTED = "COMMITTED"  # reserved for actual atomic resource transactions
    FAILED = "FAILED"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class Result:
    command_id: str
    status: Outcome
    reason: str = ""
    result: Any = None  # returned live, never persisted in the journal
    replayed: bool = False
    reason_code: str = ""

    @property
    def is_success(self) -> bool:
        return (
            self.status in (Outcome.SUCCEEDED, Outcome.COMMITTED)
            and not self.replayed
        )


def _reject(
    command: "Command",
    reason: str,
    *,
    reason_code: str,
    status: Outcome = Outcome.REJECTED,
) -> Result:
    """Construct a rejected Result with standard metadata."""
    return Result(
        command.command_id,
        status,
        reason,
        reason_code=reason_code,
    )


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Command:
    command_id: str
    idempotency_key: str
    operation: str
    resource: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    expected_version: Optional[str] = None
    deadline: Optional[float] = None
    trace_id: str = ""

    def __post_init__(self) -> None:
        for name in ("command_id", "idempotency_key", "operation", "resource"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 256:
                raise CommandRejected(
                    f"{name} must be a nonempty string (up to 256 chars)",
                    reason_code=CommandRejected.INVALID_COMMAND,
                    stage="command_validation",
                    details={"field": name},
                )

        if self.deadline is not None and not is_finite_number(self.deadline):
            # NOTE: is_finite_number rejects bool, which the old isinstance
            # check silently accepted. This is the intended stricter behaviour.
            raise CommandRejected(
                "deadline must be a finite unix timestamp",
                reason_code=CommandRejected.INVALID_COMMAND,
                stage="command_validation",
                details={"field": "deadline"},
            )

        if self.expected_version is not None and not isinstance(self.expected_version, str):
            raise CommandRejected(
                "expected_version must be a string",
                reason_code=CommandRejected.INVALID_COMMAND,
                stage="command_validation",
                details={"field": "expected_version"},
            )

        if not isinstance(self.trace_id, str) or len(self.trace_id) > 256:
            raise CommandRejected(
                "trace_id must be a string up to 256 chars",
                reason_code=CommandRejected.INVALID_COMMAND,
                stage="command_validation",
                details={"field": "trace_id"},
            )

        try:
            normalized_parameters = json.loads(_json_map(self.parameters))
        except CommandRejected as exc:
            detail = exc.args[0] if exc.args else "invalid parameters"
            raise CommandRejected(
                f"invalid command parameters: {detail}",
                reason_code=CommandRejected.INVALID_PARAMETERS,
                stage="command_validation",
                details={"field": "parameters"},
            ) from exc

        object.__setattr__(self, "parameters", normalized_parameters)

    def fingerprint(self) -> str:
        """Stable intent digest; sensitive parameters are not stored in the journal.

        Uses the shared helper :func:`_json_map` for strict JSON serialization
        (``allow_nan=False``, sorted keys, compact separators) and
        :func:`stable_digest` for the hash so future helper evolution stays
        consistent.
        """
        payload = {
            "operation": self.operation,
            "resource": self.resource,
            "parameters": self.parameters,
            "expected_version": self.expected_version,
        }
        return stable_digest(_json_map(payload), algorithm="sha256")


# ---------------------------------------------------------------------------
# Authorization and Capability types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    allowed: bool
    principal_id: str
    reason: str = ""


class Authority(Protocol):
    def authorize(
        self, request_context: Any, command: Command, capability: "Capability"
    ) -> Decision:
        """Resolve trustworthy identity and current permissions; deny by default."""
        ...


class CapabilityHandler(Protocol):
    def __call__(self, command: Command) -> Any: ...


@dataclass(frozen=True)
class Capability:
    """Only trusted application bootstrap code may register capabilities."""
    name: str
    handler: CapabilityHandler
    effect: str = "external"  # read, external, or same-SQLite-connection transaction
    risk: str = "high"
    version_reader: Optional[Callable[[str], str]] = None
    verifier: Optional[Callable[[Command, Any], bool]] = None

    def __post_init__(self) -> None:
        if (
            not self.name
            or self.effect not in {"read", "external", "transactional"}
            or self.risk not in {"low", "moderate", "high", "critical"}
            or not callable(self.handler)
        ):
            raise CommandRejected(
                "invalid registered capability",
                reason_code=CommandRejected.INVALID_CAPABILITY,
                stage="capability_registration",
                details={"capability": self.name or "<unnamed>"},
            )


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------

class CommandGateway:
    """Durable at-most-one-dispatch gateway for registered operations.

    The process must ensure all relevant side-effect adapters are accessible only
    through an equally enforced boundary. Crash after intent journal but before
    acknowledgment yields ``INDETERMINATE``; it is NOT automatically retried.
    """

    def __init__(self, journal_path: str | Path, authority: Authority):
        if authority is None or not callable(getattr(authority, "authorize", None)):
            raise ValueError("trusted authorization authority is required")

        self.authority = authority
        self._capabilities: dict[str, Capability] = {}
        self._lock = RLock()

        try:
            self._db = sqlite3.connect(
                str(journal_path), check_same_thread=False, timeout=30
            )
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS command_journal (
                    principal TEXT NOT NULL, resource TEXT NOT NULL,
                    idempotency TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    command_id TEXT NOT NULL, status TEXT NOT NULL,
                    updated REAL NOT NULL,
                    PRIMARY KEY(principal,resource,idempotency))"""
            )
            self._db.commit()
        except sqlite3.Error as exc:
            logger.exception("Failed to initialise command journal at %s", journal_path)
            raise RuntimeError(
                f"failed to initialise command journal: {exc}"
            ) from exc

        logger.info("Command Gateway initialised journal=%s", journal_path)

    # -- Capability registration -------------------------------------------

    def register(self, capability: Capability) -> None:
        with self._lock:
            if capability.name in self._capabilities:
                raise CommandRejected(
                    "capability already registered",
                    reason_code=CommandRejected.CAPABILITY_ALREADY_REGISTERED,
                    stage="capability_registration",
                    operation=capability.name,
                )
            self._capabilities[capability.name] = capability
            logger.info(
                "Registered capability name=%s effect=%s risk=%s",
                capability.name,
                capability.effect,
                capability.risk,
            )

    def close(self) -> None:
        with self._lock:
            try:
                self._db.close()
            except sqlite3.Error:
                logger.exception("Error while closing command journal")

    # -- Execute -----------------------------------------------------------

    def execute(self, command: Command, *, request_context: Any) -> Result:
        if not isinstance(command, Command):
            raise CommandRejected(
                "structured Command required",
                reason_code=CommandRejected.INVALID_COMMAND,
                stage="gateway_execute",
            )

        with self._lock:
            cap = self._capabilities.get(command.operation)
        if cap is None:
            logger.warning(
                "Rejecting unregistered capability operation=%s command_id=%s",
                command.operation,
                command.command_id,
            )
            return _reject(
                command,
                "unregistered capability",
                reason_code=CommandRejected.UNREGISTERED_CAPABILITY,
            )

        # Authorization runs before both first dispatch AND replay: a stored
        # result must never become a reusable grant after revocation.
        try:
            decision = self.authority.authorize(request_context, command, cap)
        except Exception as exc:
            logger.exception(
                "Authorization authority raised operation=%s command_id=%s",
                command.operation,
                command.command_id,
            )
            return _reject(
                command,
                f"authorization error: {type(exc).__name__}",
                reason_code=CommandRejected.AUTHORIZATION_DENIED,
                status=Outcome.DENIED,
            )

        if (
            not isinstance(decision, Decision)
            or not decision.allowed
            or not decision.principal_id
        ):
            logger.info(
                "Authorization denied operation=%s command_id=%s",
                command.operation,
                command.command_id,
            )
            return _reject(
                command,
                "authorization denied",
                reason_code=CommandRejected.AUTHORIZATION_DENIED,
                status=Outcome.DENIED,
            )

        principal = decision.principal_id
        try:
            return self._execute_in_transaction(
                command=command,
                capability=cap,
                principal=principal,
            )
        except CommandRejected:
            raise
        except sqlite3.Error as exc:
            logger.exception(
                "SQLite failure during command execution operation=%s command_id=%s",
                command.operation,
                command.command_id,
            )
            return _reject(
                command,
                f"journal failure: {type(exc).__name__}",
                reason_code=CommandRejected.DISPATCH_STATE_ERROR,
                status=Outcome.INDETERMINATE,
            )
        except Exception as exc:
            logger.exception(
                "Unexpected gateway failure operation=%s command_id=%s",
                command.operation,
                command.command_id,
            )
            raise CommandRejected(
                f"unexpected gateway error: {type(exc).__name__}",
                reason_code=CommandRejected.UNEXPECTED_ERROR,
                stage="gateway_execute",
                operation=command.operation,
                command_id=command.command_id,
                details={"error_type": type(exc).__name__},
            ) from exc

    # -- Dispatch pipeline (unchanged transaction shape, better logging) ---

    def _execute_in_transaction(
        self,
        *,
        command: Command,
        capability: Capability,
        principal: str,
    ) -> Result:
        key = (principal, command.resource, command.idempotency_key)
        fingerprint = command.fingerprint()

        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                existing = self._db.execute(
                    """SELECT fingerprint,command_id,status FROM command_journal
                       WHERE principal=? AND resource=? AND idempotency=?""",
                    key,
                ).fetchone()
                if existing is not None:
                    self._db.commit()
                    return self._handle_existing_journal_entry(command, existing)

                if command.deadline is not None and time() >= command.deadline:
                    self._db.commit()
                    logger.info(
                        "Deadline expired command_id=%s deadline=%s",
                        command.command_id,
                        command.deadline,
                    )
                    return _reject(
                        command,
                        "deadline expired",
                        reason_code=CommandRejected.DEADLINE_EXPIRED,
                    )

                if command.expected_version is not None:
                    version_result = self._check_expected_version(command, capability)
                    if version_result is not None:
                        self._db.commit()
                        return version_result

                if capability.effect == "transactional":
                    return self._execute_transactional_locked(
                        command=command,
                        capability=capability,
                        key=key,
                        fingerprint=fingerprint,
                    )

                self._db.execute(
                    """INSERT INTO command_journal
                       (principal,resource,idempotency,fingerprint,command_id,status,updated)
                       VALUES (?,?,?,?,?,?,?)""",
                    (*key, fingerprint, command.command_id, Outcome.IN_PROGRESS.value, time()),
                )
                self._db.commit()  # durable before invoking irreversible effects
            except BaseException:
                self._db.rollback()
                raise

        # Side-effect invocation happens outside the write lock.
        return self._invoke_handler(command=command, capability=capability, key=key)

    def _handle_existing_journal_entry(
        self,
        command: Command,
        existing: tuple,
    ) -> Result:
        fingerprint, command_id, status = existing
        if fingerprint != command.fingerprint():
            logger.warning(
                "Idempotency key conflict command_id=%s stored_fingerprint=%s",
                command.command_id,
                fingerprint[:12],
            )
            return _reject(
                command,
                "idempotency key conflict",
                reason_code=CommandRejected.IDEMPOTENCY_CONFLICT,
            )

        outcome = Outcome(status)
        if outcome == Outcome.IN_PROGRESS:
            logger.warning(
                "Replayed in-progress dispatch command_id=%s", command_id
            )
            return Result(
                command_id,
                Outcome.IN_PROGRESS,
                "already dispatched or awaiting reconciliation",
                replayed=True,
                reason_code=CommandRejected.REPLAY_DETECTED,
            )
        logger.info(
            "Replayed durable outcome command_id=%s status=%s",
            command_id,
            outcome.value,
        )
        return Result(
            command_id,
            outcome,
            "replayed durable outcome; details not persisted",
            replayed=True,
            reason_code=CommandRejected.REPLAY_DETECTED,
        )

    def _check_expected_version(
        self,
        command: Command,
        capability: Capability,
    ) -> Optional[Result]:
        if capability.version_reader is None:
            logger.warning("Version check unsupported command_id=%s", command.command_id)
            return _reject(
                command,
                "version check unsupported",
                reason_code=CommandRejected.VERSION_CHECK_UNSUPPORTED,
            )
        try:
            current_version = capability.version_reader(command.resource)
        except Exception as exc:
            logger.exception("version_reader raised command_id=%s", command.command_id)
            return _reject(
                command,
                f"version reader failed: {type(exc).__name__}",
                reason_code=CommandRejected.VERSION_CHECK_UNSUPPORTED,
            )
        if current_version != command.expected_version:
            logger.warning(
                "Resource version conflict resource=%s expected=%s actual=%s",
                command.resource,
                command.expected_version,
                current_version,
            )
            return _reject(
                command,
                "resource version conflict",
                reason_code=CommandRejected.VERSION_CONFLICT,
            )
        return None

    def _execute_transactional_locked(
        self,
        *,
        command: Command,
        capability: Capability,
        key: tuple,
        fingerprint: str,
    ) -> Result:
        """Transactional handlers share one COMMIT with their journal row."""
        try:
            transactional_handler = cast(
                Callable[[Command, sqlite3.Connection], Any], capability.handler
            )
            value = transactional_handler(command, self._db)
            if capability.verifier is not None and not capability.verifier(command, value):
                raise CommandRejected(
                    "transaction postcondition failed",
                    reason_code=CommandRejected.POSTCONDITION_FAILED,
                    stage="transaction_verifier",
                    operation=command.operation,
                    command_id=command.command_id,
                )
            self._db.execute(
                """INSERT INTO command_journal
                   (principal,resource,idempotency,fingerprint,command_id,status,updated)
                   VALUES (?,?,?,?,?,?,?)""",
                (*key, fingerprint, command.command_id, Outcome.COMMITTED.value, time()),
            )
            self._db.commit()
            logger.debug("Transactional command committed command_id=%s", command.command_id)
            return Result(
                command.command_id,
                Outcome.COMMITTED,
                result=value,
            )
        except CommandRejected:
            self._db.rollback()
            raise
        except Exception:
            self._db.rollback()
            logger.exception("Transactional handler failed command_id=%s", command.command_id)
            return _reject(
                command,
                "transaction rolled back",
                reason_code=CommandRejected.UNEXPECTED_ERROR,
                status=Outcome.FAILED,
            )

    def _invoke_handler(self, *, command: Command, capability: Capability, key: tuple) -> Result:
        """Invoke an external/read handler and durably record the outcome."""
        try:
            value = capability.handler(command)
            if capability.verifier is not None and not capability.verifier(command, value):
                status = Outcome.INDETERMINATE
                reason = "postcondition not verified"
                reason_code = CommandRejected.POSTCONDITION_FAILED
            else:
                status = Outcome.SUCCEEDED
                reason = ""
                reason_code = ""
        except Exception as exc:
            # Even an exception can occur AFTER an external side effect.
            if capability.effect == "read":
                status, reason = Outcome.FAILED, "read handler failed"
            else:
                status, reason = Outcome.INDETERMINATE, "handler failed; effect may have occurred"
            value = None
            reason_code = (
                CommandRejected.UNEXPECTED_ERROR
                if capability.effect == "read"
                else CommandRejected.INDETERMINATE_OUTCOME
            )
            logger.exception(
                "Handler raised command_id=%s effect=%s",
                command.command_id,
                capability.effect,
            )

        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute(
                    """UPDATE command_journal SET status=?,updated=?
                       WHERE principal=? AND resource=? AND idempotency=?""",
                    (status.value, time(), *key),
                )
                self._db.commit()
            except BaseException:
                self._db.rollback()
                # Never advertise success when durable outcome recording failed.
                return Result(
                    command.command_id,
                    Outcome.INDETERMINATE,
                    "durable outcome update failed",
                    reason_code=CommandRejected.DISPATCH_STATE_ERROR,
                )

        if status in (Outcome.SUCCEEDED, Outcome.COMMITTED):
            return Result(command.command_id, status, result=value)
        return Result(
            command.command_id,
            status,
            reason,
            reason_code=reason_code,
        )

    # -- Administrative reconciliation -------------------------------------

    def reconcile(
        self,
        principal: str,
        resource: str,
        key: str,
        *,
        observed_outcome: Outcome,
        authoritative_evidence: bool = False,
    ) -> bool:
        """Administrative reconciliation, gated by a separate trusted caller.

        This method MUST NOT be exposed as an untrusted command capability.
        It cannot turn an uncertain outcome into success without external evidence.
        """
        check = getattr(self.authority, "authorize_reconciliation", None)
        if (
            not authoritative_evidence
            or not callable(check)
            or not check(principal, resource, key, observed_outcome)
            or observed_outcome
            not in {Outcome.SUCCEEDED, Outcome.COMMITTED, Outcome.FAILED}
        ):
            raise CommandRejected(
                "trusted reconciliation authorization and evidence required",
                reason_code=CommandRejected.RECONCILIATION_DENIED,
                stage="reconciliation",
                resource=resource,
                principal_id=principal,
            )

        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cur = self._db.execute(
                    """UPDATE command_journal SET status=?,updated=?
                       WHERE principal=? AND resource=? AND idempotency=?
                       AND status IN (?,?)""",
                    (
                        observed_outcome.value,
                        time(),
                        principal,
                        resource,
                        key,
                        Outcome.IN_PROGRESS.value,
                        Outcome.INDETERMINATE.value,
                    ),
                )
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise

        updated = bool(cur.rowcount)
        logger.info(
            "Reconciliation principal=%s resource=%s key=%s outcome=%s updated=%s",
            principal,
            resource,
            key,
            observed_outcome.value,
            updated,
        )
        return updated

    # -- Inspection --------------------------------------------------------

    def inspect(self, principal: str, resource: str, key: str) -> Optional[Outcome]:
        with self._lock:
            try:
                row = self._db.execute(
                    """SELECT status FROM command_journal
                       WHERE principal=? AND resource=? AND idempotency=?""",
                    (principal, resource, key),
                ).fetchone()
            except sqlite3.Error:
                logger.exception(
                    "Journal inspect failed principal=%s resource=%s",
                    principal,
                    resource,
                )
                return None
        return Outcome(row[0]) if row else None


__all__ = [
    "Authority",
    "Capability",
    "CapabilityHandler",
    "Command",
    "CommandGateway",
    "Decision",
    "Outcome",
    "Result",
]