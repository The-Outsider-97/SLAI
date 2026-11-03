"""Durable, authorization-gated commit correlation with authoritative hosts.

The outbox persists intent BEFORE invoking a host's idempotent transaction.
Uncertain outcomes are reconciled by lookup, never blindly re-submitted.
Neither this store nor the Planning Agent owns an application's document data.
"""

from __future__ import annotations

import json
import sqlite3
import threading

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .utils.planning_errors import *
from .project_adapter import AuthoritativeProjectAdapter, HostCommitReceipt, HostCommitRequest
from .workflow_runtime import AuthorizationContext
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Commit Reconciliation Store")
printer = PrettyPrinter()

class CommitReconciliationStore:
    """SQLite outbox with explicit outcome-unknown state and host reconciliation.

    The adapter must support idempotency and authoritative lookup. An adapter
    that cannot satisfy those guarantees is unsuitable for automated commits.
    """

    def __init__(self, path: str | Path, host: AuthoritativeProjectAdapter) -> None:
        if not isinstance(host, AuthoritativeProjectAdapter):
            raise TypeError("authoritative host must implement revision, commit and lookup methods")
        self.host = host
        self._lock = threading.RLock()
        selected = Path(path).expanduser() if str(path) != ":memory:" else None
        if selected is not None:
            selected.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(selected) if selected is not None else ":memory:",
                                   timeout=15, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA busy_timeout=15000")
        with self._db:
            self._db.execute("CREATE TABLE IF NOT EXISTS commit_intents ("
                             "idempotency_key TEXT PRIMARY KEY, project_id TEXT NOT NULL, "
                             "status TEXT NOT NULL, payload TEXT NOT NULL, receipt TEXT, event_timestamp TEXT)")
            if "event_timestamp" not in {row["name"] for row in
                                          self._db.execute("PRAGMA table_info(commit_intents)")}:
                self._db.execute("ALTER TABLE commit_intents ADD COLUMN event_timestamp TEXT")
            self._db.execute("UPDATE commit_intents SET event_timestamp=? WHERE event_timestamp IS NULL",
                             (datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),))
            self._db.execute("CREATE INDEX IF NOT EXISTS ix_commit_intents_project ON commit_intents(project_id, status)")

        logger.info("Commit reconciliation store initialized at %s", selected if selected is not None else ":memory:")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def get(self, idempotency_key: str) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute("SELECT * FROM commit_intents WHERE idempotency_key=?", (idempotency_key,)).fetchone()
            if row is None:
                raise CommitReconciliationError(
                    "unknown commit intent",
                    idempotency_key=idempotency_key,
                    reason="not_found",
                )
            return {"idempotency_key": row["idempotency_key"], "project_id": row["project_id"],
                    "status": row["status"], "request": json.loads(row["payload"]),
                    "receipt": json.loads(row["receipt"]) if row["receipt"] else None,
                    "event_timestamp": row["event_timestamp"]}

    def prepare(self, request: HostCommitRequest, *, auth: AuthorizationContext) -> dict[str, Any]:
        auth.require(request.project_id, "workflow:commit")
        encoded = json.dumps(asdict(request), sort_keys=True)
        with self._lock, self._db:
            cursor = self._db.execute(
                "INSERT OR IGNORE INTO commit_intents (idempotency_key,project_id,status,payload,event_timestamp) VALUES (?,?,?,?,?)",
                (request.idempotency_key, request.project_id, "pending", encoded,
                 datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")))
            if cursor.rowcount:
                logger.info("Prepared new commit intent key=%s project=%s artifact=%s",
                            request.idempotency_key, request.project_id, request.artifact_id)
            else:
                logger.debug("Commit intent already prepared key=%s", request.idempotency_key)
            current = self.get(request.idempotency_key)
            if current["request"] != asdict(request):
                logger.error("Idempotency key reused for a different request key=%s",
                             request.idempotency_key)
                raise CommitReconciliationError(
                    "idempotency key reused for a different commit request",
                    idempotency_key=request.idempotency_key,
                    project_id=request.project_id,
                    artifact_id=request.artifact_id,
                    reason="payload_mismatch",
                )
            return current
    
    def dispatch(self, key: str, *, auth: AuthorizationContext) -> dict[str, Any]:
        current = self.get(key)
        auth.require(current["project_id"], "workflow:commit")
        if not self._transition(key, "pending", "outcome_unknown"):
            logger.debug("Dispatch skipped key=%s status=%s", key, current["status"])
            return self.get(key)
        logger.info("Dispatching commit intent key=%s project=%s artifact=%s",
                    key, current["project_id"], current["request"]["artifact_id"])
        request = HostCommitRequest(**current["request"])
        try:
            receipt = self.host.commit(request)
            result = self._record_receipt(request, receipt)
            logger.info("Recorded host receipt key=%s status=%s revision=%s",
                        key, receipt.status, receipt.committed_revision)
            return result
        except Exception as exc:
            # Timeouts or malformed responses might occur after the external
            # transaction committed. Do not treat them as proof of failure.
            logger.warning("Host commit outcome unknown key=%s err=%s: %s",
                           key, type(exc).__name__, exc)
            return self.get(key)
    
    def reconcile(self, key: str, *, auth: AuthorizationContext) -> dict[str, Any]:
        current = self.get(key)
        auth.require(current["project_id"], "workflow:commit:reconcile")
        if current["status"] != "outcome_unknown":
            return current
        request = HostCommitRequest(**current["request"])
        try:
            receipt = self.host.lookup_commit(request.project_id, key)
        except Exception as exc:
            logger.warning("Host lookup failed during reconcile key=%s err=%s: %s",
                           key, type(exc).__name__, exc)
            return current
        if receipt is None:
            logger.debug("Reconcile found no host receipt yet key=%s", key)
            return current
        try:
            result = self._record_receipt(request, receipt)
            logger.info("Reconciled commit intent key=%s status=%s", key, receipt.status)
            return result
        except CommitReconciliationError as exc:
            logger.error("Rejected conflicting host receipt key=%s: %s", key, exc)
            return self.get(key)
    
    def unresolved(self, project_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT idempotency_key FROM commit_intents "
                "WHERE project_id=? AND status='outcome_unknown'", (project_id,)).fetchall()
            intents = [self.get(row["idempotency_key"]) for row in rows]
        if intents:
            logger.warning("Project %s has %d unresolved commit intent(s)", project_id, len(intents))
        return intents

    def publish_confirmed(self, key: str, provenance_agent: Any, *, auth: AuthorizationContext) -> bool:
        """Publish only a confirmed host receipt to the idempotent provenance API."""
        current = self.get(key)
        auth.require(current["project_id"], "workflow:provenance:write")
        if current["status"] != "confirmed" or current["receipt"] is None:
            logger.debug("Provenance publish skipped key=%s status=%s", key, current["status"])
            return False
        evidence = current["receipt"]
        # Stable event ID lets ProvenanceStore deduplicate replay after crash.
        provenance_agent.record_event({"event_type": "revision_event", "event_id": f"commit:{key}",
                                       "payload": {"event_id": f"commit:{key}",
                                                   "event_type": "commit.confirmed",
                                                   "project_id": current["project_id"],
                                                   "artifact_id": evidence["artifact_id"],
                                                   "revision_id": evidence["committed_revision"],
                                                   "previous_revision_id": current["request"]["expected_revision"],
                                                   "commit_id": evidence["evidence_id"],
                                                   "timestamp": current["event_timestamp"],
                                                   "metadata": {"commit_request": current["request"], "host_evidence_id": evidence["evidence_id"]}}})
        logger.info("Published confirmed commit key=%s to provenance", key)
        return True

    def _transition(self, key: str, expected: str, status: str,
                    receipt: Optional[HostCommitReceipt] = None) -> bool:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._db.execute("UPDATE commit_intents SET status=?,receipt=?,event_timestamp=? WHERE idempotency_key=? AND status=?",
                                          (status, json.dumps(asdict(receipt), sort_keys=True) if receipt else None,
                                           datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                                           key, expected))
                self._db.commit()
                return cursor.rowcount == 1
            except Exception:
                self._db.rollback()
                raise

    def _validate_receipt(self, request: HostCommitRequest, receipt: HostCommitReceipt) -> None:
        if not isinstance(receipt, HostCommitReceipt):
            raise CommitReconciliationError(
                "host returned an invalid receipt type",
                idempotency_key=request.idempotency_key,
                project_id=request.project_id,
                artifact_id=request.artifact_id,
                reason="invalid_receipt_type",
            )
        if ((receipt.project_id, receipt.artifact_id, receipt.idempotency_key) !=
                (request.project_id, request.artifact_id, request.idempotency_key)):
            raise CommitReconciliationError(
                "host receipt does not match pending commit intent",
                idempotency_key=request.idempotency_key,
                project_id=request.project_id,
                artifact_id=request.artifact_id,
                expected_revision=request.expected_revision,
                host_status=receipt.status,
                reason="receipt_mismatch",
            )

    def _record_receipt(self, request: HostCommitRequest, receipt: HostCommitReceipt) -> dict[str, Any]:
        self._validate_receipt(request, receipt)
        self._transition(request.idempotency_key, "outcome_unknown", receipt.status, receipt)
        return self.get(request.idempotency_key)


__all__ = ["CommitReconciliationStore", "CommitReconciliationError"]
