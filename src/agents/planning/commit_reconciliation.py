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

from .project_adapter import AuthoritativeProjectAdapter, HostCommitReceipt, HostCommitRequest
from .workflow_runtime import AuthorizationContext


class CommitReconciliationError(ValueError):
    """An invalid, conflicting, or unverifiable commit transition."""


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
            self._db.execute("CREATE INDEX IF NOT EXISTS ix_commit_intents_project "
                             "ON commit_intents(project_id, status)")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def get(self, idempotency_key: str) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute("SELECT * FROM commit_intents WHERE idempotency_key=?",
                                   (idempotency_key,)).fetchone()
            if row is None:
                raise CommitReconciliationError("unknown commit intent")
            return {"idempotency_key": row["idempotency_key"], "project_id": row["project_id"],
                    "status": row["status"], "request": json.loads(row["payload"]),
                    "receipt": json.loads(row["receipt"]) if row["receipt"] else None,
                    "event_timestamp": row["event_timestamp"]}

    def prepare(self, request: HostCommitRequest, *, auth: AuthorizationContext) -> dict[str, Any]:
        auth.require(request.project_id, "workflow:commit")
        encoded = json.dumps(asdict(request), sort_keys=True)
        with self._lock, self._db:
            self._db.execute("INSERT OR IGNORE INTO commit_intents "
                             "(idempotency_key,project_id,status,payload,event_timestamp) VALUES (?,?,?,?,?)",
                             (request.idempotency_key, request.project_id, "pending", encoded,
                              datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")))
            current = self.get(request.idempotency_key)
            if current["request"] != asdict(request):
                raise CommitReconciliationError("idempotency key reused for a different commit request")
            return current

    def _transition(self, key: str, expected: str, status: str,
                    receipt: Optional[HostCommitReceipt] = None) -> bool:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._db.execute("UPDATE commit_intents SET status=?,receipt=?,event_timestamp=? "
                                          "WHERE idempotency_key=? AND status=?",
                                          (status, json.dumps(asdict(receipt), sort_keys=True) if receipt else None,
                                           datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                                           key, expected))
                self._db.commit()
                return cursor.rowcount == 1
            except Exception:
                self._db.rollback()
                raise

    @staticmethod
    def _validate_receipt(request: HostCommitRequest, receipt: HostCommitReceipt) -> None:
        if not isinstance(receipt, HostCommitReceipt):
            raise CommitReconciliationError("host returned an invalid receipt type")
        if ((receipt.project_id, receipt.artifact_id, receipt.idempotency_key) !=
                (request.project_id, request.artifact_id, request.idempotency_key)):
            raise CommitReconciliationError("host receipt does not match pending commit intent")

    def _record_receipt(self, request: HostCommitRequest, receipt: HostCommitReceipt) -> dict[str, Any]:
        self._validate_receipt(request, receipt)
        self._transition(request.idempotency_key, "outcome_unknown", receipt.status, receipt)
        return self.get(request.idempotency_key)

    def dispatch(self, key: str, *, auth: AuthorizationContext) -> dict[str, Any]:
        """Send a prepared intent at most once from this outbox.

        A crash after the intent is marked uncertain must trigger lookup,
        even if the host was never called. This sacrifices liveness for safety.
        """
        current = self.get(key)
        auth.require(current["project_id"], "workflow:commit")
        if not self._transition(key, "pending", "outcome_unknown"):
            return self.get(key)
        request = HostCommitRequest(**current["request"])
        try:
            receipt = self.host.commit(request)
            return self._record_receipt(request, receipt)
        except Exception:
            # Timeouts or malformed responses might occur after the external
            # transaction committed. Do not treat them as proof of failure.
            return self.get(key)

    def reconcile(self, key: str, *, auth: AuthorizationContext) -> dict[str, Any]:
        current = self.get(key)
        auth.require(current["project_id"], "workflow:commit:reconcile")
        if current["status"] != "outcome_unknown":
            return current
        request = HostCommitRequest(**current["request"])
        try:
            receipt = self.host.lookup_commit(request.project_id, key)
        except Exception:
            return current
        if receipt is None:
            return current
        try:
            return self._record_receipt(request, receipt)
        except CommitReconciliationError:
            # Conflicting or malformed receipts must not be accepted.
            return self.get(key)

    def unresolved(self, project_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT idempotency_key FROM commit_intents "
                                    "WHERE project_id=? AND status='outcome_unknown'",
                                    (project_id,)).fetchall()
            return [self.get(row["idempotency_key"]) for row in rows]

    def publish_confirmed(self, key: str, provenance_agent: Any, *, auth: AuthorizationContext) -> bool:
        """Publish only a confirmed host receipt to the idempotent provenance API."""
        current = self.get(key)
        auth.require(current["project_id"], "workflow:provenance:write")
        if current["status"] != "confirmed" or current["receipt"] is None:
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
                                                   "metadata": {"commit_request": current["request"],
                                                                "host_evidence_id": evidence["evidence_id"]}}})
        return True


__all__ = ["CommitReconciliationStore", "CommitReconciliationError"]
