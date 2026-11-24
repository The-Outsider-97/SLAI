"""Data subject rights workflows: access (Art. 15), rectification (16), portability (20) (GDPR Art. 12).

* Every request needs a recorded identity verification bound to tenant and subject.
* Statutory deadline: one month (default 30 days), extendable by two further months only with a recorded reason
  and subject notification (Art. 12(3)).
* Data comes from registered providers (one per store). A provider failure never yields a partial answer
  presented as complete: the request stays ``incomplete`` and nothing is released.
* Released bundles are returned to the caller once; only a hash and metadata are retained (no copy of the data).
* Tenant and subject isolation is enforced here; providers are always called with both.
Erasure, restriction and objection live in :class:`PrivacyLedger`. Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import time
from threading import RLock
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol

from .utils.store import open_db

KINDS = ("access", "rectification", "portability")
DAY = 86400.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rights_requests(request_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, subject_id TEXT NOT NULL,
  kind TEXT NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL, due_at REAL NOT NULL, extended INTEGER NOT NULL DEFAULT 0,
  extension_reason TEXT, verified_by TEXT NOT NULL, method TEXT NOT NULL, completed_at REAL, result_sha256 TEXT, detail TEXT NOT NULL DEFAULT '');
"""


class RightsError(Exception):
    """Invalid or unauthorised rights request."""


class RightsProvider(Protocol):
    name: str
    portable: bool  # True only for data the subject provided, processed on consent/contract, by automated means (Art. 20(1))

    def export(self, tenant_id: str, subject_id: str) -> Mapping[str, Any]: ...


class SubjectRights:
    def __init__(self, path, *, response_days: float = 30.0, clock: Callable[[], float] = time.time,
                 processing_info: Optional[Callable[[str], Mapping[str, Any]]] = None) -> None:
        self._db = open_db(path)
        self._db.executescript(_SCHEMA)
        self._days, self._clock, self._info = response_days, clock, processing_info
        self._providers: Dict[str, RightsProvider] = {}
        self._lock = RLock()

    def provider_names(self) -> List[str]:
        return sorted(self._providers)

    def register_provider(self, provider: RightsProvider) -> None:
        if not getattr(provider, "name", None) or not callable(getattr(provider, "export", None)):
            raise RightsError("provider needs a name and export()")
        self._providers[provider.name] = provider

    # -------------------------------------------------------------- intake
    def submit(self, request_id: str, tenant_id: str, subject_id: str, kind: str, *, verification: Mapping[str, Any]) -> float:
        if kind not in KINDS:
            raise RightsError(f"kind must be one of {KINDS}")
        for k, v in (("request_id", request_id), ("tenant_id", tenant_id), ("subject_id", subject_id)):
            if not isinstance(v, str) or not v.strip():
                raise RightsError(f"{k} is required")
        v = verification or {}
        if not (v.get("verified") is True and str(v.get("method") or "").strip() and str(v.get("verified_by") or "").strip()
                and v.get("tenant_id") == tenant_id and v.get("subject_id") == subject_id):
            raise RightsError("identity verification bound to this tenant and subject is required")
        now = self._clock()
        due = now + self._days * DAY
        self._db.execute("INSERT INTO rights_requests(request_id,tenant_id,subject_id,kind,status,created_at,due_at,verified_by,method) VALUES(?,?,?,?,?,?,?,?,?)",
                         (request_id, tenant_id, subject_id, kind, "open", now, due, v["verified_by"], v["method"]))
        return due

    def extend(self, request_id: str, *, reason: str, subject_notified: bool) -> float:
        if not reason.strip() or not subject_notified:
            raise RightsError("an extension needs a reason and prior notification of the subject (Art. 12(3))")
        row = self._row(request_id)
        if row["extended"]:
            raise RightsError("already extended once; the statutory maximum is three months")
        due = row["due_at"] + 2 * self._days * DAY
        self._db.execute("UPDATE rights_requests SET due_at=?, extended=1, extension_reason=? WHERE request_id=?", (due, reason, request_id))
        return due

    def _row(self, request_id: str) -> Dict[str, Any]:
        cur = self._db.execute("SELECT request_id,tenant_id,subject_id,kind,status,created_at,due_at,extended,completed_at FROM rights_requests WHERE request_id=?", (request_id,))
        r = cur.fetchone()
        if not r:
            raise RightsError("unknown request")
        return dict(zip(("request_id", "tenant_id", "subject_id", "kind", "status", "created_at", "due_at", "extended", "completed_at"), r))

    def status(self, request_id: str) -> Dict[str, Any]:
        return self._row(request_id)

    def list_for_subject(self, tenant_id: str, subject_id: str) -> List[Dict[str, Any]]:
        return [{"request_id": a, "kind": b, "status": c, "due_at": d} for a, b, c, d in self._db.execute(
            "SELECT request_id,kind,status,due_at FROM rights_requests WHERE tenant_id=? AND subject_id=? ORDER BY created_at", (tenant_id, subject_id))]

    def overdue(self) -> List[str]:
        now = self._clock()
        return [r[0] for r in self._db.execute("SELECT request_id FROM rights_requests WHERE status!='completed' AND due_at<?", (now,))]

    # -------------------------------------------------------------- fulfilment
    def fulfil(self, request_id: str) -> Dict[str, Any]:
        row = self._row(request_id)
        if row["kind"] == "rectification":
            raise RightsError("use rectify() for rectification requests")
        if row["status"] == "completed":
            raise RightsError("already fulfilled; the bundle is not retained and cannot be re-released")
        tenant, subject, portable = row["tenant_id"], row["subject_id"], row["kind"] == "portability"
        sections: Dict[str, Any] = {}
        failures: List[str] = []
        for name, provider in sorted(self._providers.items()):
            if portable and not getattr(provider, "portable", False):
                continue
            try:
                sections[name] = dict(provider.export(tenant, subject) or {})
            except Exception as exc:
                failures.append(f"{name}:{type(exc).__name__}")
        if failures or not self._providers:
            detail = ",".join(failures) or "no_providers_registered"
            self._db.execute("UPDATE rights_requests SET status='incomplete', detail=? WHERE request_id=?", (detail, request_id))
            raise RightsError(f"request cannot be completed: {detail}")  # nothing partial is released
        bundle: Dict[str, Any] = {"schema": "slai.subject-export.v1", "right": row["kind"], "tenant_id": tenant, "subject_id": subject,
                                  "generated_at": int(self._clock()), "sections": sections}
        if row["kind"] == "access" and self._info:
            bundle["processing_information"] = dict(self._info(tenant))  # Art. 15(1)(a)-(h): purposes, recipients, retention, rights
        bundle["format"] = "application/json" if portable else "application/json; informational"
        digest = hashlib.sha256(json.dumps(bundle, sort_keys=True, default=str).encode()).hexdigest()
        self._db.execute("UPDATE rights_requests SET status='completed', completed_at=?, result_sha256=?, detail='' WHERE request_id=?", (self._clock(), digest, request_id))
        return {"bundle": bundle, "sha256": digest}

    def rectify(self, request_id: str, changes: Mapping[str, Any], *, apply: Callable[[str, str, Mapping[str, Any]], Mapping[str, Any]]) -> Dict[str, Any]:
        """Apply a rectification through the caller's store-specific ``apply(tenant, subject, changes)``; only hashes are recorded."""
        row = self._row(request_id)
        if row["kind"] != "rectification":
            raise RightsError("not a rectification request")
        if not changes:
            raise RightsError("no changes supplied")
        try:
            result = dict(apply(row["tenant_id"], row["subject_id"], dict(changes)) or {})
        except Exception as exc:
            self._db.execute("UPDATE rights_requests SET status='incomplete', detail=? WHERE request_id=?", (type(exc).__name__, request_id))
            raise RightsError("rectification failed; request remains open") from exc
        digest = hashlib.sha256(json.dumps({"fields": sorted(changes), "result": result}, sort_keys=True, default=str).encode()).hexdigest()
        self._db.execute("UPDATE rights_requests SET status='completed', completed_at=?, result_sha256=?, detail='' WHERE request_id=?", (self._clock(), digest, request_id))
        return {"fields": sorted(changes), "sha256": digest, "recipients_to_notify": "see Art. 19: notify recipients of the rectification"}

    def close(self) -> None:
        self._db.close()
