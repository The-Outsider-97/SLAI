"""Authoritative, durable, tenant-scoped privacy ledger (audit 4.2, 4.4; GDPR Art. 7, 17, 18, 19, 21).

One ledger replaces fragmented in-process privacy state. It stores, append-only and
hash-chained:
* consent grants / withdrawals per tenant, subject and purpose (Art. 7),
* restriction (Art. 18), objection (Art. 21) and erasure-pending state,
* legal holds (Art. 17(3)),
* erasure requests with recorded identity verification, per-target deletion evidence
  and an explicit ``completed`` status that is only reachable when EVERY registered target
  has independently verified absence (or is covered by a documented exemption),
* backup tombstones (salted hashes only) so restored backups can be re-erased,
* recipient notification obligations (Art. 19).

It implements ``ConsentProvider`` and ``RestrictionProvider`` through :class:`TenantView`.
Stdlib only. No personal content is stored: subject identifiers are kept as given by the
caller (pseudonymous IDs are expected) and evidence is hashes/metadata.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence

from .utils.store import open_db

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger_events(
  seq INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL, subject_id TEXT NOT NULL,
  kind TEXT NOT NULL, purpose TEXT NOT NULL DEFAULT '', payload TEXT NOT NULL, ts REAL NOT NULL,
  prev_hash TEXT NOT NULL, hash TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_events_subject ON ledger_events(tenant_id, subject_id, kind, purpose);
CREATE TABLE IF NOT EXISTS erasure_requests(
  request_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, subject_id TEXT NOT NULL, status TEXT NOT NULL,
  ground TEXT NOT NULL, verification TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
  completed_at REAL, attempts INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS erasure_targets(
  request_id TEXT NOT NULL, target TEXT NOT NULL, status TEXT NOT NULL, evidence TEXT NOT NULL DEFAULT '',
  detail TEXT NOT NULL DEFAULT '', updated_at REAL NOT NULL, PRIMARY KEY(request_id, target));
CREATE TABLE IF NOT EXISTS legal_holds(
  hold_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, subject_id TEXT NOT NULL, basis TEXT NOT NULL,
  placed_by TEXT NOT NULL, placed_at REAL NOT NULL, released_at REAL);
CREATE TABLE IF NOT EXISTS tombstones(
  tenant_id TEXT NOT NULL, subject_digest TEXT NOT NULL, request_id TEXT NOT NULL, created_at REAL NOT NULL,
  PRIMARY KEY(tenant_id, subject_digest));
CREATE TABLE IF NOT EXISTS recipient_notices(
  request_id TEXT NOT NULL, recipient TEXT NOT NULL, status TEXT NOT NULL, updated_at REAL NOT NULL,
  PRIMARY KEY(request_id, recipient));
"""

_GENESIS = "0" * 64
ERASURE_GROUNDS = frozenset({
    "art17_1_a_no_longer_necessary", "art17_1_b_consent_withdrawn", "art17_1_c_objection",
    "art17_1_d_unlawful_processing", "art17_1_e_legal_obligation", "art17_1_f_child_isss",
})


class LedgerError(Exception):
    """Invalid request, missing verification or integrity failure."""


class DeletionAdapter(Protocol):
    """A data store that holds subject data. Both methods MUST be real operations on that store."""

    name: str

    def delete(self, tenant_id: str, subject_id: str) -> Mapping[str, Any]: ...

    def verify_absent(self, tenant_id: str, subject_id: str) -> bool: ...


@dataclass(frozen=True)
class ErasureOutcome:
    request_id: str
    status: str  # completed | incomplete | held | rejected
    targets: Mapping[str, str]
    notes: tuple = ()


def _h(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


class PrivacyLedger:
    def __init__(self, path, *, tombstone_salt: bytes, clock: Callable[[], float] = time.time) -> None:
        if not tombstone_salt or len(tombstone_salt) < 16:
            raise LedgerError("tombstone_salt (>=16 bytes) required")
        self._db = open_db(path)
        self._db.executescript(_SCHEMA)
        self._salt = tombstone_salt
        self._clock = clock
        self._lock = RLock()
        self._adapters: Dict[str, DeletionAdapter] = {}
        self._recipients: Dict[str, List[str]] = {}

    # ------------------------------------------------------------ chain
    def _append(self, tenant: str, subject: str, kind: str, purpose: str, payload: Mapping[str, Any]) -> int:
        with self._lock:
            row = self._db.execute("SELECT hash FROM ledger_events ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row[0] if row else _GENESIS
            body = json.dumps(payload, sort_keys=True, default=str)
            ts = self._clock()
            digest = _h(prev, tenant, subject, kind, purpose, body, repr(ts))
            cur = self._db.execute(
                "INSERT INTO ledger_events(tenant_id,subject_id,kind,purpose,payload,ts,prev_hash,hash) VALUES(?,?,?,?,?,?,?,?)",
                (tenant, subject, kind, purpose, body, ts, prev, digest),
            )
            return int(cur.lastrowid)

    def verify_chain(self) -> bool:
        prev = _GENESIS
        for tenant, subject, kind, purpose, body, ts, p, h in self._db.execute(
            "SELECT tenant_id,subject_id,kind,purpose,payload,ts,prev_hash,hash FROM ledger_events ORDER BY seq"
        ):
            if p != prev or h != _h(prev, tenant, subject, kind, purpose, body, repr(ts)):
                return False
            prev = h
        return True

    @staticmethod
    def _need(**values: str) -> None:
        for k, v in values.items():
            if not isinstance(v, str) or not v.strip():
                raise LedgerError(f"{k} is required")

    # ------------------------------------------------------------ consent (Art. 7)
    def record_consent(self, tenant_id: str, subject_id: str, purpose: str, *, explicit: bool,
                       notice_version: str, evidence_ref: str, parental_authorisation_ref: Optional[str] = None) -> int:
        self._need(tenant_id=tenant_id, subject_id=subject_id, purpose=purpose, notice_version=notice_version, evidence_ref=evidence_ref)
        return self._append(tenant_id, subject_id, "consent_granted", purpose, {
            "explicit": bool(explicit), "notice_version": notice_version, "evidence_ref": evidence_ref,
            "parental_authorisation_ref": parental_authorisation_ref})

    def withdraw_consent(self, tenant_id: str, subject_id: str, purpose: str, *, evidence_ref: str) -> int:
        self._need(tenant_id=tenant_id, subject_id=subject_id, purpose=purpose, evidence_ref=evidence_ref)
        return self._append(tenant_id, subject_id, "consent_withdrawn", purpose, {"evidence_ref": evidence_ref})

    def consent_snapshot(self, tenant_id: str, subject_id: str, purpose: str) -> Dict[str, Any]:
        row = self._db.execute(
            "SELECT kind,payload FROM ledger_events WHERE tenant_id=? AND subject_id=? AND purpose=? "
            "AND kind IN ('consent_granted','consent_withdrawn') ORDER BY seq DESC LIMIT 1", (tenant_id, subject_id, purpose)).fetchone()
        if not row:
            return {"active": False, "withdrawn": False}
        payload = json.loads(row[1])
        if row[0] == "consent_withdrawn":
            return {"active": False, "withdrawn": True}
        return {"active": True, "withdrawn": False, "explicit": payload["explicit"],
                "parental_authorisation": bool(payload.get("parental_authorisation_ref")), "purposes": [purpose]}

    def list_consents(self, tenant_id: str, subject_id: str) -> List[Dict[str, Any]]:
        """Current consent state per purpose (latest event wins) - for subject-facing views."""
        latest: Dict[str, Dict[str, Any]] = {}
        for purpose, kind, payload, ts in self._db.execute(
                "SELECT purpose,kind,payload,ts FROM ledger_events WHERE tenant_id=? AND subject_id=? "
                "AND kind IN ('consent_granted','consent_withdrawn') ORDER BY seq", (tenant_id, subject_id)):
            p = json.loads(payload)
            latest[purpose] = {"purpose": purpose, "active": kind == "consent_granted", "since": ts,
                               "notice_version": p.get("notice_version"), "explicit": p.get("explicit")}
        return sorted(latest.values(), key=lambda r: r["purpose"])

    def erasure_requests(self, tenant_id: str, subject_id: str) -> List[Dict[str, Any]]:
        return [{"request_id": r, "status": s, "created_at": c, "completed_at": d} for r, s, c, d in self._db.execute(
            "SELECT request_id,status,created_at,completed_at FROM erasure_requests WHERE tenant_id=? AND subject_id=? ORDER BY created_at",
            (tenant_id, subject_id))]

    # ------------------------------------------------------------ restriction / objection (Art. 18, 21)
    def set_restriction(self, tenant_id: str, subject_id: str, restricted: bool, *, evidence_ref: str) -> int:
        self._need(tenant_id=tenant_id, subject_id=subject_id, evidence_ref=evidence_ref)
        return self._append(tenant_id, subject_id, "restriction", "", {"restricted": bool(restricted), "evidence_ref": evidence_ref})

    def set_objection(self, tenant_id: str, subject_id: str, purpose: str, objected: bool, *, evidence_ref: str) -> int:
        self._need(tenant_id=tenant_id, subject_id=subject_id, purpose=purpose, evidence_ref=evidence_ref)
        return self._append(tenant_id, subject_id, "objection", purpose, {"objected": bool(objected), "evidence_ref": evidence_ref})

    def subject_state(self, tenant_id: str, subject_id: str) -> Dict[str, Any]:
        r = self._db.execute("SELECT payload FROM ledger_events WHERE tenant_id=? AND subject_id=? AND kind='restriction' ORDER BY seq DESC LIMIT 1",
                             (tenant_id, subject_id)).fetchone()
        restricted = bool(json.loads(r[0])["restricted"]) if r else False
        objected = []
        latest: Dict[str, bool] = {}
        for purpose, payload in self._db.execute(
                "SELECT purpose,payload FROM ledger_events WHERE tenant_id=? AND subject_id=? AND kind='objection' ORDER BY seq", (tenant_id, subject_id)):
            latest[purpose] = bool(json.loads(payload)["objected"])
        objected = sorted(p for p, v in latest.items() if v)
        pending = self._db.execute(
            "SELECT 1 FROM erasure_requests WHERE tenant_id=? AND subject_id=? AND status IN ('open','incomplete','held') LIMIT 1",
            (tenant_id, subject_id)).fetchone() is not None
        return {"restricted": restricted, "objected_purposes": objected, "erasure_pending": pending}

    # ------------------------------------------------------------ legal holds (Art. 17(3))
    def place_hold(self, hold_id: str, tenant_id: str, subject_id: str, *, basis: str, placed_by: str) -> None:
        self._need(hold_id=hold_id, tenant_id=tenant_id, subject_id=subject_id, basis=basis, placed_by=placed_by)
        self._db.execute("INSERT INTO legal_holds(hold_id,tenant_id,subject_id,basis,placed_by,placed_at) VALUES(?,?,?,?,?,?)",
                         (hold_id, tenant_id, subject_id, basis, placed_by, self._clock()))
        self._append(tenant_id, subject_id, "legal_hold_placed", "", {"hold_id": hold_id, "basis": basis, "placed_by": placed_by})

    def release_hold(self, hold_id: str, *, released_by: str) -> None:
        self._need(hold_id=hold_id, released_by=released_by)
        row = self._db.execute("SELECT tenant_id,subject_id FROM legal_holds WHERE hold_id=? AND released_at IS NULL", (hold_id,)).fetchone()
        if not row:
            raise LedgerError("unknown or already released hold")
        self._db.execute("UPDATE legal_holds SET released_at=? WHERE hold_id=?", (self._clock(), hold_id))
        self._append(row[0], row[1], "legal_hold_released", "", {"hold_id": hold_id, "released_by": released_by})

    def active_holds(self, tenant_id: str, subject_id: str) -> List[str]:
        return [r[0] for r in self._db.execute(
            "SELECT hold_id FROM legal_holds WHERE tenant_id=? AND subject_id=? AND released_at IS NULL", (tenant_id, subject_id))]

    # ------------------------------------------------------------ deletion (Art. 17, 19)
    def adapter_names(self) -> List[str]:
        return sorted(self._adapters)

    def register_adapter(self, adapter: DeletionAdapter) -> None:
        if not getattr(adapter, "name", None) or not callable(getattr(adapter, "delete", None)) or not callable(getattr(adapter, "verify_absent", None)):
            raise LedgerError("adapter needs name, delete() and verify_absent()")
        self._adapters[adapter.name] = adapter

    def register_recipient(self, tenant_id: str, recipient: str) -> None:
        self._recipients.setdefault(tenant_id, [])
        if recipient not in self._recipients[tenant_id]:
            self._recipients[tenant_id].append(recipient)

    def _subject_digest(self, tenant_id: str, subject_id: str) -> str:
        return hmac.new(self._salt, f"{tenant_id}\x1f{subject_id}".encode(), hashlib.sha256).hexdigest()

    def request_erasure(self, request_id: str, tenant_id: str, subject_id: str, *, verification: Mapping[str, Any],
                        ground: str = "unassessed") -> str:
        """Open an erasure request. NO reason text is required (Art. 12(2)); verification is."""
        self._need(request_id=request_id, tenant_id=tenant_id, subject_id=subject_id)
        v = verification or {}
        if not (v.get("verified") is True and str(v.get("method") or "").strip() and str(v.get("verified_by") or "").strip()
                and str(v.get("subject_id") or "") == subject_id and str(v.get("tenant_id") or "") == tenant_id):
            raise LedgerError("identity verification (verified, method, verified_by, tenant_id, subject_id) is required")
        if ground != "unassessed" and ground not in ERASURE_GROUNDS:
            raise LedgerError("unknown Art. 17(1) ground")
        now = self._clock()
        self._db.execute("INSERT INTO erasure_requests(request_id,tenant_id,subject_id,status,ground,verification,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                         (request_id, tenant_id, subject_id, "open", ground, json.dumps({k: v[k] for k in ("method", "verified_by")}), now, now))
        self._append(tenant_id, subject_id, "erasure_requested", "", {"request_id": request_id, "ground": ground})
        return request_id

    def execute_erasure(self, request_id: str) -> ErasureOutcome:
        row = self._db.execute("SELECT tenant_id,subject_id,status FROM erasure_requests WHERE request_id=?", (request_id,)).fetchone()
        if not row:
            raise LedgerError("unknown erasure request")
        tenant, subject, status = row
        if status == "completed":
            return ErasureOutcome(request_id, "completed", self._targets(request_id), ("already completed",))
        if not self._adapters:
            raise LedgerError("no deletion adapters registered: completion cannot be verified")
        holds = self.active_holds(tenant, subject)
        if holds:
            self._set_status(request_id, "held")
            self._append(tenant, subject, "erasure_held", "", {"request_id": request_id, "holds": holds})
            return ErasureOutcome(request_id, "held", self._targets(request_id), (f"legal hold(s): {','.join(holds)}",))
        self._db.execute("UPDATE erasure_requests SET attempts=attempts+1 WHERE request_id=?", (request_id,))
        for name, adapter in sorted(self._adapters.items()):
            ok, evidence, detail = False, "", ""
            try:
                result = dict(adapter.delete(tenant, subject) or {})
                evidence = _h(name, json.dumps(result, sort_keys=True, default=str))
                ok = bool(adapter.verify_absent(tenant, subject))  # independent verification, not the delete() return value
                detail = "" if ok else "verify_absent returned False"
            except Exception as exc:  # failure must leave the request incomplete, never completed
                detail = f"{type(exc).__name__}"
            self._db.execute("INSERT OR REPLACE INTO erasure_targets(request_id,target,status,evidence,detail,updated_at) VALUES(?,?,?,?,?,?)",
                             (request_id, name, "verified" if ok else "failed", evidence, detail, self._clock()))
        targets = self._targets(request_id)
        if all(s == "verified" for s in targets.values()):
            self._set_status(request_id, "completed", completed=True)
            self._db.execute("INSERT OR IGNORE INTO tombstones(tenant_id,subject_digest,request_id,created_at) VALUES(?,?,?,?)",
                             (tenant, self._subject_digest(tenant, subject), request_id, self._clock()))
            for recipient in self._recipients.get(tenant, []):
                self._db.execute("INSERT OR IGNORE INTO recipient_notices(request_id,recipient,status,updated_at) VALUES(?,?,?,?)",
                                 (request_id, recipient, "pending", self._clock()))
            self._append(tenant, subject, "erasure_completed", "", {"request_id": request_id, "targets": sorted(targets)})
            return ErasureOutcome(request_id, "completed", targets)
        self._set_status(request_id, "incomplete")
        self._append(tenant, subject, "erasure_incomplete", "", {"request_id": request_id, "failed": sorted(t for t, s in targets.items() if s != "verified")})
        return ErasureOutcome(request_id, "incomplete", targets, ("retry required",))

    def _targets(self, request_id: str) -> Dict[str, str]:
        return {t: s for t, s in self._db.execute("SELECT target,status FROM erasure_targets WHERE request_id=?", (request_id,))}

    def _set_status(self, request_id: str, status: str, *, completed: bool = False) -> None:
        self._db.execute("UPDATE erasure_requests SET status=?, updated_at=?, completed_at=? WHERE request_id=?",
                         (status, self._clock(), self._clock() if completed else None, request_id))

    def pending_recipient_notices(self, request_id: str) -> List[str]:
        return [r[0] for r in self._db.execute("SELECT recipient FROM recipient_notices WHERE request_id=? AND status='pending'", (request_id,))]

    def confirm_recipient_notice(self, request_id: str, recipient: str) -> None:
        self._db.execute("UPDATE recipient_notices SET status='notified', updated_at=? WHERE request_id=? AND recipient=?", (self._clock(), request_id, recipient))

    # ------------------------------------------------------------ backup resurrection protection
    def must_reerase_after_restore(self, tenant_id: str, subject_id: str) -> bool:
        """After restoring a backup, call this per subject found in it: True means the subject was erased and must be re-erased."""
        return self._db.execute("SELECT 1 FROM tombstones WHERE tenant_id=? AND subject_digest=?",
                                (tenant_id, self._subject_digest(tenant_id, subject_id))).fetchone() is not None

    # ------------------------------------------------------------ provider views
    def view(self, tenant_id: str) -> "TenantView":
        return TenantView(self, tenant_id)

    def provider(self) -> "LedgerProvider":
        """Tenant-aware provider for the policy engine (tenant comes from each request's context)."""
        return LedgerProvider(self)

    def close(self) -> None:
        self._db.close()


class TenantView:
    """``ConsentProvider`` + ``RestrictionProvider`` bound to one tenant (no cross-tenant reads)."""

    def __init__(self, ledger: PrivacyLedger, tenant_id: str) -> None:
        self._l, self._t = ledger, tenant_id

    def consent_snapshot(self, subject_id: str, purpose: str) -> Mapping[str, Any]:
        return self._l.consent_snapshot(self._t, subject_id, purpose)

    def subject_state(self, subject_id: str) -> Mapping[str, Any]:
        return self._l.subject_state(self._t, subject_id)


class LedgerProvider:
    """Consent + restriction provider that requires an explicit tenant on every call."""

    tenant_aware = True

    def __init__(self, ledger: PrivacyLedger) -> None:
        self._l = ledger

    def consent_snapshot(self, subject_id: str, purpose: str, *, tenant_id: str) -> Mapping[str, Any]:
        return self._l.consent_snapshot(tenant_id, subject_id, purpose)

    def subject_state(self, subject_id: str, *, tenant_id: str) -> Mapping[str, Any]:
        return self._l.subject_state(tenant_id, subject_id)
