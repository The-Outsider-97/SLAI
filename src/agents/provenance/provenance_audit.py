"""Tamper-evident, append-only audit ledger for the SLAI Provenance subsystem.

Why this exists
---------------
``ProvenanceStore`` keeps provenance *facts* (entities, relations, lineage, custody) as one JSON document that is rewritten on every commit. That is a good
query store, but it is not tamper-evident: an edited or truncated file reads back as valid. Applications that must show *who did what, when, to which
revision, on which evidence* (SPADRA's model history and its building-regulation findings and corrections are the first) need an ordered record in which
any later edit, deletion, reordering or truncation can be detected. ``AuditLedger`` is that record. It does not replace the store; it sits beside it, and the
Provenance Agent can mirror each event into the store's entity/relation graph (see ``ProvenanceAgent.record_audit_event``).

What it guarantees, and what it does not
----------------------------------------
* **Append-only file, hash chain.** Every record carries the SHA-256 of the previous record (``prev``) and its own hash over a canonical encoding of every
  other field. ``verify()`` recomputes the chain and reports a changed record, a removed or reordered record, a repeated sequence number, and a torn last line
  (a crash in the middle of a write; ``recover()`` removes only that incomplete line).
* **Idempotent.** ``append`` with an ``event_id`` that is already recorded returns the existing record when the content is identical and raises
  ``ProvenanceConflictError`` when it is not, so a retry or a replayed event never doubles a record and never rewrites history.
* **Durable.** A record is written with a single ``write`` to a file opened for appending, flushed and ``fsync``-ed before ``append`` returns.
* **Integrity is not identity.** A hash chain proves the file is internally consistent and unchanged *since it was written*; it does not prove who wrote it, and
  anyone who can rewrite the whole file can rebuild a consistent chain. Authenticity is claimed only when the ledger was opened with ``hmac_key``: each record
  then carries an HMAC-SHA256 tag and ``verify`` reports ``authenticity == "hmac-verified"``; without a key it reports ``"unsigned"``. The key is never stored.
  Anchoring the chain head elsewhere (``head()``, e.g. in a saved project file) detects a wholesale replacement of the file.
* **Minimal data.** Payloads are canonicalised JSON of bounded size. Keys that name secrets, private reasoning or restricted source text are refused, so the
  ledger cannot be used to leak them; ``export(redact=...)`` removes further keys for a permitted export without breaking verification of the original.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from .utils.provenance_errors import ProvenanceConflictError, ProvenanceStorageError, ProvenanceValidationError
from .utils.provenance_helpers import canonical_json_dumps, canonicalize_value

__all__ = ["AuditLedger", "GENESIS", "AUDIT_SCHEMA"]

AUDIT_SCHEMA = "slai.provenance.audit.v1"
GENESIS = "0" * 64
MAX_PAYLOAD_BYTES = 64 * 1024
#: payload keys that are refused outright (compared case-insensitively, at any depth)
FORBIDDEN_KEYS = frozenset({"password", "passwd", "secret", "token", "access_token", "refresh_token", "api_key", "apikey", "authorization", "cookie", "private_key",
                            "reasoning", "chain_of_thought", "scratchpad", "source_text", "restricted_text", "licensed_text", "full_text"})
_FIELDS = ("schema", "seq", "event_id", "ts", "kind", "actor", "subject", "revision", "causes", "payload", "prev")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _scan(value: Any, depth: int = 0) -> None:
    if depth > 24:
        raise ProvenanceValidationError("audit payload is nested too deeply")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise ProvenanceValidationError("audit payloads must not carry secrets, private reasoning or restricted source text", context={"key": str(key)})
            _scan(item, depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _scan(item, depth + 1)


class AuditLedger:
    """One append-only JSON-lines file. Safe for several threads of one process; one writer process per file is assumed (the file is opened for append, and a
    second process appending would still produce a chain that ``verify`` reports honestly)."""

    _guard = threading.RLock()
    _locks: Dict[str, threading.RLock] = {}

    def __init__(self, path: str | Path, *, hmac_key: Optional[bytes] = None) -> None:
        self.path = Path(path).expanduser().resolve()
        self._hmac_key = bytes(hmac_key) if hmac_key else None
        with AuditLedger._guard:
            self._lock = AuditLedger._locks.setdefault(str(self.path), threading.RLock())
        self._index: Dict[str, Dict[str, Any]] = {}
        self._seq = 0
        self._head = GENESIS
        self._loaded_size = -1
        self._torn_tail = False
        with self._lock:
            self._load()

    # ------------------------------------------------------------------------------------------------ encoding
    def _hash(self, record: Mapping[str, Any]) -> str:
        return _sha(canonical_json_dumps({k: record[k] for k in _FIELDS}))

    def _tag(self, digest: str) -> Optional[str]:
        return hmac.new(self._hmac_key, digest.encode("ascii"), hashlib.sha256).hexdigest() if self._hmac_key else None

    # ------------------------------------------------------------------------------------------------ loading
    def _read_lines(self) -> List[str]:
        if not self.path.exists():
            return []
        try:
            raw = self.path.read_bytes()
        except OSError as exc:
            raise ProvenanceStorageError("audit ledger could not be read", context={"path": str(self.path)}, cause=exc) from exc
        text = raw.decode("utf-8", errors="replace")
        self._torn_tail = bool(text) and not text.endswith("\n")
        return text.split("\n")[:-1] if text.endswith("\n") else text.split("\n")

    def _load(self) -> None:
        self._index.clear()
        self._seq, self._head, self._torn_tail = 0, GENESIS, False
        for line in self._read_lines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                self._index[rec["event_id"]] = rec
                self._seq, self._head = int(rec["seq"]), str(rec["hash"])
            except (ValueError, KeyError, TypeError):
                continue                                           # reported by verify(); loading stays tolerant so the damage can be inspected
        self._loaded_size = self.path.stat().st_size if self.path.exists() else 0

    def _refresh(self) -> None:
        size = self.path.stat().st_size if self.path.exists() else 0
        if size != self._loaded_size:
            self._load()

    # ------------------------------------------------------------------------------------------------ writing
    def append(self, *, event_id: str, kind: str, actor: Mapping[str, Any], subject: Optional[Mapping[str, Any]] = None, revision: Optional[Mapping[str, Any]] = None,
               causes: Optional[Sequence[str]] = None, payload: Optional[Mapping[str, Any]] = None, ts: Optional[float] = None) -> Dict[str, Any]:
        """Record one event. ``actor`` is ``{"id": ..., "type": "human"|"ai"|"system"}``; ``subject`` the affected objects; ``revision`` the before/after
        revisions; ``causes`` the event ids this event follows from (a correction's approval, a finding's evaluation)."""
        event_id = str(event_id or "").strip()
        if not event_id or len(event_id) > 200:
            raise ProvenanceValidationError("an audit event needs a stable event_id of at most 200 characters")
        if not str(kind or "").strip():
            raise ProvenanceValidationError("an audit event needs a kind")
        if not isinstance(actor, Mapping) or not actor.get("id") or actor.get("type") not in {"human", "ai", "system"}:
            raise ProvenanceValidationError("actor must be {'id': ..., 'type': 'human'|'ai'|'system'}")
        body = canonicalize_value({"kind": str(kind), "actor": dict(actor), "subject": dict(subject or {}), "revision": dict(revision or {}),
                                   "causes": [str(c) for c in (causes or ())], "payload": dict(payload or {})})
        _scan(body)
        if len(canonical_json_dumps(body["payload"]).encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise ProvenanceValidationError("audit payload is too large; store a reference and a digest, not the content", context={"limit": MAX_PAYLOAD_BYTES})
        fingerprint = _sha(canonical_json_dumps(body))
        with self._lock:
            self._refresh()
            existing = self._index.get(event_id)
            if existing is not None:
                if existing.get("fingerprint") != fingerprint:
                    raise ProvenanceConflictError("event_id was reused with different content", context={"event_id": event_id})
                return dict(existing)
            rec: Dict[str, Any] = {"schema": AUDIT_SCHEMA, "seq": self._seq + 1, "event_id": event_id, "ts": float(ts if ts is not None else time.time()), "kind": body["kind"],
                                   "actor": body["actor"], "subject": body["subject"], "revision": body["revision"], "causes": body["causes"], "payload": body["payload"],
                                   "prev": self._head}
            rec["hash"] = self._hash(rec)
            rec["fingerprint"] = fingerprint
            tag = self._tag(rec["hash"])
            if tag:
                rec["sig"] = tag
            line = json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self._torn_tail:
                    raise ProvenanceStorageError("the ledger ends in an incomplete record; call recover() before appending", context={"path": str(self.path)})
                fd = os.open(str(self.path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                try:
                    os.write(fd, line.encode("utf-8"))
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except OSError as exc:
                raise ProvenanceStorageError("audit record could not be written durably", context={"path": str(self.path)}, cause=exc) from exc
            self._index[event_id] = rec
            self._seq, self._head = rec["seq"], rec["hash"]
            self._loaded_size = self.path.stat().st_size
            return dict(rec)

    def writable(self) -> bool:
        """True when a record could be appended now (the folder is writable and the file does not end in a torn record)."""
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._refresh()
                if self._torn_tail:
                    return False
                fd = os.open(str(self.path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                os.close(fd)
                return True
            except OSError:
                return False

    def recover(self) -> Dict[str, Any]:
        """Remove only a torn (incomplete) last line left by a crash. Complete records are never touched."""
        with self._lock:
            lines = self.path.read_bytes().split(b"\n") if self.path.exists() else []
            if not lines or lines[-1] == b"":
                return {"removed_bytes": 0}
            removed = len(lines[-1])
            with open(self.path, "wb") as handle:
                handle.write(b"\n".join(lines[:-1]) + (b"\n" if len(lines) > 1 else b""))
                handle.flush()
                os.fsync(handle.fileno())
            self._load()
            return {"removed_bytes": removed}

    # ------------------------------------------------------------------------------------------------ reading
    def head(self) -> Dict[str, Any]:
        with self._lock:
            self._refresh()
            return {"seq": self._seq, "hash": self._head}

    def records(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._refresh()
            return sorted((dict(r) for r in self._index.values()), key=lambda r: int(r.get("seq", 0)))

    def query(self, *, kind: Optional[str] = None, subject_id: Optional[str] = None, actor_type: Optional[str] = None, after_seq: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
        out = []
        for rec in self.records():
            if int(rec["seq"]) <= after_seq:
                continue
            if kind and not str(rec["kind"]).startswith(kind):
                continue
            if actor_type and rec["actor"].get("type") != actor_type:
                continue
            if subject_id and subject_id not in canonical_json_dumps(rec["subject"]):
                continue
            out.append(rec)
            if len(out) >= limit:
                break
        return out

    def verify(self) -> Dict[str, Any]:
        """Recompute the whole chain from the file. Returns ``ok`` and every problem found; nothing is repaired."""
        with self._lock:
            problems: List[Dict[str, Any]] = []
            prev, expected, count, seen = GENESIS, 1, 0, set()
            lines = self._read_lines()
            if self._torn_tail:
                problems.append({"problem": "torn_tail", "detail": "the last record is incomplete (a crash while writing); recover() removes it"})
                lines = lines[:-1]
            for n, line in enumerate(lines, 1):
                if not line.strip():
                    problems.append({"problem": "blank_line", "line": n})
                    continue
                try:
                    rec = json.loads(line)
                    body = {k: rec[k] for k in _FIELDS}
                except (ValueError, KeyError, TypeError):
                    problems.append({"problem": "unreadable_record", "line": n})
                    continue
                count += 1
                if rec.get("event_id") in seen:
                    problems.append({"problem": "duplicate_event_id", "line": n, "event_id": rec.get("event_id")})
                seen.add(rec.get("event_id"))
                if int(rec["seq"]) != expected:
                    problems.append({"problem": "sequence_gap_or_reorder", "line": n, "expected": expected, "found": rec["seq"]})
                if rec["prev"] != prev:
                    problems.append({"problem": "chain_broken", "line": n, "detail": "this record does not follow the one before it (a record was removed, inserted or reordered)"})
                if self._hash(body) != rec.get("hash"):
                    problems.append({"problem": "record_changed", "line": n, "event_id": rec.get("event_id")})
                if self._hmac_key is not None and not hmac.compare_digest(str(rec.get("sig", "")), self._tag(str(rec.get("hash"))) or ""):
                    problems.append({"problem": "signature_invalid", "line": n, "event_id": rec.get("event_id")})
                prev, expected = str(rec.get("hash")), int(rec["seq"]) + 1
            return {"ok": not problems, "records": count, "head": {"seq": expected - 1, "hash": prev}, "problems": problems,
                    "authenticity": "hmac-verified" if self._hmac_key is not None and not problems else ("hmac-failed" if self._hmac_key is not None else "unsigned"),
                    "note": "The chain shows the file is consistent and unchanged since it was written. It does not show who wrote it unless the ledger was opened with its HMAC key."}

    def verify_anchor(self, anchor: Mapping[str, Any]) -> Dict[str, Any]:
        """Compare the ledger with a chain head kept somewhere else (for example in a saved project): a shorter ledger, or one whose record at that sequence differs,
        means history was cut or replaced after the anchor was taken. A longer ledger is normal (work since the save)."""
        seq, digest = int(anchor.get("seq", 0)), str(anchor.get("hash", GENESIS))
        if seq == 0:
            return {"ok": True, "detail": "anchor is the empty ledger"}
        for rec in self.records():
            if int(rec["seq"]) == seq:
                same = rec["hash"] == digest
                return {"ok": same, "detail": "anchor matches" if same else "the record at the anchored sequence differs: history was replaced"}
        return {"ok": False, "detail": "the ledger is shorter than the anchor: records were removed"}

    def export(self, *, redact: Iterable[str] = ()) -> Dict[str, Any]:
        """The records for a permitted export. ``redact`` names payload keys to remove; the export says so, and carries the ledger head so the full ledger can still
        be verified by whoever holds it. A redacted export is not itself verifiable."""
        hide = {str(k).lower() for k in redact}

        def strip(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {k: ("[redacted]" if str(k).lower() in hide else strip(v)) for k, v in value.items()}
            if isinstance(value, list):
                return [strip(v) for v in value]
            return value
        records = [strip({k: v for k, v in r.items() if k not in ("fingerprint",)}) for r in self.records()]
        return {"schema": AUDIT_SCHEMA, "head": self.head(), "redacted_keys": sorted(hide), "records": records}

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.records())
