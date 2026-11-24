"""Tamper-evident regulatory evidence store.

Evidence is *metadata about an artefact* (type, producer, digest, minimised
summary), never the artefact's personal content. Every record is HMAC-signed and
hash-chained to its predecessor, so removal, reordering or editing is detectable
by ``verify_chain``. The producer is derived from a verified identity token, not
from a caller-supplied string, which is what makes "independent verification"
meaningful: the Governance Agent cannot mint evidence that counts as independent.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from .regulatory_context import derive_key
from .utils.identity import IdentityAuthority, IdentityVerificationError
from .utils.store import open_db

__all__ = ["EvidenceError", "EvidenceRecord", "EvidenceStore", "EVIDENCE_KEY_ENV", "EVIDENCE_TYPES"]

EVIDENCE_KEY_ENV = "SLAI_GOVERNANCE_EVIDENCE_KEY"

EVIDENCE_TYPES = frozenset(
    {
        "ropa", "lawful_basis", "consent", "dpia", "lia", "fria", "risk_register", "processor_record",
        "transfer_record", "ai_system_documentation", "dataset_documentation", "model_documentation",
        "human_oversight_record", "compliance_evaluation", "incident_record", "corrective_action",
        "release_conformity", "post_market_monitoring", "test_run", "deletion_verification",
        "isolation_test", "policy_integrity", "log_sanitisation_test", "privacy_notice", "dpa_contract",
        "training_eligibility", "model_evaluation", "ai_literacy", "dpo_appointment",
    }
)

_GENESIS = "0" * 64
_MAX_STR = 256


class EvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class EvidenceRecord:
    seq: int
    control_id: str
    evidence_type: str
    producer: str
    artifact_digest: str
    artifact_ref: str
    result: str  # "pass" | "fail" | "info"
    summary: Mapping[str, Any]
    created_at: float
    prev_hash: str
    entry_hash: str

    def age(self, now: float) -> float:
        return now - self.created_at


def _minimise(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Allow scalars and short scalar lists only; refuse anything that could carry a payload."""
    out: Dict[str, Any] = {}
    for key, value in summary.items():
        if isinstance(value, (bool, int, float)) or value is None:
            out[str(key)[:64]] = value
        elif isinstance(value, str):
            if len(value) > _MAX_STR:
                raise EvidenceError(f"summary field '{key}' too long; store a digest/ref instead")
            out[str(key)[:64]] = value
        elif isinstance(value, (list, tuple)) and len(value) <= 32 and all(isinstance(v, (str, int, float, bool)) for v in value):
            out[str(key)[:64]] = [v if not isinstance(v, str) else v[:_MAX_STR] for v in value]
        else:
            raise EvidenceError(f"summary field '{key}' has an unsupported type; evidence is metadata only")
    return out


class EvidenceStore:
    def __init__(
        self,
        authority: IdentityAuthority,
        *,
        path: Union[str, Path] = ":memory:",
        master_key: Optional[bytes] = None,
        env: Optional[Mapping[str, str]] = None,
    ):
        if master_key is None:
            raw = (env if env is not None else os.environ).get(EVIDENCE_KEY_ENV)
            master_key = raw.encode("utf-8") if raw else None
        self._key = derive_key(master_key, "evidence") if master_key and len(master_key) >= 32 else None
        self._authority = authority
        self._lock = RLock()
        self._db = open_db(path)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS evidence (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, control_id TEXT NOT NULL, evidence_type TEXT NOT NULL,
                producer TEXT NOT NULL, artifact_digest TEXT NOT NULL, artifact_ref TEXT NOT NULL,
                result TEXT NOT NULL, summary TEXT NOT NULL, created_at REAL NOT NULL,
                prev_hash TEXT NOT NULL, entry_hash TEXT NOT NULL, signature TEXT NOT NULL)"""
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS ev_control ON evidence(control_id)")

    @property
    def available(self) -> bool:
        return self._key is not None

    def append(
        self,
        *,
        producer_token: str,
        control_id: str,
        evidence_type: str,
        artifact_digest: str,
        artifact_ref: str,
        result: str,
        summary: Optional[Mapping[str, Any]] = None,
        now: Optional[float] = None,
    ) -> EvidenceRecord:
        if not self.available:
            raise EvidenceError("evidence signing key not configured")
        if evidence_type not in EVIDENCE_TYPES:
            raise EvidenceError(f"unknown evidence_type '{evidence_type}'")
        if result not in ("pass", "fail", "info"):
            raise EvidenceError("result must be pass|fail|info")
        if len(artifact_digest) != 64 or any(c not in "0123456789abcdef" for c in artifact_digest):
            raise EvidenceError("artifact_digest must be a lowercase SHA-256 hex digest")
        if not artifact_ref.strip():
            raise EvidenceError("artifact_ref required")
        try:
            producer = self._authority.verify(producer_token).principal_id
        except IdentityVerificationError as exc:
            raise EvidenceError(f"producer identity invalid: {exc.code}") from exc
        t = time.time() if now is None else now
        minimised = _minimise(summary or {})
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute("SELECT entry_hash FROM evidence ORDER BY seq DESC LIMIT 1").fetchone()
                prev = row[0] if row else _GENESIS
                body = json.dumps([control_id, evidence_type, producer, artifact_digest, artifact_ref, result, minimised, t, prev], sort_keys=True)
                entry_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
                sig = self._sign(entry_hash)
                cur = self._db.execute(
                    "INSERT INTO evidence(control_id, evidence_type, producer, artifact_digest, artifact_ref, result, summary, created_at, prev_hash, entry_hash, signature) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (control_id, evidence_type, producer, artifact_digest, artifact_ref, result, json.dumps(minimised, sort_keys=True), t, prev, entry_hash, sig),
                )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return EvidenceRecord(cur.lastrowid, control_id, evidence_type, producer, artifact_digest, artifact_ref, result, minimised, t, prev, entry_hash)

    def records(self, control_id: Optional[str] = None) -> List[EvidenceRecord]:
        q = "SELECT seq, control_id, evidence_type, producer, artifact_digest, artifact_ref, result, summary, created_at, prev_hash, entry_hash FROM evidence"
        args: Tuple[Any, ...] = ()
        if control_id:
            q += " WHERE control_id=?"
            args = (control_id,)
        rows = self._db.execute(q + " ORDER BY seq", args).fetchall()
        return [EvidenceRecord(r[0], r[1], r[2], r[3], r[4], r[5], r[6], json.loads(r[7]), r[8], r[9], r[10]) for r in rows]

    def verify_chain(self) -> Tuple[bool, Optional[int]]:
        """Recompute every hash and signature. Returns (ok, first_bad_seq)."""
        if not self.available:
            return False, None
        prev = _GENESIS
        rows = self._db.execute(
            "SELECT seq, control_id, evidence_type, producer, artifact_digest, artifact_ref, result, summary, created_at, prev_hash, entry_hash, signature FROM evidence ORDER BY seq"
        ).fetchall()
        for r in rows:
            body = json.dumps([r[1], r[2], r[3], r[4], r[5], r[6], json.loads(r[7]), r[8], r[9]], sort_keys=True)
            if r[9] != prev or hashlib.sha256(body.encode("utf-8")).hexdigest() != r[10] or not hmac.compare_digest(r[11], self._sign(r[10])):
                return False, r[0]
            prev = r[10]
        return True, None

    def _sign(self, entry_hash: str) -> str:
        assert self._key is not None
        return hmac.new(self._key, entry_hash.encode("ascii"), hashlib.sha256).hexdigest()
