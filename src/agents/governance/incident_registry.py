"""Regulatory incident lifecycle: GDPR personal-data breaches and AI Act serious incidents.

Lifecycle: detection -> classification -> containment -> risk assessment ->
escalation -> notification decision -> remediation -> verification -> closure.

Notifications are *prepared, never transmitted*: ``prepare_notification`` creates
a draft and ``record_submission`` only records that a named, authorised human
submitted it externally (with the regulator's reference). The registry keeps the
Art. 33(5) record of every breach, notified or not, and tracks deadlines.

Deadline sources: GDPR Art. 33(1) 72h from awareness; Art. 33(2) processor-to-
controller "without undue delay"; Art. 34 data-subject communication "without
undue delay" when the risk is high; AI Act Art. 73: 15 days from awareness,
2 days for widespread infringement / serious disruption of critical
infrastructure, 10 days for death. Verify against the consolidated texts.
"""

from __future__ import annotations

import json
import time

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from .processing_context import new_id
from .utils.store import open_db

__all__ = ["IncidentError", "IncidentKind", "IncidentRegistry", "IncidentState", "BreachRisk"]

HOUR = 3600.0
DAY = 86400.0


class IncidentError(ValueError):
    pass


class IncidentKind(str, Enum):
    PERSONAL_DATA_BREACH = "personal_data_breach"
    AI_SERIOUS_INCIDENT = "ai_serious_incident"
    PROHIBITED_PRACTICE_ATTEMPT = "prohibited_practice_attempt"
    GOVERNANCE_FAILURE = "governance_failure"


class IncidentState(str, Enum):
    DETECTED = "detected"
    CLASSIFIED = "classified"
    CONTAINED = "contained"
    RISK_ASSESSED = "risk_assessed"
    ESCALATED = "escalated"
    NOTIFICATION_DECIDED = "notification_decided"
    REMEDIATED = "remediated"
    VERIFIED = "verified"
    CLOSED = "closed"


_ORDER = list(IncidentState)


class BreachRisk(str, Enum):
    UNLIKELY = "unlikely"  # Art. 33(1): no notification, but document
    RISK = "risk"  # notify supervisory authority
    HIGH = "high"  # notify authority AND data subjects (Art. 34), subject to exceptions


_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    incident_id TEXT PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL,
    record_json TEXT NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS incident_timeline (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, incident_id TEXT NOT NULL, event TEXT NOT NULL,
    actor TEXT NOT NULL, detail TEXT NOT NULL, at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS notifications (
    notification_id TEXT PRIMARY KEY, incident_id TEXT NOT NULL, audience TEXT NOT NULL,
    status TEXT NOT NULL, due_at REAL, record_json TEXT NOT NULL);
"""


class IncidentRegistry:
    def __init__(self, path: Union[str, Path] = ":memory:", *, authorised_roles=("dpo", "authorised_reporter")):
        self._lock = RLock()
        self._db = open_db(path)
        self._db.executescript(_SCHEMA)
        self.authorised_roles = frozenset(authorised_roles)

    # ------------------------------------------------------------------ lifecycle
    def detect(
        self,
        kind: IncidentKind,
        *,
        reporter: str,
        description: str,
        affected_systems: Tuple[str, ...] = (),
        affected_subject_count: Optional[int] = None,
        evidence_refs: Tuple[str, ...] = (),
        aware_at: Optional[float] = None,
        is_processor: bool = False,
        now: Optional[float] = None,
    ) -> str:
        t = time.time() if now is None else now
        if not description.strip() or not reporter:
            raise IncidentError("description and reporter required")
        iid = new_id("inc")
        rec = {
            "incident_id": iid, "kind": kind.value, "description": description[:512], "reporter": reporter,
            "affected_systems": list(affected_systems), "affected_subject_count": affected_subject_count,
            "evidence_refs": list(evidence_refs), "detected_at": t, "aware_at": aware_at if aware_at is not None else t,
            "is_processor": is_processor, "containment": [], "risk": None, "ai_flags": {}, "notifications": [],
            "corrective_actions": [],
        }
        with self._lock:
            self._db.execute("INSERT INTO incidents VALUES (?,?,?,?,?)", (iid, kind.value, IncidentState.DETECTED.value, json.dumps(rec), t))
            self._log(iid, "detected", reporter, {"kind": kind.value}, t)
        return iid

    def advance(self, incident_id: str, to: IncidentState, *, actor: str, detail: Optional[Mapping[str, Any]] = None, now: Optional[float] = None) -> None:
        t = time.time() if now is None else now
        with self._lock:
            state, rec = self._load(incident_id)
            if _ORDER.index(to) != _ORDER.index(state) + 1:
                raise IncidentError(f"illegal transition {state.value} -> {to.value}; steps must not be skipped")
            if to is IncidentState.CONTAINED and not rec["containment"]:
                raise IncidentError("record at least one containment action first")
            if to is IncidentState.RISK_ASSESSED and rec["risk"] is None:
                raise IncidentError("record a risk assessment first")
            if to is IncidentState.NOTIFICATION_DECIDED and not rec["notifications"] and self._notification_expected(rec):
                raise IncidentError("notification required: prepare and record notification before deciding")
            if to is IncidentState.CLOSED and any(a["status"] != "verified" for a in rec["corrective_actions"]):
                raise IncidentError("all corrective actions must be verified before closure")
            self._save(incident_id, to, rec, t)
            self._log(incident_id, f"state:{to.value}", actor, detail or {}, t)

    def record_containment(self, incident_id: str, action: str, *, actor: str, now: Optional[float] = None) -> None:
        t = time.time() if now is None else now
        with self._lock:
            state, rec = self._load(incident_id)
            rec["containment"].append({"action": action[:256], "by": actor, "at": t})
            self._save(incident_id, state, rec, t)
            self._log(incident_id, "containment", actor, {"action": action[:256]}, t)

    def assess_breach_risk(
        self, incident_id: str, risk: BreachRisk, rationale: str, *, actor: str,
        subjects_protected_by_encryption: bool = False, subsequent_measures_remove_high_risk: bool = False,
        now: Optional[float] = None,
    ) -> None:
        t = time.time() if now is None else now
        if len(rationale.strip()) < 15:
            raise IncidentError("risk rationale required")
        with self._lock:
            state, rec = self._load(incident_id)
            rec["risk"] = {
                "level": risk.value, "rationale": rationale[:512], "by": actor, "at": t,
                "art34_encryption_exception": subjects_protected_by_encryption,
                "art34_subsequent_measures_exception": subsequent_measures_remove_high_risk,
            }
            self._save(incident_id, state, rec, t)
            self._log(incident_id, "risk_assessed", actor, {"level": risk.value}, t)

    def record_ai_flags(self, incident_id: str, *, widespread_or_critical_infrastructure: bool = False, death: bool = False, actor: str, now: Optional[float] = None) -> None:
        t = time.time() if now is None else now
        with self._lock:
            state, rec = self._load(incident_id)
            rec["ai_flags"] = {"widespread_or_critical_infrastructure": widespread_or_critical_infrastructure, "death": death}
            self._save(incident_id, state, rec, t)

    # ------------------------------------------------------------------ deadlines & notifications
    def deadlines(self, incident_id: str) -> Dict[str, Any]:
        _, rec = self._load(incident_id)
        aware = rec["aware_at"]
        out: Dict[str, Any] = {}
        kind = IncidentKind(rec["kind"])
        if kind is IncidentKind.PERSONAL_DATA_BREACH:
            if rec["is_processor"]:
                out["controller_notification"] = {"due": "without_undue_delay", "ref": "GDPR Art. 33(2)"}
            else:
                risk = (rec["risk"] or {}).get("level")
                if risk in (None, BreachRisk.RISK.value, BreachRisk.HIGH.value):
                    out["supervisory_authority"] = {"due_at": aware + 72 * HOUR, "ref": "GDPR Art. 33(1)", "pending_risk_assessment": risk is None}
                if risk == BreachRisk.HIGH.value and not self._art34_exempt(rec):
                    out["data_subjects"] = {"due": "without_undue_delay", "ref": "GDPR Art. 34"}
        if kind is IncidentKind.AI_SERIOUS_INCIDENT:
            flags = rec["ai_flags"]
            days = 10 if flags.get("death") else 2 if flags.get("widespread_or_critical_infrastructure") else 15
            out["market_surveillance_authority"] = {"due_at": aware + days * DAY, "ref": "AI Act Art. 73", "days": days}
        return out

    def overdue(self, *, now: Optional[float] = None) -> List[Dict[str, Any]]:
        t = time.time() if now is None else now
        rows = self._db.execute("SELECT incident_id, state FROM incidents WHERE state!='closed'").fetchall()
        out = []
        for iid, state in rows:
            _, rec = self._load(iid)
            sent = {n["audience"] for n in rec["notifications"] if n["status"] == "submitted"}
            for audience, d in self.deadlines(iid).items():
                if "due_at" in d and d["due_at"] < t and audience not in sent:
                    out.append({"incident_id": iid, "audience": audience, "due_at": d["due_at"], "ref": d["ref"]})
        return out

    def prepare_notification(self, incident_id: str, audience: str, content: Mapping[str, Any], *, actor: str, now: Optional[float] = None) -> str:
        """Create a DRAFT. Nothing is sent; a human must submit it externally."""
        t = time.time() if now is None else now
        with self._lock:
            state, rec = self._load(incident_id)
            due = (self.deadlines(incident_id).get(audience) or {}).get("due_at")
            nid = new_id("notif")
            entry = {"notification_id": nid, "audience": audience, "status": "draft", "prepared_by": actor, "prepared_at": t, "due_at": due,
                     "content": {k: str(v)[:512] for k, v in content.items()}}
            rec["notifications"].append(entry)
            self._db.execute("INSERT INTO notifications VALUES (?,?,?,?,?,?)", (nid, incident_id, audience, "draft", due, json.dumps(entry)))
            self._save(incident_id, state, rec, t)
            self._log(incident_id, "notification_prepared", actor, {"audience": audience}, t)
            return nid

    def record_submission(self, incident_id: str, notification_id: str, submitter_roles: Tuple[str, ...], submitter: str, external_reference: str, *, now: Optional[float] = None) -> None:
        """Record that an *authorised human* submitted the draft externally. No transmission happens here."""
        t = time.time() if now is None else now
        if not (self.authorised_roles & set(submitter_roles)):
            raise IncidentError("submitter lacks organisational authority to notify regulators")
        if not external_reference.strip():
            raise IncidentError("external reference from the regulator's portal required")
        with self._lock:
            state, rec = self._load(incident_id)
            for n in rec["notifications"]:
                if n["notification_id"] == notification_id:
                    n.update(status="submitted", submitted_by=submitter, submitted_at=t, external_reference=external_reference)
                    break
            else:
                raise IncidentError("unknown notification")
            self._db.execute("UPDATE notifications SET status='submitted' WHERE notification_id=?", (notification_id,))
            self._save(incident_id, state, rec, t)
            self._log(incident_id, "notification_submitted", submitter, {"audience": n["audience"]}, t)

    def add_corrective_action(self, incident_id: str, action: str, *, owner: str, due_at: float, now: Optional[float] = None) -> int:
        t = time.time() if now is None else now
        with self._lock:
            state, rec = self._load(incident_id)
            rec["corrective_actions"].append({"action": action[:256], "owner": owner, "due_at": due_at, "status": "open"})
            self._save(incident_id, state, rec, t)
            return len(rec["corrective_actions"]) - 1

    def verify_corrective_action(self, incident_id: str, index: int, *, verifier: str, evidence_ref: str, now: Optional[float] = None) -> None:
        t = time.time() if now is None else now
        with self._lock:
            state, rec = self._load(incident_id)
            action = rec["corrective_actions"][index]
            if verifier == action["owner"]:
                raise IncidentError("corrective action must be verified by someone other than its owner")
            action.update(status="verified", verified_by=verifier, evidence_ref=evidence_ref)
            self._save(incident_id, state, rec, t)

    # ------------------------------------------------------------------ queries
    def get(self, incident_id: str) -> Dict[str, Any]:
        state, rec = self._load(incident_id)
        return {**rec, "state": state.value}

    def breach_register(self) -> List[Dict[str, Any]]:
        """GDPR Art. 33(5): every personal-data breach, including those not notified."""
        rows = self._db.execute("SELECT incident_id FROM incidents WHERE kind=?", (IncidentKind.PERSONAL_DATA_BREACH.value,)).fetchall()
        return [self.get(r[0]) for r in rows]

    def timeline(self, incident_id: str) -> List[Dict[str, Any]]:
        rows = self._db.execute("SELECT event, actor, detail, at FROM incident_timeline WHERE incident_id=? ORDER BY seq", (incident_id,)).fetchall()
        return [{"event": r[0], "actor": r[1], "detail": json.loads(r[2]), "at": r[3]} for r in rows]

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _art34_exempt(rec: Mapping[str, Any]) -> bool:
        risk = rec.get("risk") or {}
        return bool(risk.get("art34_encryption_exception") or risk.get("art34_subsequent_measures_exception"))

    def _notification_expected(self, rec: Mapping[str, Any]) -> bool:
        kind = IncidentKind(rec["kind"])
        if kind is IncidentKind.AI_SERIOUS_INCIDENT:
            return True
        if kind is IncidentKind.PERSONAL_DATA_BREACH:
            return (rec["risk"] or {}).get("level") in (BreachRisk.RISK.value, BreachRisk.HIGH.value)
        return False

    def _load(self, iid: str) -> Tuple[IncidentState, Dict[str, Any]]:
        row = self._db.execute("SELECT state, record_json FROM incidents WHERE incident_id=?", (iid,)).fetchone()
        if row is None:
            raise IncidentError(f"unknown incident {iid}")
        return IncidentState(row[0]), json.loads(row[1])

    def _save(self, iid: str, state: IncidentState, rec: Mapping[str, Any], t: float) -> None:
        self._db.execute("UPDATE incidents SET state=?, record_json=?, updated_at=? WHERE incident_id=?", (state.value, json.dumps(rec), t, iid))

    def _log(self, iid: str, event: str, actor: str, detail: Mapping[str, Any], t: float) -> None:
        self._db.execute("INSERT INTO incident_timeline(incident_id, event, actor, detail, at) VALUES (?,?,?,?,?)", (iid, event, actor, json.dumps(detail), t))
