"""Records of Processing Activities (GDPR Art. 30) and the lawful-basis anchor.

A processing activity is the only place a lawful basis can come from. Runtime
requests reference ``activity_id``; they never carry their own basis. The
registry validates evidence structurally at registration (see ``lawful_basis``),
stores every revision in an append-only history and uses optimistic concurrency
so concurrent agents/instances sharing one database cannot overwrite each other.
"""

from __future__ import annotations

import json
import sqlite3
import time

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from .lawful_basis import validate_art9_evidence, validate_art10_evidence, validate_basis_evidence
from .processing_context import (
    Art9Condition,
    Art10Condition,
    DataClass,
    LawfulBasis,
    Operation,
    SPECIAL_DATA_CLASSES,
)

__all__ = [
    "ActivityRegistrationError",
    "ActivityStatus",
    "ProcessingActivity",
    "ProcessingRegistry",
    "ConcurrencyError",
]

TRANSFER_MECHANISMS = frozenset({"adequacy", "scc", "bcr", "art49_derogation"})


class ActivityStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    RETIRED = "retired"


class ActivityRegistrationError(ValueError):
    def __init__(self, defects: List[str]):
        self.defects = list(defects)
        super().__init__("; ".join(self.defects))


class ConcurrencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessingActivity:
    activity_id: str
    name: str
    controller: str
    purposes: Tuple[str, ...]
    data_categories: Tuple[DataClass, ...]
    subject_categories: Tuple[str, ...]
    basis: LawfulBasis
    basis_evidence: Mapping[str, Any]
    operations: Tuple[Operation, ...]
    retention_days: Optional[int] = None
    retention_criteria: str = ""
    recipients: Tuple[str, ...] = ()
    transfers: Tuple[Mapping[str, Any], ...] = ()
    security_measures: Tuple[str, ...] = ()
    art9_condition: Optional[Art9Condition] = None
    art9_evidence: Mapping[str, Any] = field(default_factory=dict)
    art10_condition: Optional[Art10Condition] = None
    art10_evidence: Mapping[str, Any] = field(default_factory=dict)
    controller_is_public_authority: bool = False
    dpia_required: bool = False
    dpia_ref: str = ""
    allows_learning: bool = False  # purpose compatibility for LEARN (Art. 5(1)(b), Art. 6(4))
    art22_exception: Optional[str] = None  # "contract" | "law" | "explicit_consent"
    processor_contract_refs: Tuple[str, ...] = ()
    ai_system_ids: Tuple[str, ...] = ()
    status: ActivityStatus = ActivityStatus.ACTIVE
    valid_from: float = 0.0
    valid_to: Optional[float] = None
    version: int = 1

    def is_active(self, now: float) -> bool:
        if self.status is not ActivityStatus.ACTIVE:
            return False
        if now < self.valid_from:
            return False
        return self.valid_to is None or now < self.valid_to

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activity_id": self.activity_id,
            "name": self.name,
            "controller": self.controller,
            "purposes": list(self.purposes),
            "data_categories": [c.value for c in self.data_categories],
            "subject_categories": list(self.subject_categories),
            "basis": self.basis.value,
            "basis_evidence": dict(self.basis_evidence),
            "operations": [o.value for o in self.operations],
            "retention_days": self.retention_days,
            "retention_criteria": self.retention_criteria,
            "recipients": list(self.recipients),
            "transfers": [dict(t) for t in self.transfers],
            "security_measures": list(self.security_measures),
            "art9_condition": self.art9_condition.value if self.art9_condition else None,
            "art9_evidence": dict(self.art9_evidence),
            "art10_condition": self.art10_condition.value if self.art10_condition else None,
            "art10_evidence": dict(self.art10_evidence),
            "controller_is_public_authority": self.controller_is_public_authority,
            "dpia_required": self.dpia_required,
            "dpia_ref": self.dpia_ref,
            "allows_learning": self.allows_learning,
            "art22_exception": self.art22_exception,
            "processor_contract_refs": list(self.processor_contract_refs),
            "ai_system_ids": list(self.ai_system_ids),
            "status": self.status.value,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ProcessingActivity":
        return cls(
            activity_id=str(d["activity_id"]),
            name=str(d.get("name", "")),
            controller=str(d.get("controller", "")),
            purposes=tuple(d.get("purposes", ())),
            data_categories=tuple(DataClass(c) for c in d.get("data_categories", ())),
            subject_categories=tuple(d.get("subject_categories", ())),
            basis=LawfulBasis(d["basis"]),
            basis_evidence=dict(d.get("basis_evidence", {})),
            operations=tuple(Operation(o) for o in d.get("operations", ())),
            retention_days=d.get("retention_days"),
            retention_criteria=str(d.get("retention_criteria", "")),
            recipients=tuple(d.get("recipients", ())),
            transfers=tuple(dict(t) for t in d.get("transfers", ())),
            security_measures=tuple(d.get("security_measures", ())),
            art9_condition=Art9Condition(d["art9_condition"]) if d.get("art9_condition") else None,
            art9_evidence=dict(d.get("art9_evidence", {})),
            art10_condition=Art10Condition(d["art10_condition"]) if d.get("art10_condition") else None,
            art10_evidence=dict(d.get("art10_evidence", {})),
            controller_is_public_authority=bool(d.get("controller_is_public_authority", False)),
            dpia_required=bool(d.get("dpia_required", False)),
            dpia_ref=str(d.get("dpia_ref", "")),
            allows_learning=bool(d.get("allows_learning", False)),
            art22_exception=d.get("art22_exception"),
            processor_contract_refs=tuple(d.get("processor_contract_refs", ())),
            ai_system_ids=tuple(d.get("ai_system_ids", ())),
            status=ActivityStatus(d.get("status", "active")),
            valid_from=float(d.get("valid_from", 0.0)),
            valid_to=d.get("valid_to"),
            version=int(d.get("version", 1)),
        )


def validate_activity(activity: ProcessingActivity) -> List[str]:
    """Structural validation of an Art. 30 record and its lawful-basis evidence."""
    d: List[str] = []
    if not activity.activity_id.strip():
        d.append("activity_id_required")
    if not activity.controller.strip():
        d.append("controller_required (Art. 30(1)(a))")
    if not activity.purposes or not all(p.strip() for p in activity.purposes):
        d.append("purposes_required (Art. 30(1)(b))")
    if not activity.data_categories or DataClass.UNCLASSIFIED in activity.data_categories:
        d.append("data_categories_required_and_must_be_classified (Art. 30(1)(c))")
    if not activity.subject_categories:
        d.append("subject_categories_required (Art. 30(1)(c))")
    if not activity.operations:
        d.append("operations_required")
    if activity.retention_days is None and not activity.retention_criteria.strip():
        d.append("retention_period_or_criteria_required (Art. 30(1)(f))")
    if activity.retention_days is not None and activity.retention_days <= 0:
        d.append("retention_days_must_be_positive")
    if not activity.security_measures:
        d.append("security_measures_required (Art. 30(1)(g), Art. 32)")

    d.extend(
        validate_basis_evidence(
            activity.basis,
            activity.basis_evidence,
            controller_is_public_authority=activity.controller_is_public_authority,
        )
    )

    classes = set(activity.data_categories)
    if classes & SPECIAL_DATA_CLASSES:
        if activity.art9_condition is None:
            d.append("art9_condition_required_for_special_category_or_biometric_data")
        else:
            d.extend(validate_art9_evidence(activity.art9_condition, activity.art9_evidence))
    if DataClass.CRIMINAL_OFFENCE in classes:
        if activity.art10_condition is None:
            d.append("art10_condition_required_for_criminal_offence_data")
        else:
            d.extend(validate_art10_evidence(activity.art10_condition, activity.art10_evidence))

    for t in activity.transfers:
        if not str(t.get("country", "")).strip():
            d.append("transfer_country_required (Art. 30(1)(e))")
        if t.get("mechanism") not in TRANSFER_MECHANISMS:
            d.append(f"transfer_mechanism_invalid:{t.get('mechanism')}")
        if not str(t.get("ref", "")).strip():
            d.append("transfer_evidence_ref_required")
    if activity.dpia_required and not activity.dpia_ref.strip():
        d.append("dpia_ref_required (Art. 35)")
    if activity.art22_exception not in (None, "contract", "law", "explicit_consent"):
        d.append("art22_exception_invalid")
    if activity.valid_to is not None and activity.valid_to <= activity.valid_from:
        d.append("valid_to_must_be_after_valid_from")
    return d


_SCHEMA = """
CREATE TABLE IF NOT EXISTS activities (
    activity_id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    record_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS activity_history (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    change TEXT NOT NULL,
    actor TEXT NOT NULL,
    record_json TEXT NOT NULL,
    at REAL NOT NULL
);
"""


class ProcessingRegistry:
    """SQLite-backed Art. 30 register. ``path=':memory:'`` is allowed for tests only."""

    def __init__(self, path: Union[str, Path] = ":memory:"):
        self._lock = RLock()
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        if str(path) != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------
    def register(self, activity: ProcessingActivity, *, actor: str) -> ProcessingActivity:
        defects = validate_activity(activity)
        if defects:
            raise ActivityRegistrationError(defects)
        if not actor:
            raise ActivityRegistrationError(["actor_required"])
        now = time.time()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT version FROM activities WHERE activity_id=?", (activity.activity_id,)
                ).fetchone()
                if row is not None:
                    raise ConcurrencyError(f"activity {activity.activity_id} already registered; use update()")
                stored = _with(activity, version=1, valid_from=activity.valid_from or now)
                self._write(stored, "register", actor, now, insert=True)
                self._db.execute("COMMIT")
                return stored
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def update(
        self, activity: ProcessingActivity, *, expected_version: int, actor: str
    ) -> ProcessingActivity:
        defects = validate_activity(activity)
        if defects:
            raise ActivityRegistrationError(defects)
        now = time.time()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT version FROM activities WHERE activity_id=?", (activity.activity_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(activity.activity_id)
                if row[0] != expected_version:
                    raise ConcurrencyError(
                        f"version conflict for {activity.activity_id}: have {row[0]}, expected {expected_version}"
                    )
                stored = _with(activity, version=expected_version + 1)
                self._write(stored, "update", actor, now, insert=False)
                self._db.execute("COMMIT")
                return stored
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def set_status(self, activity_id: str, status: ActivityStatus, *, actor: str, reason: str) -> ProcessingActivity:
        if not reason.strip():
            raise ValueError("reason required for status change")
        current = self.get(activity_id)
        if current is None:
            raise KeyError(activity_id)
        return self.update(_with(current, status=status), expected_version=current.version, actor=actor)

    def get(self, activity_id: Optional[str]) -> Optional[ProcessingActivity]:
        if not activity_id:
            return None
        with self._lock:
            row = self._db.execute(
                "SELECT record_json FROM activities WHERE activity_id=?", (activity_id,)
            ).fetchone()
        return ProcessingActivity.from_dict(json.loads(row[0])) if row else None

    def list(self, *, status: Optional[ActivityStatus] = None) -> List[ProcessingActivity]:
        with self._lock:
            rows = self._db.execute("SELECT record_json FROM activities ORDER BY activity_id").fetchall()
        items = [ProcessingActivity.from_dict(json.loads(r[0])) for r in rows]
        return [a for a in items if status is None or a.status is status]

    def history(self, activity_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT seq, version, change, actor, at FROM activity_history WHERE activity_id=? ORDER BY seq",
                (activity_id,),
            ).fetchall()
        return [{"seq": r[0], "version": r[1], "change": r[2], "actor": r[3], "at": r[4]} for r in rows]

    def export_ropa(self) -> List[Dict[str, Any]]:
        """Art. 30(4): record made available to the supervisory authority on request."""
        return [a.to_dict() for a in self.list()]

    # ------------------------------------------------------------------
    def _write(self, a: ProcessingActivity, change: str, actor: str, now: float, *, insert: bool) -> None:
        blob = json.dumps(a.to_dict(), sort_keys=True)
        if insert:
            self._db.execute(
                "INSERT INTO activities(activity_id, version, record_json, updated_at) VALUES (?,?,?,?)",
                (a.activity_id, a.version, blob, now),
            )
        else:
            self._db.execute(
                "UPDATE activities SET version=?, record_json=?, updated_at=? WHERE activity_id=?",
                (a.version, blob, now, a.activity_id),
            )
        self._db.execute(
            "INSERT INTO activity_history(activity_id, version, change, actor, record_json, at) VALUES (?,?,?,?,?,?)",
            (a.activity_id, a.version, change, actor, blob, now),
        )


def _with(activity: ProcessingActivity, **changes: Any) -> ProcessingActivity:
    data = activity.to_dict()
    for key, value in changes.items():
        data[key] = value.value if isinstance(value, Enum) else value
    return ProcessingActivity.from_dict(data)
