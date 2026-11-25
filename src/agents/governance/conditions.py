"""Conditional-approval enforcement lifecycle (GOV-P1-01).

``ALLOW_WITH_CONTROLS`` is not an unconditional permission. Every control attached to a decision is a **tracked obligation** with
a phase, an owner, a verifier and a deadline:

    PRE      must hold *before* the operation may start. An unmet mandatory PRE condition turns the decision into
             REVIEW_REQUIRED (GOV-CND-02) inside the policy engine, i.e. BEFORE any single-use approval or classification
             evidence is consumed. Satisfaction is bound to the exact request fingerprint, so it cannot be borrowed by another
             request, tenant or purpose.
    RUNTIME  re-verified at the side-effect boundary on every effect (a failing or unverifiable runtime condition refuses it).
    RELEASE  must hold before the output leaves the agent (e.g. Art. 50 marking); enforced by the agent gate.
    POST     follow-up the operation leaves behind (reviews, open legal questions); tracked, owned, deadline-driven, escalated.

* Unknown controls fail closed (GOV-CND-01): a control nobody knows how to verify must not be silently dropped.
* Where governance can fulfil a condition itself it does so and records signed evidence (``EvidenceStore``): ``log_transfer`` and
  ``notify_human_oversight``. Where a human or another system owns it (``apply_supplementary_measures``, reviewer conditions),
  it stays PENDING until a verified holder of the owner role attests it with an evidence reference. The requester of the
  operation can never attest its own condition.
* Mandatory conditions are waivable only where the catalogue says so, only by an owner-role holder, with a rationale.
* Every transition is appended to a hash-chained event log. Restarts keep obligations; a RELEASE obligation that is unresolved
  after a restart still blocks the release.

The catalogue is data (``ConditionSpec``) and extensible (``ConditionCatalogue.register``); nothing here knows about a specific
agent. Stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import time

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .processing_context import (Decision, GovernanceDenied, IdentityContext, ProcessingContext, ProcessingDecision, RuleResult, new_id)
from .utils.identity import IdentityAuthority, IdentityVerificationError
from .utils.store import open_db

__all__ = [
    "ConditionCatalogue", "ConditionError", "ConditionManager", "ConditionSpec", "ConditionsUnmet", "Obligation", "Phase", "Status",
    "default_catalogue",
]

_GENESIS = "0" * 64
INTERNAL_CONTROL_PREFIXES = ("consume_approval:",)  # handled by the core's atomic consumption, not obligations


class Phase(str, Enum):
    PRE = "pre"
    RUNTIME = "runtime"
    RELEASE = "release"
    POST = "post"


class Status(str, Enum):
    PENDING = "pending"
    SATISFIED = "satisfied"
    FAILED = "failed"
    WAIVED = "waived"


class ConditionError(PermissionError):
    """Raised for an attestation, waiver or failure that is not permitted."""


class ConditionsUnmet(GovernanceDenied):
    """A runtime or release gate refused because mandatory conditions are unresolved."""

    def __init__(self, auth_id: str, phase: Phase, unmet: Sequence[str], rule_id: str = "GOV-CND-03"):
        self.unmet, self.phase = tuple(unmet), phase
        decision = ProcessingDecision(
            request_id=auth_id or "-", decision=Decision.BLOCK,
            results=(RuleResult(rule_id, Decision.BLOCK, f"unmet_{phase.value}_conditions:{','.join(unmet)}", ("GDPR Art. 25", "GDPR Art. 32")),),
            policy_version="", context_fingerprint="")
        super().__init__(decision)


@dataclass(frozen=True)
class ConditionSpec:
    name: str  # exact control name, or a prefix ending in ":" matching "<prefix><anything>"
    phase: Phase
    owner_roles: Tuple[str, ...] = ("dpo", "governance_admin")
    mandatory: bool = True
    waivable: bool = False
    auto: Optional[str] = None  # key of a governance-side fulfiller, if governance can satisfy it itself
    ttl_seconds: float = 900.0  # how long a PRE satisfaction stays valid for the same request fingerprint
    due_seconds: Optional[float] = None  # POST deadline
    legal_refs: Tuple[str, ...] = ()
    description: str = ""


def default_catalogue() -> List[ConditionSpec]:
    return [
        ConditionSpec("log_transfer", Phase.PRE, ("governance_admin",), auto="transfer_record", legal_refs=("GDPR Art. 30(1)(e)", "GDPR Art. 44"),
                      description="record the restricted transfer (destination, mechanism) in the signed evidence log"),
        ConditionSpec("apply_supplementary_measures", Phase.PRE, ("dpo", "security_officer", "governance_admin"), auto="supplementary_measures",
                      legal_refs=("GDPR Art. 46", "GDPR Art. 32"),
                      description="supplementary technical/organisational measures from the transfer impact assessment are in place"),
        ConditionSpec("notify_human_oversight", Phase.PRE, ("governance_admin",), auto="oversight_notice", legal_refs=("AI Act Art. 14",),
                      description="durably notify the oversight function so it can monitor and intervene"),
        ConditionSpec("art50_transparency_obligations", Phase.RELEASE, ("governance_admin",), legal_refs=("AI Act Art. 50",),
                      description="output carries the required marking/disclosure before release"),
        ConditionSpec("high_risk_readiness_controls", Phase.POST, ("governance_admin", "dpo"), mandatory=False, due_seconds=30 * 86400.0,
                      legal_refs=("AI Act Art. 8-15",), description="high-risk readiness obligations tracked to completion"),
        ConditionSpec("open_question:", Phase.POST, ("dpo",), mandatory=False, due_seconds=90 * 86400.0,
                      description="unresolved legal question flagged by the regulatory profile needs a documented answer"),
        ConditionSpec("approved_condition:", Phase.RELEASE, ("oversight_reviewer", "dpo", "governance_admin"), waivable=True,
                      legal_refs=("AI Act Art. 14", "GDPR Art. 22(3)"),
                      description="condition imposed by the human reviewer; default phase RELEASE, or pre:/runtime:/post:/release: prefixed"),
    ]


class ConditionCatalogue:
    def __init__(self, specs: Iterable[ConditionSpec] = ()):
        self._exact: Dict[str, ConditionSpec] = {}
        self._prefix: Dict[str, ConditionSpec] = {}
        for s in specs:
            self.register(s)

    def register(self, spec: ConditionSpec) -> None:
        (self._prefix if spec.name.endswith(":") else self._exact)[spec.name] = spec

    def resolve(self, control: str) -> Optional[ConditionSpec]:
        spec = self._exact.get(control)
        if spec is not None:
            return spec
        for prefix, candidate in self._prefix.items():
            if control.startswith(prefix) and len(control) > len(prefix):
                if prefix == "approved_condition:":
                    return _typed_reviewer_condition(candidate, control[len(prefix):])
                return candidate
        return None

    def names(self) -> List[str]:
        return sorted([*self._exact, *self._prefix])


def _typed_reviewer_condition(base: ConditionSpec, text: str) -> ConditionSpec:
    for phase in Phase:
        if text.startswith(f"{phase.value}:") and len(text) > len(phase.value) + 1:
            return ConditionSpec(base.name, phase, base.owner_roles, base.mandatory, base.waivable, None, base.ttl_seconds, base.due_seconds,
                                 base.legal_refs, base.description)
    return base


@dataclass(frozen=True)
class Obligation:
    obligation_id: str
    scope: str  # request fingerprint (PRE gate) or authorisation id
    condition: str
    phase: Phase
    mandatory: bool
    owner_roles: Tuple[str, ...]
    status: Status
    requester: str
    tenant_id: Optional[str]
    created_at: float
    due_at: Optional[float]
    resolved_at: Optional[float] = None
    resolved_by: Optional[str] = None
    evidence_ref: Optional[str] = None
    note: str = ""

    @property
    def unresolved(self) -> bool:
        return self.status in (Status.PENDING, Status.FAILED)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS obligations (
    obligation_id TEXT PRIMARY KEY, scope TEXT NOT NULL, condition TEXT NOT NULL, phase TEXT NOT NULL, mandatory INTEGER NOT NULL,
    owner_roles TEXT NOT NULL, status TEXT NOT NULL, requester TEXT NOT NULL, tenant_id TEXT, created_at REAL NOT NULL, due_at REAL,
    resolved_at REAL, resolved_by TEXT, evidence_ref TEXT, note TEXT NOT NULL DEFAULT '', escalated INTEGER NOT NULL DEFAULT 0,
    UNIQUE(scope, condition));
CREATE INDEX IF NOT EXISTS ob_scope ON obligations(scope);
CREATE TABLE IF NOT EXISTS condition_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, obligation_id TEXT NOT NULL, kind TEXT NOT NULL, actor TEXT NOT NULL, at REAL NOT NULL,
    detail TEXT NOT NULL, prev_hash TEXT NOT NULL, hash TEXT NOT NULL);
"""

Fulfiller = Callable[[ProcessingContext, str], Optional[str]]
RuntimeVerifier = Callable[[str, str, str], Tuple[bool, str]]


class ConditionManager:
    def __init__(
        self,
        *,
        path: Union[str, Path] = ":memory:",
        authority: Optional[IdentityAuthority] = None,
        evidence: Any = None,
        activities: Any = None,
        catalogue: Optional[ConditionCatalogue] = None,
        clock: Callable[[], float] = time.time,
        terminate: Optional[Callable[[str, str], None]] = None,
    ):
        self.catalogue = catalogue or ConditionCatalogue(default_catalogue())
        self._authority, self._evidence, self._activities, self._clock = authority, evidence, activities, clock
        self._terminate = terminate  # (auth_id, reason): stop the running operation (ReassessmentEngine.terminate)
        self._lock = RLock()
        self._db = open_db(path)
        self._db.executescript(_SCHEMA)
        self.on_change: Optional[Callable[[str, str, str], None]] = None
        self._runtime_verifiers: Dict[str, RuntimeVerifier] = {}
        self._fulfillers: Dict[str, Fulfiller] = {
            "transfer_record": self._fulfil_transfer_record,
            "oversight_notice": self._fulfil_oversight_notice,
            "supplementary_measures": self._fulfil_supplementary_measures,
        }
        self.stats = {"assessed": 0, "auto_satisfied": 0, "attested": 0, "waived": 0, "failed": 0, "gate_refusals": 0}

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------ extension points
    def register_fulfiller(self, key: str, fulfiller: Fulfiller) -> None:
        """Governance-side fulfilment: returns an evidence reference when the condition now holds, else None."""
        self._fulfillers[key] = fulfiller

    def register_runtime_verifier(self, condition: str, verifier: RuntimeVerifier) -> None:
        """``verifier(auth_id, effect_kind, target) -> (holds, reason)``, called at the side-effect boundary on every effect."""
        self._runtime_verifiers[condition] = verifier

    # ------------------------------------------------------------------ policy-engine hook (PRE)
    def assess(self, ctx: ProcessingContext, controls: Sequence[str], now: Optional[float] = None) -> List[RuleResult]:
        """Called by the engine for an otherwise-permitted decision. Returns extra results: BLOCK for unknown controls and
        REVIEW_REQUIRED while mandatory PRE conditions are unmet. Idempotent: obligations are keyed by (fingerprint, condition)."""
        t = self._clock() if now is None else now
        out: List[RuleResult] = []
        unknown: List[str] = []
        unmet: List[str] = []
        fingerprint = ctx.fingerprint()
        requester = ctx.identity.principal_id if ctx.identity is not None else "-"
        for control in controls:
            if control.startswith(INTERNAL_CONTROL_PREFIXES):
                continue
            spec = self.catalogue.resolve(control)
            if spec is None:
                unknown.append(control)
                continue
            if spec.phase is not Phase.PRE:
                continue
            with self._lock:
                self.stats["assessed"] += 1
                row = self._load(fingerprint, control)
                if row is not None and row.status in (Status.SATISFIED, Status.WAIVED) and (row.resolved_at or 0) + spec.ttl_seconds >= t:
                    continue
                if row is None or row.status is Status.SATISFIED or (row.status is Status.WAIVED):  # stale satisfaction: start over
                    row = self._upsert_pending(fingerprint, control, spec, requester, ctx.tenant_id, t, reset=True)
                if row.status is Status.PENDING and spec.auto:
                    ref = self._try_fulfil(spec, ctx, control)
                    if ref:
                        self._resolve(row, Status.SATISFIED, "governance", ref, "auto", t)
                        self.stats["auto_satisfied"] += 1
                        continue
            if spec.mandatory:
                unmet.append(control)
        if unknown:
            out.append(RuleResult("GOV-CND-01", Decision.BLOCK, f"unknown_control_cannot_be_verified:{','.join(sorted(unknown))}",
                                  ("GDPR Art. 25", "GDPR Art. 32")))
        if unmet:
            out.append(RuleResult("GOV-CND-02", Decision.REVIEW_REQUIRED, f"mandatory_pre_conditions_unmet:{','.join(sorted(unmet))}",
                                  ("GDPR Art. 5(2)", "GDPR Art. 24")))
        return out

    # ------------------------------------------------------------------ binding to an authorisation
    def bind(self, ctx: ProcessingContext, decision: ProcessingDecision, auth_id: str, now: Optional[float] = None) -> List[Obligation]:
        """After a decision was allowed: create the RUNTIME / RELEASE / POST obligations of this authorisation."""
        t = self._clock() if now is None else now
        requester = ctx.identity.principal_id if ctx.identity is not None else "-"
        created: List[Obligation] = []
        for control in decision.conditions:
            if control.startswith(INTERNAL_CONTROL_PREFIXES):
                continue
            spec = self.catalogue.resolve(control)
            if spec is None or spec.phase is Phase.PRE:
                continue
            with self._lock:
                row = self._upsert_pending(auth_id, control, spec, requester, ctx.tenant_id, t, reset=False)
                if row.status is Status.PENDING and spec.auto:
                    ref = self._try_fulfil(spec, ctx, control)
                    if ref:
                        row = self._resolve(row, Status.SATISFIED, "governance", ref, "auto", t)
                created.append(row)
        return created

    # ------------------------------------------------------------------ attestation / waiver / failure
    def satisfy(self, obligation_id: str, attester: IdentityContext, evidence_ref: str, note: str = "", *, now: Optional[float] = None) -> Obligation:
        row, who = self._authorise_owner(obligation_id, attester)
        if not evidence_ref or not evidence_ref.strip():
            raise ConditionError("an attestation needs an evidence reference")
        if who == row.requester:
            raise ConditionError("the requester of the operation cannot attest its own condition")
        with self._lock:
            out = self._resolve(row, Status.SATISFIED, who, evidence_ref.strip(), note, self._clock() if now is None else now)
            self.stats["attested"] += 1
        self._emit("condition_satisfied", f"obligation:{obligation_id}", row.condition)
        return out

    def waive(self, obligation_id: str, attester: IdentityContext, rationale: str, *, now: Optional[float] = None) -> Obligation:
        row, who = self._authorise_owner(obligation_id, attester)
        spec = self.catalogue.resolve(row.condition)
        if spec is None or not spec.waivable:
            raise ConditionError(f"condition '{row.condition}' is a mandatory control and cannot be waived")
        if len((rationale or "").strip()) < 15:
            raise ConditionError("a waiver needs a documented rationale")
        if who == row.requester:
            raise ConditionError("the requester of the operation cannot waive its own condition")
        with self._lock:
            out = self._resolve(row, Status.WAIVED, who, None, rationale.strip(), self._clock() if now is None else now)
            self.stats["waived"] += 1
        self._emit("condition_waived", f"obligation:{obligation_id}", row.condition)
        return out

    def fail(self, obligation_id: str, attester: IdentityContext, reason: str, *, now: Optional[float] = None) -> Obligation:
        """Owner (or any verified holder of an owner role) reports that the condition is NOT met. A mandatory condition that
        fails after the operation started stops the operation."""
        row, who = self._authorise_owner(obligation_id, attester)
        with self._lock:
            out = self._resolve(row, Status.FAILED, who, None, (reason or "failed").strip(), self._clock() if now is None else now)
            self.stats["failed"] += 1
        if row.mandatory and row.phase is not Phase.PRE and self._terminate is not None:
            self._terminate(row.scope, f"condition_failed:{row.condition}")
        self._emit("condition_failed", f"obligation:{obligation_id}", row.condition)
        return out

    def satisfy_system(self, scope: str, condition: str, evidence_ref: str, *, now: Optional[float] = None) -> Optional[Obligation]:
        """Governance fulfilling its own condition (e.g. the agent gate marked the output). Not an attestation by a person."""
        with self._lock:
            row = self._load(scope, condition)
            if row is None or not row.unresolved:
                return row
            out = self._resolve(row, Status.SATISFIED, "governance", evidence_ref, "system", self._clock() if now is None else now)
            self.stats["auto_satisfied"] += 1
            return out

    # ------------------------------------------------------------------ gates
    def runtime_gate(self, auth_id: Optional[str], kind: str, target: str) -> None:
        """Side-effect boundary: every RUNTIME obligation must be verifiable *now*; otherwise the effect is refused."""
        if not auth_id:
            return
        unmet: List[str] = []
        for row in self.obligations(auth_id, phase=Phase.RUNTIME):
            if row.status in (Status.SATISFIED, Status.WAIVED) and row.evidence_ref != "runtime":
                continue
            verifier = self._runtime_verifiers.get(row.condition)
            if verifier is None:
                if row.mandatory:
                    unmet.append(row.condition)
                continue
            try:
                holds, _reason = verifier(auth_id, kind, target)
            except Exception:
                holds = False
            if not holds and row.mandatory:
                unmet.append(row.condition)
        if unmet:
            self.stats["gate_refusals"] += 1
            raise ConditionsUnmet(auth_id, Phase.RUNTIME, unmet, "GOV-CND-03")

    def release_unmet(self, auth_id: Optional[str]) -> List[str]:
        if not auth_id:
            return []
        return [r.condition for r in self.obligations(auth_id, phase=Phase.RELEASE) if r.mandatory and r.unresolved]

    def release_gate(self, auth_id: Optional[str]) -> None:
        unmet = self.release_unmet(auth_id)
        if unmet:
            self.stats["gate_refusals"] += 1
            raise ConditionsUnmet(auth_id or "-", Phase.RELEASE, unmet, "GOV-CND-04")

    # ------------------------------------------------------------------ queries
    def obligations(self, scope: str, *, phase: Optional[Phase] = None) -> List[Obligation]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM obligations WHERE scope=? ORDER BY created_at, condition", (scope,)).fetchall()
        out = [self._row(r) for r in rows]
        return [o for o in out if phase is None or o.phase is phase]

    def get(self, obligation_id: str) -> Optional[Obligation]:
        with self._lock:
            r = self._db.execute("SELECT * FROM obligations WHERE obligation_id=?", (obligation_id,)).fetchone()
        return self._row(r) if r else None

    def pending(self, tenant_id: Optional[str] = None) -> List[Obligation]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM obligations WHERE status IN ('pending','failed') ORDER BY created_at").fetchall()
        return [o for o in map(self._row, rows) if tenant_id is None or o.tenant_id == tenant_id]

    def overdue(self, now: Optional[float] = None) -> List[Obligation]:
        t = self._clock() if now is None else now
        return [o for o in self.pending() if o.due_at is not None and o.due_at < t]

    def escalate_overdue(self, now: Optional[float] = None) -> int:
        """Announce each overdue obligation once (reviewers, dashboards and the incident bridge subscribe to the change bus)."""
        count = 0
        for o in self.overdue(now):
            with self._lock:
                cur = self._db.execute("UPDATE obligations SET escalated=1 WHERE obligation_id=? AND escalated=0", (o.obligation_id,))
                fresh = cur.rowcount == 1
            if fresh:
                self._append_event(o.obligation_id, "escalated", "governance", "overdue", self._clock() if now is None else now)
                self._emit("condition_overdue", f"obligation:{o.obligation_id}", o.condition)
                count += 1
        return count

    def events(self, obligation_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT kind,actor,at,detail FROM condition_events WHERE obligation_id=? ORDER BY seq", (obligation_id,)).fetchall()
        return [{"kind": k, "actor": a, "at": t, "detail": d} for k, a, t, d in rows]

    def verify_log(self) -> bool:
        with self._lock:
            prev = _GENESIS
            for row in self._db.execute("SELECT obligation_id,kind,actor,at,detail,prev_hash,hash FROM condition_events ORDER BY seq"):
                expected = hashlib.sha256((prev + "|".join(map(str, (row[0], row[1], row[2], repr(row[3]), row[4])))).encode("utf-8")).hexdigest()
                if row[5] != prev or row[6] != expected:
                    return False
                prev = row[6]
        return True

    def summary(self, now: Optional[float] = None) -> Dict[str, Any]:
        pending = self.pending()
        return {"pending": len(pending), "overdue": len(self.overdue(now)),
                "pending_mandatory": sum(1 for o in pending if o.mandatory), "stats": dict(self.stats), "log_intact": self.verify_log()}

    # ------------------------------------------------------------------ internals
    def _authorise_owner(self, obligation_id: str, attester: IdentityContext) -> Tuple[Obligation, str]:
        row = self.get(obligation_id)
        if row is None:
            raise ConditionError("unknown obligation")
        if not row.unresolved and row.status is not Status.SATISFIED:
            raise ConditionError(f"obligation is already {row.status.value}")
        if self._authority is None:
            raise ConditionError("identity authority unavailable: attestations cannot be verified")
        try:
            verified = self._authority.verify(getattr(attester, "token", None), now=self._clock())
        except IdentityVerificationError as exc:
            raise ConditionError(f"attester identity invalid: {exc.code}") from exc
        if verified.actor_type != "human":
            raise ConditionError("only a human owner can attest or waive a condition")
        if row.tenant_id and verified.tenant_id not in (None, row.tenant_id):
            raise ConditionError("attester belongs to another tenant")
        if not (set(row.owner_roles) & set(verified.roles)):
            raise ConditionError("attester does not hold an owner role for this condition")
        return row, verified.principal_id

    def _try_fulfil(self, spec: ConditionSpec, ctx: ProcessingContext, control: str) -> Optional[str]:
        fulfiller = self._fulfillers.get(spec.auto or "")
        if fulfiller is None:
            return None
        try:
            return fulfiller(ctx, control)
        except Exception:  # cannot fulfil => stays pending => fail closed
            return None

    def _producer_token(self) -> str:
        assert self._authority is not None
        return self._authority.issue("governance:conditions", None, actor_type="service", ttl_seconds=60.0)

    def _evidence_ref(self, control_id: str, evidence_type: str, ctx: ProcessingContext, label: str, summary: Mapping[str, Any]) -> Optional[str]:
        if self._evidence is None or self._authority is None or not getattr(self._evidence, "available", False):
            return None
        record = self._evidence.append(
            producer_token=self._producer_token(), control_id=control_id, evidence_type=evidence_type,
            artifact_digest=hashlib.sha256(f"{label}|{ctx.fingerprint()}".encode("utf-8")).hexdigest(),
            artifact_ref=f"{label}:{ctx.fingerprint()[:16]}", result="info", summary=dict(summary), now=self._clock())
        return f"evidence:{record.seq}"

    def _fulfil_transfer_record(self, ctx: ProcessingContext, control: str) -> Optional[str]:
        return self._evidence_ref("GOV-TRF-09", "transfer_record", ctx, "transfer", {
            "destination": (ctx.destination_country or "").upper(), "recipients": len(ctx.recipients), "operation": ctx.operation.value,
            "tenant": ctx.tenant_id or "-"})

    def _fulfil_oversight_notice(self, ctx: ProcessingContext, control: str) -> Optional[str]:
        return self._evidence_ref("GOV-HO-14", "human_oversight_record", ctx, "oversight_notice", {
            "capability": ctx.capability or "-", "ai_system": ctx.ai_system_id or "-", "tenant": ctx.tenant_id or "-"})

    def _fulfil_supplementary_measures(self, ctx: ProcessingContext, control: str) -> Optional[str]:
        activity = self._activities.get(ctx.activity_id) if (self._activities is not None and ctx.activity_id) else None
        dest = (ctx.destination_country or "").upper()
        for entry in (getattr(activity, "transfers", ()) or ()):
            if str(entry.get("country", "")).upper() == dest and str(entry.get("supplementary_measures_ref") or "").strip():
                return f"activity:{ctx.activity_id}:{entry['supplementary_measures_ref']}"
        return None

    def _emit(self, kind: str, key: str, detail: str) -> None:
        if self.on_change is not None:
            self.on_change(kind, key, detail)

    # -- persistence
    def _load(self, scope: str, condition: str) -> Optional[Obligation]:
        r = self._db.execute("SELECT * FROM obligations WHERE scope=? AND condition=?", (scope, condition)).fetchone()
        return self._row(r) if r else None

    def _upsert_pending(self, scope: str, condition: str, spec: ConditionSpec, requester: str, tenant: Optional[str], t: float, *, reset: bool) -> Obligation:
        existing = self._load(scope, condition)
        if existing is not None and not reset:
            return existing
        due = t + spec.due_seconds if spec.due_seconds else None
        if existing is not None:
            self._db.execute("UPDATE obligations SET status='pending', created_at=?, due_at=?, resolved_at=NULL, resolved_by=NULL, evidence_ref=NULL, note='', "
                             "requester=? WHERE obligation_id=?", (t, due, requester, existing.obligation_id))
            self._append_event(existing.obligation_id, "reopened", "governance", "stale_or_waived", t)
            return self._load(scope, condition)  # type: ignore[return-value]
        oid = new_id("cnd")
        self._db.execute(
            "INSERT INTO obligations(obligation_id,scope,condition,phase,mandatory,owner_roles,status,requester,tenant_id,created_at,due_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (oid, scope, condition, spec.phase.value, int(spec.mandatory), json.dumps(list(spec.owner_roles)), Status.PENDING.value, requester, tenant, t, due))
        self._append_event(oid, "created", "governance", f"{spec.phase.value}:{'mandatory' if spec.mandatory else 'tracked'}", t)
        return self._load(scope, condition)  # type: ignore[return-value]

    def _resolve(self, row: Obligation, status: Status, actor: str, evidence_ref: Optional[str], note: str, t: float) -> Obligation:
        self._db.execute("UPDATE obligations SET status=?, resolved_at=?, resolved_by=?, evidence_ref=?, note=? WHERE obligation_id=?",
                         (status.value, t, actor, evidence_ref, note[:500], row.obligation_id))
        self._append_event(row.obligation_id, status.value, actor, (evidence_ref or note)[:200], t)
        return self.get(row.obligation_id)  # type: ignore[return-value]

    def _append_event(self, obligation_id: str, kind: str, actor: str, detail: str, t: float) -> None:
        r = self._db.execute("SELECT hash FROM condition_events ORDER BY seq DESC LIMIT 1").fetchone()
        prev = r[0] if r else _GENESIS
        digest = hashlib.sha256((prev + "|".join(map(str, (obligation_id, kind, actor, repr(t), detail)))).encode("utf-8")).hexdigest()
        self._db.execute("INSERT INTO condition_events(obligation_id,kind,actor,at,detail,prev_hash,hash) VALUES (?,?,?,?,?,?,?)",
                         (obligation_id, kind, actor, t, detail, prev, digest))

    @staticmethod
    def _row(r: Sequence[Any]) -> Obligation:
        return Obligation(r[0], r[1], r[2], Phase(r[3]), bool(r[4]), tuple(json.loads(r[5])), Status(r[6]), r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14])
