"""Human oversight, approvals, emergency termination and contestation.

Two separate legal regimes are kept separate on purpose:

* GDPR Art. 22 (solely automated decisions with legal/similarly significant
  effect): restriction, exceptions and the Art. 22(3) safeguard of *human
  intervention*, contest and expression of view.
* AI Act Art. 14 (human oversight of high-risk systems).

``OversightManager`` provides the shared mechanism: a request is bound to the
exact context fingerprint, shown to a verified, qualified *human* with a decision
dossier, and approved with a scoped, expiring, revocable, usually single-use
approval. An AI agent's approval is never accepted as human oversight.
"""

from __future__ import annotations

import json
import time

from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from pathlib import Path

from .processing_context import IdentityContext, ProcessingContext, fingerprint, new_id
from .utils.identity import IdentityAuthority, IdentityVerificationError
from .utils.store import open_db

__all__ = [
    "ApprovalStatus",
    "ContestationOutcome",
    "DecisionDossier",
    "OversightError",
    "OversightLevel",
    "OversightManager",
    "required_oversight_level",
]

DAY = 86400.0


class OversightError(PermissionError):
    def __init__(self, code: str, message: str = ""):
        self.code = code
        super().__init__(message or code)


class OversightLevel(str, Enum):
    NONE = "none"  # routine, lower-risk: no human in the loop
    NOTIFY = "notify"  # logged and reviewable after the fact
    REVIEW = "review"  # human review before execution
    APPROVAL = "approval"  # qualified human approval, four-eyes, single use


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REVOKED = "revoked"
    CONSUMED = "consumed"
    CANCELLED = "cancelled"


class ContestationOutcome(str, Enum):
    UPHELD = "upheld"
    OVERTURNED = "overturned"
    REVISED = "revised"


def required_oversight_level(
    ctx: ProcessingContext,
    *,
    high_risk_system: bool = False,
    capability_requires_approval: bool = False,
) -> OversightLevel:
    """Risk-dependent, deliberately narrow: routine lower-risk work needs no human."""
    if ctx.automated_decision and ctx.significant_effect:
        return OversightLevel.APPROVAL
    if capability_requires_approval:
        return OversightLevel.APPROVAL
    if high_risk_system and (ctx.automated_decision or ctx.significant_effect):
        return OversightLevel.APPROVAL
    if ctx.subject_is_child and ctx.automated_decision:
        return OversightLevel.REVIEW
    if high_risk_system:
        return OversightLevel.NOTIFY
    return OversightLevel.NONE


@dataclass(frozen=True)
class DecisionDossier:
    """What the human must be shown. Incomplete dossiers cannot be approved."""

    summary: str
    evidence_refs: Tuple[str, ...]
    uncertainty: str  # explicit statement of confidence/uncertainty
    alternatives: Tuple[str, ...]
    risks: Tuple[str, ...]
    limitations: Tuple[str, ...]
    explanation: str = ""  # meaningful explanation of the decision logic

    def defects(self) -> List[str]:
        d: List[str] = []
        if len(self.summary.strip()) < 10:
            d.append("summary_missing")
        if not self.evidence_refs:
            d.append("evidence_missing")
        if not self.uncertainty.strip():
            d.append("uncertainty_missing")
        if not self.alternatives:
            d.append("alternatives_missing")
        if not self.risks:
            d.append("risks_missing")
        if not self.limitations:
            d.append("limitations_missing")
        return d

    def digest(self) -> str:
        return fingerprint(
            [self.summary, self.evidence_refs, self.uncertainty, self.alternatives, self.risks, self.limitations, self.explanation]
        )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY, status TEXT NOT NULL, record_json TEXT NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS oversight_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, approval_id TEXT, event TEXT NOT NULL,
    actor TEXT NOT NULL, detail TEXT NOT NULL, at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS emergency (
    scope TEXT PRIMARY KEY, active INTEGER NOT NULL, actor TEXT NOT NULL, reason TEXT NOT NULL, at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS contestations (
    case_id TEXT PRIMARY KEY, status TEXT NOT NULL, record_json TEXT NOT NULL, updated_at REAL NOT NULL);
"""


class OversightManager:
    def __init__(
        self,
        path: Union[str, Path] = ":memory:",
        *,
        qualified_roles: Iterable[str] = ("oversight_reviewer", "dpo", "governance_admin"),
        emergency_roles: Iterable[str] = ("safety_officer", "governance_admin"),
        max_approval_ttl: float = 3600.0,
        min_review_seconds: float = 5.0,
        bias_window_seconds: float = 3600.0,
        bias_max_fast_approvals: int = 5,
        authority: Optional[IdentityAuthority] = None,
    ):
        self._authority = authority
        self._lock = RLock()
        self._db = open_db(path)
        self._db.executescript(_SCHEMA)
        self.qualified_roles = frozenset(qualified_roles)
        self.emergency_roles = frozenset(emergency_roles)
        self.max_approval_ttl = max_approval_ttl
        self.min_review_seconds = min_review_seconds
        self.bias_window_seconds = bias_window_seconds
        self.bias_max_fast_approvals = bias_max_fast_approvals

    def close(self) -> None:
        self._db.close()

    # ------------------------------------------------------------------ requests
    def request_approval(
        self,
        ctx: ProcessingContext,
        dossier: DecisionDossier,
        requester: IdentityContext,
        *,
        level: OversightLevel = OversightLevel.APPROVAL,
        ttl: float = 900.0,
        single_use: bool = True,
        now: Optional[float] = None,
    ) -> str:
        t = time.time() if now is None else now
        if level is OversightLevel.NONE:
            raise OversightError("no_oversight_required")
        defects = dossier.defects()
        if defects:
            raise OversightError("dossier_incomplete", ",".join(defects))
        requester = self._authenticate(requester, t)
        ttl = min(float(ttl), self.max_approval_ttl)
        approval_id = new_id("appr")
        record = {
            "approval_id": approval_id,
            "context_fingerprint": ctx.fingerprint(),
            "level": level.value,
            "dossier_digest": dossier.digest(),
            "requester": requester.principal_id,
            "created_at": t,
            "expires_at": t + ttl,
            "single_use": single_use,
            "presented_at": {},
            "approver": None,
            "decision": None,
        }
        self._save(approval_id, ApprovalStatus.PENDING, record, t)
        self._log(approval_id, "requested", requester.principal_id, {"level": level.value}, t)
        return approval_id

    def mark_presented(self, approval_id: str, reviewer: IdentityContext, *, now: Optional[float] = None) -> None:
        """Record when the dossier was actually shown (for the minimum-review-time guard)."""
        t = time.time() if now is None else now
        with self._lock:
            status, rec = self._load(approval_id)
            rec["presented_at"][reviewer.principal_id] = t
            self._save(approval_id, status, rec, t)

    # ------------------------------------------------------------------ decisions
    def decide(
        self,
        approval_id: str,
        reviewer: IdentityContext,
        *,
        approve: bool,
        rationale: str,
        reviewed_dossier_digest: str,
        now: Optional[float] = None,
    ) -> ApprovalStatus:
        t = time.time() if now is None else now
        with self._lock:
            status, rec = self._load(approval_id)
            reviewer = self._require_human(reviewer, t)
            if status is not ApprovalStatus.PENDING:
                raise OversightError("not_pending", status.value)
            if t >= rec["expires_at"]:
                self._save(approval_id, ApprovalStatus.EXPIRED, rec, t)
                raise OversightError("expired")
            if reviewer.principal_id == rec["requester"]:
                raise OversightError("four_eyes_violation", "requester cannot approve own request")
            if not (self.qualified_roles & set(reviewer.roles)):
                raise OversightError("reviewer_not_qualified")
            if reviewed_dossier_digest != rec["dossier_digest"]:
                raise OversightError("dossier_not_reviewed", "digest does not match the dossier shown")
            if len(rationale.strip()) < 15:
                raise OversightError("rationale_required", "state your reasoning (>=15 chars)")
            shown = rec["presented_at"].get(reviewer.principal_id)
            if approve:
                if shown is None:
                    raise OversightError("dossier_not_presented")
                if t - shown < self.min_review_seconds:
                    raise OversightError("review_too_fast", "automation-bias guard: minimum review time not met")
                if self._rubber_stamping(reviewer.principal_id, t):
                    raise OversightError("automation_bias_guard", "too many rapid approvals; escalate to another reviewer")
            rec["approver"] = reviewer.principal_id
            rec["decision"] = {"approve": approve, "rationale": rationale.strip(), "at": t, "review_seconds": (t - shown) if shown else None}
            new = ApprovalStatus.APPROVED if approve else ApprovalStatus.REJECTED
            self._save(approval_id, new, rec, t)
            seconds = rec["decision"]["review_seconds"]
            fast = seconds is not None and seconds < self.min_review_seconds * 2
            self._log(
                approval_id,
                "approved" if approve else "rejected",
                reviewer.principal_id,
                {"rationale_len": len(rationale), "fast": bool(fast and approve)},
                t,
            )
            return new

    def verify(self, approval_id: Optional[str], ctx: ProcessingContext, *, now: Optional[float] = None) -> Tuple[bool, str]:
        """Pure check; does not consume. Altered, expired, revoked or foreign approvals fail."""
        t = time.time() if now is None else now
        if not approval_id:
            return False, "no_approval"
        try:
            status, rec = self._load(approval_id)
        except OversightError:
            return False, "unknown_approval"
        if status is not ApprovalStatus.APPROVED:
            return False, f"approval_{status.value}"
        if t >= rec["expires_at"]:
            return False, "approval_expired"
        if rec["context_fingerprint"] != ctx.fingerprint():
            return False, "material_change:approval_bound_to_different_request"
        return True, "ok"

    def consume(self, approval_id: str, ctx: ProcessingContext, *, now: Optional[float] = None) -> None:
        t = time.time() if now is None else now
        with self._lock:
            ok, reason = self.verify(approval_id, ctx, now=t)
            if not ok:
                raise OversightError("approval_invalid", reason)
            status, rec = self._load(approval_id)
            if rec["single_use"]:
                self._save(approval_id, ApprovalStatus.CONSUMED, rec, t)
            self._log(approval_id, "consumed", ctx.identity.principal_id if ctx.identity else "unknown", {}, t)

    def revoke(self, approval_id: str, human: IdentityContext, reason: str, *, now: Optional[float] = None) -> None:
        t = time.time() if now is None else now
        with self._lock:
            human = self._require_human(human, t)
            status, rec = self._load(approval_id)
            if status in (ApprovalStatus.REVOKED, ApprovalStatus.CONSUMED, ApprovalStatus.EXPIRED):
                return
            if not reason.strip():
                raise OversightError("reason_required")
            self._save(approval_id, ApprovalStatus.REVOKED, rec, t)
            self._log(approval_id, "revoked", human.principal_id, {"reason": reason}, t)

    def intervene(self, approval_id: str, human: IdentityContext, action: str, reason: str, *, now: Optional[float] = None) -> None:
        """Human override/cancellation of a pending or approved action."""
        if action not in ("override", "cancel"):
            raise ValueError("action must be 'override' or 'cancel'")
        t = time.time() if now is None else now
        with self._lock:
            human = self._require_human(human, t)
            if not (self.qualified_roles & set(human.roles)):
                raise OversightError("reviewer_not_qualified")
            status, rec = self._load(approval_id)
            self._save(approval_id, ApprovalStatus.CANCELLED, rec, t)
            self._log(approval_id, f"intervention:{action}", human.principal_id, {"reason": reason}, t)

    def status(self, approval_id: str) -> ApprovalStatus:
        return self._load(approval_id)[0]

    # ------------------------------------------------------------------ emergency
    def activate_emergency(self, scope: str, human: IdentityContext, reason: str, *, now: Optional[float] = None) -> None:
        """Kill switch: scope is 'global', 'tenant:<id>', 'agent:<id>' or 'capability:<id>'."""
        t = time.time() if now is None else now
        human = self._require_human(human, t)
        if not (self.emergency_roles & set(human.roles)):
            raise OversightError("not_authorised_for_emergency_control")
        if not reason.strip():
            raise OversightError("reason_required")
        with self._lock:
            self._db.execute(
                "INSERT INTO emergency(scope, active, actor, reason, at) VALUES (?,?,?,?,?) "
                "ON CONFLICT(scope) DO UPDATE SET active=1, actor=excluded.actor, reason=excluded.reason, at=excluded.at",
                (scope, 1, human.principal_id, reason, t),
            )
            self._log(None, "emergency_activated", human.principal_id, {"scope": scope, "reason": reason}, t)

    def release_emergency(self, scope: str, human: IdentityContext, reason: str, *, now: Optional[float] = None) -> None:
        t = time.time() if now is None else now
        human = self._require_human(human, t)
        if not (self.emergency_roles & set(human.roles)):
            raise OversightError("not_authorised_for_emergency_control")
        if not reason.strip():
            raise OversightError("reason_required")
        with self._lock:
            self._db.execute("UPDATE emergency SET active=0, actor=?, reason=?, at=? WHERE scope=?", (human.principal_id, reason, t, scope))
            self._log(None, "emergency_released", human.principal_id, {"scope": scope, "reason": reason}, t)

    def halted_scope(self, ctx: ProcessingContext) -> Optional[str]:
        scopes = ["global"]
        if ctx.tenant_id:
            scopes.append(f"tenant:{ctx.tenant_id}")
        if ctx.agent_id:
            scopes.append(f"agent:{ctx.agent_id}")
        if ctx.capability:
            scopes.append(f"capability:{ctx.capability}")
        with self._lock:
            marks = ",".join("?" for _ in scopes)
            row = self._db.execute(f"SELECT scope FROM emergency WHERE active=1 AND scope IN ({marks}) LIMIT 1", scopes).fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------ contestation (Art. 22(3), Art. 12(3))
    def file_contestation(self, subject_id: str, decision_ref: str, grounds: str, *, now: Optional[float] = None) -> str:
        t = time.time() if now is None else now
        if not grounds.strip():
            raise OversightError("grounds_required")
        case_id = new_id("contest")
        rec = {
            "case_id": case_id, "subject_id": subject_id, "decision_ref": decision_ref, "grounds": grounds,
            "filed_at": t, "respond_by": t + 30 * DAY, "reviewer": None, "outcome": None, "reasoning": None,
            "original_approver": self._approver_of(decision_ref),
        }
        with self._lock:
            self._db.execute("INSERT INTO contestations VALUES (?,?,?,?)", (case_id, "filed", json.dumps(rec), t))
        return case_id

    def resolve_contestation(
        self, case_id: str, reviewer: IdentityContext, outcome: ContestationOutcome, reasoning: str, *, now: Optional[float] = None
    ) -> None:
        t = time.time() if now is None else now
        reviewer = self._require_human(reviewer, t)
        if not (self.qualified_roles & set(reviewer.roles)):
            raise OversightError("reviewer_not_qualified")
        if len(reasoning.strip()) < 15:
            raise OversightError("reasoning_required")
        with self._lock:
            row = self._db.execute("SELECT status, record_json FROM contestations WHERE case_id=?", (case_id,)).fetchone()
            if row is None:
                raise OversightError("unknown_case")
            rec = json.loads(row[1])
            if row[0] != "filed":
                raise OversightError("already_resolved")
            if reviewer.principal_id == rec.get("original_approver"):
                raise OversightError("independent_reviewer_required", "contestation must be reviewed by someone other than the original approver")
            rec.update(reviewer=reviewer.principal_id, outcome=outcome.value, reasoning=reasoning.strip(), resolved_at=t)
            self._db.execute("UPDATE contestations SET status=?, record_json=?, updated_at=? WHERE case_id=?", ("resolved", json.dumps(rec), t, case_id))

    def overdue_contestations(self, *, now: Optional[float] = None) -> List[Dict[str, Any]]:
        t = time.time() if now is None else now
        rows = self._db.execute("SELECT record_json FROM contestations WHERE status='filed'").fetchall()
        return [r for r in (json.loads(x[0]) for x in rows) if r["respond_by"] < t]

    # ------------------------------------------------------------------ internals
    def _authenticate(self, who: IdentityContext, t: float) -> IdentityContext:
        """Re-verify the signed token; a hand-built ``verified=True`` context is worthless."""
        if self._authority is None:
            raise OversightError("identity_authority_unavailable")
        try:
            verified = self._authority.verify(who.token, now=t)
        except IdentityVerificationError as exc:
            raise OversightError(f"identity_{exc.code}") from exc
        if verified.principal_id != who.principal_id:
            raise OversightError("identity_mismatch")
        return verified

    def _require_human(self, who: IdentityContext, t: float) -> IdentityContext:
        who = self._authenticate(who, t)
        if who.actor_type != "human":
            raise OversightError("ai_agent_approval_is_not_human_oversight")
        return who

    def _rubber_stamping(self, principal: str, t: float) -> bool:
        rows = self._db.execute(
            "SELECT detail, at FROM oversight_log WHERE event='approved' AND actor=? AND at>=?",
            (principal, t - self.bias_window_seconds),
        ).fetchall()
        fast = sum(1 for d, _ in rows if json.loads(d).get("fast"))
        return fast >= self.bias_max_fast_approvals

    def _approver_of(self, decision_ref: str) -> Optional[str]:
        try:
            return self._load(decision_ref)[1].get("approver")
        except OversightError:
            return None

    def _load(self, approval_id: str) -> Tuple[ApprovalStatus, Dict[str, Any]]:
        row = self._db.execute("SELECT status, record_json FROM approvals WHERE approval_id=?", (approval_id,)).fetchone()
        if row is None:
            raise OversightError("unknown_approval")
        return ApprovalStatus(row[0]), json.loads(row[1])

    def _save(self, approval_id: str, status: ApprovalStatus, rec: Mapping[str, Any], t: float) -> None:
        self._db.execute(
            "INSERT INTO approvals(approval_id, status, record_json, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(approval_id) DO UPDATE SET status=excluded.status, record_json=excluded.record_json, updated_at=excluded.updated_at",
            (approval_id, status.value, json.dumps(rec, sort_keys=True), t),
        )

    def _log(self, approval_id: Optional[str], event: str, actor: str, detail: Mapping[str, Any], t: float) -> None:
        self._db.execute(
            "INSERT INTO oversight_log(approval_id, event, actor, detail, at) VALUES (?,?,?,?,?)",
            (approval_id, event, actor, json.dumps(detail, sort_keys=True), t),
        )
