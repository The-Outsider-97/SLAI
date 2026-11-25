"""Event-driven runtime reassessment (GOV-P1-06) and time-of-check/time-of-use protection (GOV-P0-03).

An authorisation is a fact about a *moment*. This module keeps every allowed decision alive as a **lease** while the operation
it authorised may still run, indexes what each lease depends on (subject, tenant, principal, activity, AI system, capability,
approval, evidence, policy ...) and, when one of those facts changes, reassesses *only the affected leases*:

    change event -> dependency index -> affected leases -> (fail closed while reassessing) -> re-evaluate -> CONTINUE | SUSPEND | CANCEL

* ``lease.check()`` is what a side-effect boundary calls immediately before acting. A lease under reassessment, suspended,
  cancelled, expired or unknown (e.g. after a restart) refuses, so authorisation and execution cannot drift apart.
* Re-evaluation uses the same ``PolicyEngine`` as the original decision; there is no second rule set. An approval that was
  legitimately consumed by *this* authorisation keeps backing it (it must not look "missing" on re-evaluation) unless it was
  revoked, cancelled or no longer matches the request.
* Every reassessment is appended to a hash-chained log with the original decision, the change, the new decision and the action
  taken, so the evidence of what changed and why survives the operation.

Stdlib only. Memory holds the live leases (they cannot survive a restart by design); the log is durable.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union

from .processing_context import Decision, ProcessingContext, ProcessingDecision, RuleResult, new_id
from .utils.store import open_db

__all__ = [
    "AuthorizationRevoked", "ChangeBus", "Lease", "LeaseState", "ReassessmentAction", "ReassessmentEngine", "ReassessmentReport",
    "dependency_keys",
]


class AuthorizationRevoked(PermissionError):
    """Raised by ``Lease.check`` when the authorisation behind an operation no longer holds."""

    def __init__(self, auth_id: str, state: str, reason: str):
        self.auth_id, self.state, self.reason = auth_id, state, reason
        super().__init__(f"authorisation {auth_id} is {state}: {reason}")


class LeaseState(str, Enum):
    ACTIVE = "active"
    REASSESSING = "reassessing"  # fails closed: nothing may act while the facts are being re-evaluated
    SUSPENDED = "suspended"  # needs remediation / human review / reauthorisation
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    COMPLETED = "completed"


class ReassessmentAction(str, Enum):
    CONTINUE = "continue"
    SUSPEND = "suspend"
    CANCEL = "cancel"


@dataclass
class Lease:
    auth_id: str
    request_id: str
    fingerprint: str
    deps: frozenset
    decision: ProcessingDecision
    ctx: ProcessingContext
    expires_at: float
    state: LeaseState = LeaseState.ACTIVE
    epoch: int = 0
    reason: str = ""
    consumed_approval: Optional[str] = None
    resumed_from: Optional[str] = None
    _engine: Any = field(default=None, repr=False, compare=False)

    def check(self) -> "Lease":
        """Call immediately before a side effect. Raises ``AuthorizationRevoked`` unless the authorisation still holds."""
        return self._engine.check(self.auth_id)

    def complete(self) -> None:
        self._engine.complete(self.auth_id)


@dataclass(frozen=True)
class ReassessmentReport:
    trigger: str
    key: str
    affected: int
    continued: Tuple[str, ...] = ()
    suspended: Tuple[str, ...] = ()
    cancelled: Tuple[str, ...] = ()


def dependency_keys(ctx: ProcessingContext, decision: Optional[ProcessingDecision] = None) -> frozenset:
    """Everything an allowed decision rests on. A change to any of these key names triggers reassessment of the lease."""
    t = ctx.tenant_id or "-"
    keys: Set[str] = {"policy", "global"}
    if ctx.tenant_id:
        keys.add(f"tenant:{ctx.tenant_id}")
    for sid in ctx.subject_ids:
        keys.add(f"subject:{t}:{sid}")
        keys.add(f"purpose:{t}:{sid}:{ctx.purpose}")
    if ctx.identity is not None:
        keys.add(f"principal:{ctx.identity.tenant_id or '-'}:{ctx.identity.principal_id}")
    if ctx.activity_id:
        keys.add(f"activity:{ctx.activity_id}")
    if ctx.ai_system_id:
        keys.add(f"ai_system:{ctx.ai_system_id}")
    if ctx.capability:
        keys.add(f"capability:{ctx.capability}")
    if ctx.agent_id:
        keys.add(f"agent:{ctx.agent_id}")
    if ctx.resource:
        keys.add(f"resource:{t}:{ctx.resource}")
    if ctx.approval_id:
        keys.add(f"approval:{ctx.approval_id}")
    if ctx.grant_id:
        keys.add(f"grant:{ctx.grant_id}")
    if decision is not None and decision.classification_evidence_id:
        keys.add(f"evidence:{decision.classification_evidence_id}")
    return frozenset(keys)


class ChangeBus:
    """Tiny synchronous publish/subscribe used by components to announce material changes (no new infrastructure)."""

    def __init__(self) -> None:
        self._subs: List[Callable[[str, str, str], None]] = []
        self._lock = RLock()
        self.epoch = 0  # bumped on every change: lets safe caches prove they have not been outdated

    def subscribe(self, callback: Callable[[str, str, str], None]) -> None:
        with self._lock:
            self._subs.append(callback)

    def publish(self, kind: str, key: str, detail: str = "") -> None:
        with self._lock:
            self.epoch += 1
            subs = list(self._subs)
        for cb in subs:
            cb(kind, key, detail)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS reassessments (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, auth_id TEXT NOT NULL, request_id TEXT NOT NULL, trigger_kind TEXT NOT NULL,
    trigger_key TEXT NOT NULL, detail TEXT NOT NULL, original_decision TEXT NOT NULL, original_fingerprint TEXT NOT NULL,
    new_decision TEXT NOT NULL, new_rules TEXT NOT NULL, action TEXT NOT NULL, at REAL NOT NULL, prev_hash TEXT NOT NULL, hash TEXT NOT NULL);
"""
_GENESIS = "0" * 64


class ReassessmentEngine:
    def __init__(
        self,
        evaluate: Callable[[ProcessingContext], ProcessingDecision],
        *,
        path: Union[str, Path] = ":memory:",
        backs_approval: Optional[Callable[[str, ProcessingContext, float], Tuple[bool, str]]] = None,
        lease_ttl: float = 300.0,
        max_leases: int = 20000,
        clock: Callable[[], float] = time.time,
    ):
        self._evaluate = evaluate
        self._backs = backs_approval
        self.lease_ttl = float(lease_ttl)
        self.max_leases = int(max_leases)
        self._clock = clock
        self._lock = RLock()
        self._leases: Dict[str, Lease] = {}
        self._index: Dict[str, Set[str]] = {}
        self._listeners: List[Callable[[Lease, ReassessmentAction, ProcessingDecision], None]] = []
        self._db = open_db(path)
        self._db.executescript(_SCHEMA)
        self.stats = {"tracked": 0, "events": 0, "affected": 0, "continued": 0, "suspended": 0, "cancelled": 0, "evaluations": 0}

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        with self._lock:
            self._db.close()

    def on_action(self, listener: Callable[[Lease, ReassessmentAction, ProcessingDecision], None]) -> None:
        """Components that run operations (executors, workers) register here to stop work on SUSPEND/CANCEL."""
        self._listeners.append(listener)

    def track(self, ctx: ProcessingContext, decision: ProcessingDecision, *, consumed_approval: Optional[str] = None,
              resumed_from: Optional[str] = None) -> Lease:
        now = self._clock()
        expires = now + self.lease_ttl
        if ctx.identity is not None and ctx.identity.expires_at is not None:
            expires = min(expires, ctx.identity.expires_at)
        lease = Lease(auth_id=new_id("auth"), request_id=decision.request_id, fingerprint=decision.context_fingerprint,
                      deps=dependency_keys(ctx, decision), decision=decision, ctx=ctx, expires_at=expires,
                      consumed_approval=consumed_approval, resumed_from=resumed_from, _engine=self)
        with self._lock:
            if len(self._leases) >= self.max_leases:
                self._prune(now)
            self._leases[lease.auth_id] = lease
            for key in lease.deps:
                self._index.setdefault(key, set()).add(lease.auth_id)
            self.stats["tracked"] += 1
        return lease

    def get(self, auth_id: Optional[str]) -> Optional[Lease]:
        with self._lock:
            return self._leases.get(auth_id or "")

    def check(self, auth_id: Optional[str]) -> Lease:
        """Fail-closed validity check for the moment of the side effect."""
        if not auth_id:
            raise AuthorizationRevoked("-", "missing", "no authorisation lease")
        with self._lock:
            lease = self._leases.get(auth_id)
            if lease is None:
                raise AuthorizationRevoked(auth_id, "unknown", "lease unknown (expired, pruned or process restarted): re-authorise")
            if lease.state is LeaseState.ACTIVE and self._clock() >= lease.expires_at:
                self._set_state(lease, LeaseState.EXPIRED, "lease_expired")
            if lease.state is not LeaseState.ACTIVE:
                raise AuthorizationRevoked(auth_id, lease.state.value, lease.reason or lease.state.value)
            return lease

    def terminate(self, auth_id: str, reason: str) -> bool:
        """Stop a running authorisation because a fact outside the policy rules changed (a mandatory condition failed)."""
        with self._lock:
            lease = self._leases.get(auth_id)
            if lease is None or lease.state not in (LeaseState.ACTIVE, LeaseState.REASSESSING):
                return False
            self._set_state(lease, LeaseState.CANCELLED, reason)
            self.stats["cancelled"] += 1
            self._log(lease, "terminated", f"authorization:{auth_id}", reason, lease.decision, ReassessmentAction.CANCEL)
        for listener in list(self._listeners):
            try:
                listener(lease, ReassessmentAction.CANCEL, lease.decision)
            except Exception:
                pass
        return True

    def complete(self, auth_id: str) -> None:
        with self._lock:
            lease = self._leases.get(auth_id)
            if lease is not None and lease.state is LeaseState.ACTIVE:
                self._set_state(lease, LeaseState.COMPLETED, "operation_finished")

    def active(self) -> List[Lease]:
        with self._lock:
            return [l for l in self._leases.values() if l.state in (LeaseState.ACTIVE, LeaseState.REASSESSING)]

    # ------------------------------------------------------------------ change events
    def notify(self, kind: str, key: str, detail: str = "") -> ReassessmentReport:
        """A material fact changed. Reassess exactly the leases that depend on ``key``; leave every other lease alone."""
        with self._lock:
            self.stats["events"] += 1
            targets = [self._leases[a] for a in sorted(self._index.get(key, ())) if a in self._leases]
            targets = [l for l in targets if l.state in (LeaseState.ACTIVE, LeaseState.REASSESSING)]
            for lease in targets:
                lease.state = LeaseState.REASSESSING  # fail closed *before* evaluating: no TOCTOU window
                lease.reason = f"reassessing:{kind}"
            self.stats["affected"] += len(targets)
        continued, suspended, cancelled = [], [], []
        for lease in targets:
            action, new_decision = self._reassess(lease, kind, key, detail)
            {ReassessmentAction.CONTINUE: continued, ReassessmentAction.SUSPEND: suspended,
             ReassessmentAction.CANCEL: cancelled}[action].append(lease.auth_id)
            for listener in list(self._listeners):
                try:
                    listener(lease, action, new_decision)
                except Exception:  # a broken listener must never undo the (already fail-closed) outcome
                    pass
        return ReassessmentReport(kind, key, len(targets), tuple(continued), tuple(suspended), tuple(cancelled))

    def _reassess(self, lease: Lease, kind: str, key: str, detail: str) -> Tuple[ReassessmentAction, ProcessingDecision]:
        ctx = lease.ctx
        approval = lease.consumed_approval or ctx.approval_id
        try:
            new = self._evaluate(dataclasses.replace(ctx, approval_id=None, grant_id=None) if approval else ctx)
            new = self._honour_consumed_approval(lease, ctx, new, approval)
            with self._lock:
                self.stats["evaluations"] += 1
        except Exception as exc:  # cannot assess => cannot continue
            new = ProcessingDecision(lease.request_id, Decision.BLOCK, (RuleResult("GOV-ERR-99", Decision.BLOCK, f"reassessment_error:{type(exc).__name__}"),),
                                     lease.decision.policy_version, lease.fingerprint, self._clock())
        if new.allowed:
            action = ReassessmentAction.CONTINUE
        elif new.decision in (Decision.REVIEW_REQUIRED, Decision.REDACT, Decision.RESTRICT):
            action = ReassessmentAction.SUSPEND  # remediable: human review, redaction, narrowed scope, reauthorisation
        else:
            action = ReassessmentAction.CANCEL  # BLOCK / QUARANTINE: the operation may not proceed
        with self._lock:
            if action is ReassessmentAction.CONTINUE:
                lease.state, lease.reason, lease.epoch = LeaseState.ACTIVE, "", lease.epoch + 1
                lease.decision = new
            else:
                self._set_state(lease, LeaseState.SUSPENDED if action is ReassessmentAction.SUSPEND else LeaseState.CANCELLED,
                                ",".join(r.rule_id for r in new.blocking_rules) or new.decision.value)
            self.stats[{"continue": "continued", "suspend": "suspended", "cancel": "cancelled"}[action.value]] += 1
            self._log(lease, kind, key, detail, new, action)
        return action, new

    def _honour_consumed_approval(self, lease: Lease, ctx: ProcessingContext, new: ProcessingDecision, approval: Optional[str]) -> ProcessingDecision:
        """The human approval that legitimately backed this authorisation was consumed when it ran. Re-evaluation must not
        mistake that for "no approval": the overridable review result is tolerated while the approval still backs the request."""
        if not approval or self._backs is None:
            return new
        overridable = [r for r in new.results if r.decision is Decision.REVIEW_REQUIRED and r.overridable]
        if not overridable:
            return new
        ok, reason = self._backs(approval, ctx, self._clock())
        if not ok:
            return new
        results = tuple(RuleResult(r.rule_id, Decision.ALLOW_WITH_CONTROLS, f"human_approval_still_valid:{approval}", r.legal_refs) if r in overridable else r
                        for r in new.results)
        decision = max((r.decision for r in results), key=lambda d: d.severity, default=Decision.ALLOW)
        return dataclasses.replace(new, results=results, decision=decision)

    # ------------------------------------------------------------------ remediation / resumption
    def resume(self, auth_id: str, reauthorize: Callable[[ProcessingContext], ProcessingDecision]) -> Optional[Lease]:
        """After remediation (e.g. a human approved, data redacted) a SUSPENDED operation re-enters through the normal
        authorisation path. Cancelled operations never resume. Returns the new lease, or None if still not allowed."""
        with self._lock:
            lease = self._leases.get(auth_id)
            if lease is None or lease.state is not LeaseState.SUSPENDED:
                raise AuthorizationRevoked(auth_id, getattr(lease, "state", LeaseState.CANCELLED).value if lease else "unknown",
                                           "only a suspended authorisation can be resumed")
        decision = reauthorize(lease.ctx)
        if not decision.allowed:
            return None
        with self._lock:
            lease.reason = "resumed_as:" + (decision.authorization_id or "")
        return self.get(decision.authorization_id)

    # ------------------------------------------------------------------ evidence log
    def _set_state(self, lease: Lease, state: LeaseState, reason: str) -> None:
        lease.state, lease.reason = state, reason

    def _log(self, lease: Lease, kind: str, key: str, detail: str, new: ProcessingDecision, action: ReassessmentAction) -> None:
        now = self._clock()
        row = self._db.execute("SELECT hash FROM reassessments ORDER BY seq DESC LIMIT 1").fetchone()
        prev = row[0] if row else _GENESIS
        fields = (lease.auth_id, lease.request_id, kind, key, detail, lease.decision.decision.value if action is not ReassessmentAction.CONTINUE else "n/a",
                  lease.fingerprint, new.decision.value, json.dumps([r.rule_id for r in new.results]), action.value, repr(now))
        digest = hashlib.sha256((prev + "|".join(map(str, fields))).encode("utf-8")).hexdigest()
        self._db.execute(
            "INSERT INTO reassessments(auth_id,request_id,trigger_kind,trigger_key,detail,original_decision,original_fingerprint,new_decision,new_rules,"
            "action,at,prev_hash,hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (lease.auth_id, lease.request_id, kind, key, detail, fields[5], lease.fingerprint, new.decision.value,
             json.dumps([r.rule_id for r in new.results]), action.value, now, prev, digest))

    def log(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT auth_id,request_id,trigger_kind,trigger_key,detail,original_decision,original_fingerprint,new_decision,new_rules,action,at "
                "FROM reassessments ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
        keys = ("auth_id", "request_id", "trigger", "key", "detail", "original_decision", "original_fingerprint", "new_decision", "new_rules", "action", "at")
        return [dict(zip(keys, r[:8] + (json.loads(r[8]),) + r[9:])) for r in rows]

    def verify_log(self) -> bool:
        with self._lock:
            prev = _GENESIS
            for row in self._db.execute(
                    "SELECT auth_id,request_id,trigger_kind,trigger_key,detail,original_decision,original_fingerprint,new_decision,new_rules,action,at,prev_hash,hash "
                    "FROM reassessments ORDER BY seq"):
                fields = (row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], repr(row[10]))
                if row[11] != prev or row[12] != hashlib.sha256((prev + "|".join(map(str, fields))).encode("utf-8")).hexdigest():
                    return False
                prev = row[12]
        return True

    # ------------------------------------------------------------------ housekeeping
    def _prune(self, now: float) -> None:
        dead = [a for a, l in self._leases.items() if l.state in (LeaseState.COMPLETED, LeaseState.CANCELLED, LeaseState.EXPIRED) or now >= l.expires_at]
        for auth_id in dead:
            lease = self._leases.pop(auth_id)
            for key in lease.deps:
                bucket = self._index.get(key)
                if bucket is not None:
                    bucket.discard(auth_id)
                    if not bucket:
                        del self._index[key]
