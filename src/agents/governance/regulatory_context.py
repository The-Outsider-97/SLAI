"""Trusted, versioned regulatory authorization grant (assignment section 8).

An ``AuthorizationGrant`` is the portable form of a positive governance decision.
It is issued only by ``GrantAuthority`` (HMAC-SHA256, domain-separated key),
bound to policy version, tenant, principal, purpose, activity and capabilities,
expires, can be revoked (by id, subject or activity), and can only be *narrowed*
when delegated. Receivers must call ``verify``; a grant carried in a dict, header
or message that fails verification confers nothing.

The wire form is a compact ``<payload>.<signature>`` string so it survives any
interoperability protocol that can carry a string (HTTP header, JSON field,
LSTP metadata, subprocess argument).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

from .policy_registry import PolicyRegistry
from .processing_context import Decision, ProcessingContext, ProcessingDecision, new_id
from .utils.store import open_db

__all__ = ["AuthorizationGrant", "GrantAuthority", "GrantError", "GRANT_KEY_ENV", "derive_key"]

GRANT_KEY_ENV = "SLAI_GOVERNANCE_GRANT_KEY"
_MIN_KEY = 32


class GrantError(PermissionError):
    def __init__(self, code: str, message: str = ""):
        self.code = code
        super().__init__(message or code)


def derive_key(master: bytes, label: str) -> bytes:
    """Domain separation so identity and grant signatures can never be swapped."""
    return hmac.new(master, b"slai-gov-v1:" + label.encode("utf-8"), hashlib.sha256).digest()


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


@dataclass(frozen=True)
class AuthorizationGrant:
    grant_id: str
    request_id: str
    principal_id: str
    tenant_id: Optional[str]
    application_id: Optional[str]
    agent_id: Optional[str]
    controller_id: Optional[str]
    processor_id: Optional[str]
    intended_purpose: str
    processing_activity: Optional[str]
    lawful_basis: Optional[str]
    lawful_basis_evidence_ref: Optional[str]
    data_classifications: Tuple[str, ...]
    subject_references: Tuple[str, ...]
    special_category_condition: Optional[str]
    retention_policy_ref: Optional[str]
    permitted_recipients: Tuple[str, ...]
    transfer_authorization_ref: Optional[str]
    ai_risk_classification: Optional[str]
    oversight_requirement: str
    human_approval_ref: Optional[str]
    policy_version: str
    authorization_expiry: float
    provenance_ref: Optional[str]
    capabilities: Tuple[str, ...]
    delegation_depth: int
    parent_grant_id: Optional[str]
    issued_at: float

    @property
    def authorization_expiry_iso(self) -> str:
        return datetime.fromtimestamp(self.authorization_expiry, tz=timezone.utc).isoformat()

    def to_contract(self) -> Dict[str, Any]:
        """Section-8 ``processing_context`` mapping (names as in the assignment)."""
        return {
            "request_id": self.request_id,
            "principal_id": self.principal_id,
            "tenant_id": self.tenant_id,
            "application_id": self.application_id,
            "agent_id": self.agent_id,
            "controller_id": self.controller_id,
            "processor_id": self.processor_id,
            "intended_purpose": self.intended_purpose,
            "processing_activity": self.processing_activity,
            "lawful_basis": self.lawful_basis,
            "lawful_basis_evidence_ref": self.lawful_basis_evidence_ref,
            "data_classifications": list(self.data_classifications),
            "subject_references": list(self.subject_references),
            "special_category_condition": self.special_category_condition,
            "retention_policy_ref": self.retention_policy_ref,
            "permitted_recipients": list(self.permitted_recipients),
            "transfer_authorization_ref": self.transfer_authorization_ref,
            "ai_risk_classification": self.ai_risk_classification,
            "oversight_requirement": self.oversight_requirement,
            "human_approval_ref": self.human_approval_ref,
            "policy_version": self.policy_version,
            "authorization_expiry": self.authorization_expiry_iso,
            "provenance_ref": self.provenance_ref,
        }

    @classmethod
    def _from_payload(cls, d: Mapping[str, Any]) -> "AuthorizationGrant":
        tup = lambda k: tuple(d.get(k, ()))
        return cls(
            grant_id=d["grant_id"], request_id=d["request_id"], principal_id=d["principal_id"],
            tenant_id=d.get("tenant_id"), application_id=d.get("application_id"), agent_id=d.get("agent_id"),
            controller_id=d.get("controller_id"), processor_id=d.get("processor_id"),
            intended_purpose=d["intended_purpose"], processing_activity=d.get("processing_activity"),
            lawful_basis=d.get("lawful_basis"), lawful_basis_evidence_ref=d.get("lawful_basis_evidence_ref"),
            data_classifications=tup("data_classifications"), subject_references=tup("subject_references"),
            special_category_condition=d.get("special_category_condition"),
            retention_policy_ref=d.get("retention_policy_ref"), permitted_recipients=tup("permitted_recipients"),
            transfer_authorization_ref=d.get("transfer_authorization_ref"),
            ai_risk_classification=d.get("ai_risk_classification"),
            oversight_requirement=d.get("oversight_requirement", "none"),
            human_approval_ref=d.get("human_approval_ref"), policy_version=d["policy_version"],
            authorization_expiry=float(d["authorization_expiry"]), provenance_ref=d.get("provenance_ref"),
            capabilities=tup("capabilities"), delegation_depth=int(d.get("delegation_depth", 0)),
            parent_grant_id=d.get("parent_grant_id"), issued_at=float(d["issued_at"]),
        )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS grants (
    grant_id TEXT PRIMARY KEY, principal_id TEXT NOT NULL, tenant_id TEXT, activity TEXT,
    subjects TEXT NOT NULL, parent_id TEXT, expires_at REAL NOT NULL, revoked INTEGER NOT NULL DEFAULT 0,
    revoke_reason TEXT);
CREATE INDEX IF NOT EXISTS grants_activity ON grants(activity);
"""


class GrantAuthority:
    def __init__(
        self,
        policies: PolicyRegistry,
        *,
        master_key: Optional[bytes] = None,
        path: Union[str, Path] = ":memory:",
        max_ttl: float = 900.0,
        max_delegation_depth: int = 2,
        env: Optional[Mapping[str, str]] = None,
    ):
        if master_key is None:
            raw = (env if env is not None else os.environ).get(GRANT_KEY_ENV)
            master_key = raw.encode("utf-8") if raw else None
        self._key = derive_key(master_key, "grant") if master_key and len(master_key) >= _MIN_KEY else None
        self._policies = policies
        self._lock = RLock()
        self._db = open_db(path)
        self._db.executescript(_SCHEMA)
        self.max_ttl = max_ttl
        self.max_delegation_depth = max_delegation_depth

    @property
    def available(self) -> bool:
        return self._key is not None

    # ------------------------------------------------------------------ issue
    def issue(
        self,
        ctx: ProcessingContext,
        decision: ProcessingDecision,
        *,
        principal_id: str,
        lawful_basis: Optional[str] = None,
        special_category_condition: Optional[str] = None,
        retention_policy_ref: Optional[str] = None,
        transfer_authorization_ref: Optional[str] = None,
        ai_risk_classification: Optional[str] = None,
        oversight_requirement: str = "none",
        human_approval_ref: Optional[str] = None,
        ttl: Optional[float] = None,
        now: Optional[float] = None,
    ) -> str:
        if not self.available:
            raise GrantError("unavailable", "grant signing key not configured")
        if not decision.allowed:
            raise GrantError("not_authorised", f"decision is {decision.decision.value}")
        if decision.request_id != ctx.request_id or decision.context_fingerprint != ctx.fingerprint():
            raise GrantError("decision_context_mismatch")
        t = time.time() if now is None else now
        ttl = min(float(ttl if ttl is not None else self.max_ttl), self.max_ttl)
        grant = AuthorizationGrant(
            grant_id=new_id("grant"), request_id=ctx.request_id, principal_id=principal_id, tenant_id=ctx.tenant_id,
            application_id=ctx.application_id, agent_id=ctx.agent_id, controller_id=ctx.controller_id,
            processor_id=ctx.processor_id, intended_purpose=ctx.purpose, processing_activity=ctx.activity_id,
            lawful_basis=lawful_basis,
            lawful_basis_evidence_ref=f"activity:{ctx.activity_id}" if ctx.activity_id else None,
            data_classifications=(ctx.effective_data_class.value,), subject_references=tuple(ctx.subject_ids),
            special_category_condition=special_category_condition, retention_policy_ref=retention_policy_ref,
            permitted_recipients=tuple(ctx.recipients), transfer_authorization_ref=transfer_authorization_ref,
            ai_risk_classification=ai_risk_classification, oversight_requirement=oversight_requirement,
            human_approval_ref=human_approval_ref, policy_version=decision.policy_version,
            authorization_expiry=t + ttl, provenance_ref=ctx.provenance_ref,
            capabilities=(ctx.capability,) if ctx.capability else (), delegation_depth=self.max_delegation_depth,
            parent_grant_id=None, issued_at=t,
        )
        return self._seal(grant)

    # ------------------------------------------------------------------ verify
    def verify(
        self,
        token: Optional[str],
        *,
        ctx: Optional[ProcessingContext] = None,
        now: Optional[float] = None,
    ) -> AuthorizationGrant:
        if not token:
            raise GrantError("missing")
        if not self.available:
            raise GrantError("unavailable")
        try:
            body, sig = token.split(".", 1)
        except ValueError as exc:
            raise GrantError("malformed") from exc
        if not hmac.compare_digest(sig, self._sign(body)):
            raise GrantError("forged")
        try:
            grant = AuthorizationGrant._from_payload(json.loads(_b64d(body)))
        except (ValueError, KeyError, TypeError) as exc:
            raise GrantError("malformed") from exc
        t = time.time() if now is None else now
        if t >= grant.authorization_expiry:
            raise GrantError("expired")
        if grant.policy_version != self._policies.active_version():
            raise GrantError("policy_version_changed", "policy set changed since issuance; re-authorise")
        row = self._db.execute("SELECT revoked FROM grants WHERE grant_id=?", (grant.grant_id,)).fetchone()
        if row is None or row[0]:
            raise GrantError("revoked" if row else "unknown_grant")
        if grant.parent_grant_id:
            parent = self._db.execute("SELECT revoked FROM grants WHERE grant_id=?", (grant.parent_grant_id,)).fetchone()
            if parent is None or parent[0]:
                raise GrantError("parent_revoked")
        if ctx is not None:
            self._check_scope(grant, ctx)
        return grant

    @staticmethod
    def _check_scope(g: AuthorizationGrant, ctx: ProcessingContext) -> None:
        if ctx.purpose != g.intended_purpose:
            raise GrantError("purpose_mismatch")
        if ctx.tenant_id != g.tenant_id:
            raise GrantError("tenant_mismatch")
        if ctx.activity_id != g.processing_activity:
            raise GrantError("activity_mismatch")
        if ctx.agent_id and g.agent_id and ctx.agent_id != g.agent_id:
            raise GrantError("agent_mismatch")
        if g.capabilities and ctx.capability not in g.capabilities:
            raise GrantError("capability_not_granted")
        if ctx.effective_data_class.value not in g.data_classifications:
            raise GrantError("data_class_not_granted")
        if not set(ctx.subject_ids) <= set(g.subject_references):
            raise GrantError("subject_not_granted")
        if not set(ctx.recipients) <= set(g.permitted_recipients):
            raise GrantError("recipient_not_granted")

    # ------------------------------------------------------------------ delegate / revoke
    def delegate(
        self,
        token: str,
        *,
        to_agent: str,
        capabilities: Iterable[str],
        ttl: Optional[float] = None,
        now: Optional[float] = None,
    ) -> str:
        """Issue a strictly narrower grant. Scope can only shrink; depth decrements."""
        t = time.time() if now is None else now
        parent = self.verify(token, now=t)
        if parent.delegation_depth <= 0:
            raise GrantError("delegation_not_permitted")
        caps = tuple(capabilities)
        if not caps or (parent.capabilities and not set(caps) <= set(parent.capabilities)):
            raise GrantError("delegation_exceeds_scope")
        expiry = min(parent.authorization_expiry, t + float(ttl if ttl is not None else self.max_ttl))
        child_fields = {**asdict(parent), "grant_id": new_id("grant"), "agent_id": to_agent, "capabilities": caps,
                        "delegation_depth": parent.delegation_depth - 1, "parent_grant_id": parent.grant_id,
                        "authorization_expiry": expiry, "issued_at": t}
        return self._seal(AuthorizationGrant._from_payload(child_fields))

    def revoke(self, grant_id: str, reason: str) -> int:
        if not reason.strip():
            raise ValueError("reason required")
        with self._lock:
            cur = self._db.execute("UPDATE grants SET revoked=1, revoke_reason=? WHERE grant_id=? AND revoked=0", (reason, grant_id))
            return cur.rowcount

    def revoke_for_subject(self, subject_id: str, reason: str) -> int:
        """Consent withdrawal / objection / restriction propagation: kills every live grant naming the subject."""
        with self._lock:
            rows = self._db.execute("SELECT grant_id, subjects FROM grants WHERE revoked=0").fetchall()
            ids = [g for g, subj in rows if subject_id in json.loads(subj)]
            for gid in ids:
                self._db.execute("UPDATE grants SET revoked=1, revoke_reason=? WHERE grant_id=?", (reason, gid))
            return len(ids)

    def revoke_for_activity(self, activity_id: str, reason: str) -> int:
        with self._lock:
            return self._db.execute(
                "UPDATE grants SET revoked=1, revoke_reason=? WHERE activity=? AND revoked=0", (reason, activity_id)
            ).rowcount

    # ------------------------------------------------------------------ internals
    def _seal(self, g: AuthorizationGrant) -> str:
        payload = asdict(g)
        body = _b64e(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        with self._lock:
            self._db.execute(
                "INSERT INTO grants(grant_id, principal_id, tenant_id, activity, subjects, parent_id, expires_at) VALUES (?,?,?,?,?,?,?)",
                (g.grant_id, g.principal_id, g.tenant_id, g.processing_activity, json.dumps(list(g.subject_references)),
                 g.parent_grant_id, g.authorization_expiry),
            )
        return f"{body}.{self._sign(body)}"

    def _sign(self, body: str) -> str:
        assert self._key is not None
        return _b64e(hmac.new(self._key, body.encode("ascii"), hashlib.sha256).digest())
