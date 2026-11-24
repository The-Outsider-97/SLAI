"""Reusable Privacy Dashboard API (transport-agnostic) for SLAI applications.

Every call takes an identity token that is verified by the IdentityAuthority. A data subject (a human whose principal id
equals the subject id, same tenant) sees and acts on their own data only; staff with a DPO/support role may act within their
own tenant. Nothing from another tenant or subject is ever returned. Results are plain JSON-safe dicts so any UI (SLAI Writer,
SReader, ...) or HTTP layer can use them; :meth:`handle` is a tiny router for HTTP-style integration.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from .privacy_ledger import LedgerError, PrivacyLedger
from .subject_rights import RightsError, SubjectRights
from .utils.identity import IdentityAuthority, IdentityVerificationError

STAFF_ROLES = ("dpo", "privacy_support")


class DashboardError(Exception):
    def __init__(self, status: int, code: str):
        super().__init__(code)
        self.status, self.code = status, code


class PrivacyDashboard:
    def __init__(self, identity: IdentityAuthority, ledger: PrivacyLedger, rights: SubjectRights, activities=None) -> None:
        self._id, self._l, self._r, self._a = identity, ledger, rights, activities

    # ------------------------------------------------------------ authn/authz
    def _authorise(self, token: Optional[str], subject_id: str):
        try:
            ident = self._id.verify(token)
        except IdentityVerificationError as exc:
            raise DashboardError(401, f"unauthenticated:{exc}") from exc
        if not ident.tenant_id:
            raise DashboardError(403, "tenant_required")
        is_staff = ident.actor_type == "human" and any(r in ident.roles for r in STAFF_ROLES)
        is_self = ident.actor_type == "human" and ident.principal_id == subject_id
        if not (is_self or is_staff):
            raise DashboardError(403, "forbidden")
        return ident

    @staticmethod
    def _verification(ident, subject_id: str, how: str) -> Dict[str, Any]:
        return {"verified": True, "method": how, "verified_by": "identity_authority", "tenant_id": ident.tenant_id, "subject_id": subject_id}

    # ------------------------------------------------------------ views
    def overview(self, token: str, subject_id: str) -> Dict[str, Any]:
        ident = self._authorise(token, subject_id)
        t = ident.tenant_id
        state = self._l.subject_state(t, subject_id)
        purposes = []
        if self._a is not None:
            purposes = [{"name": a["name"], "purposes": a["purposes"], "basis": a["basis"], "retention_days": a["retention_days"],
                         "recipients": a["recipients"]} for a in self._a.export_ropa()]
        return {"subject_id": subject_id, "consents": self._l.list_consents(t, subject_id), "restricted": state["restricted"],
                "objected_purposes": state["objected_purposes"], "erasure_requests": self._l.erasure_requests(t, subject_id),
                "rights_requests": self._r.list_for_subject(t, subject_id), "processing_activities": purposes}

    # ------------------------------------------------------------ actions
    def withdraw_consent(self, token: str, subject_id: str, purpose: str) -> Dict[str, Any]:
        ident = self._authorise(token, subject_id)
        self._l.withdraw_consent(ident.tenant_id, subject_id, purpose, evidence_ref=f"dashboard:{ident.principal_id}")
        return {"purpose": purpose, "active": False}

    def object_to_purpose(self, token: str, subject_id: str, purpose: str, objected: bool = True) -> Dict[str, Any]:
        ident = self._authorise(token, subject_id)
        self._l.set_objection(ident.tenant_id, subject_id, purpose, objected, evidence_ref=f"dashboard:{ident.principal_id}")
        return {"purpose": purpose, "objected": objected}

    def restrict(self, token: str, subject_id: str, restricted: bool = True) -> Dict[str, Any]:
        ident = self._authorise(token, subject_id)
        self._l.set_restriction(ident.tenant_id, subject_id, restricted, evidence_ref=f"dashboard:{ident.principal_id}")
        return {"restricted": restricted}

    def request_right(self, token: str, subject_id: str, kind: str, request_id: str) -> Dict[str, Any]:
        ident = self._authorise(token, subject_id)
        v = self._verification(ident, subject_id, "authenticated_session")
        if kind == "erasure":  # no reason field exists or is accepted (Art. 12(2))
            self._l.request_erasure(request_id, ident.tenant_id, subject_id, verification=v)
            return {"request_id": request_id, "kind": kind, "status": "open"}
        due = self._r.submit(request_id, ident.tenant_id, subject_id, kind, verification=v)
        return {"request_id": request_id, "kind": kind, "status": "open", "due_at": due}

    # ------------------------------------------------------------ HTTP-style router
    def handle(self, method: str, path: str, token: Optional[str], body: Optional[Mapping[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        """Routes: GET /subjects/{id}; POST /subjects/{id}/consents/{purpose}/withdraw|objection|restriction|requests."""
        body = dict(body or {})
        parts = [p for p in path.strip("/").split("/") if p]
        try:
            if len(parts) >= 2 and parts[0] == "subjects":
                sid = parts[1]
                if method == "GET" and len(parts) == 2:
                    return 200, self.overview(token or "", sid)
                if method == "POST" and parts[2:3] == ["consents"] and parts[4:5] == ["withdraw"]:
                    return 200, self.withdraw_consent(token or "", sid, parts[3])
                if method == "POST" and parts[2:] == ["objection"]:
                    return 200, self.object_to_purpose(token or "", sid, str(body.get("purpose", "")), bool(body.get("objected", True)))
                if method == "POST" and parts[2:] == ["restriction"]:
                    return 200, self.restrict(token or "", sid, bool(body.get("restricted", True)))
                if method == "POST" and parts[2:] == ["requests"]:
                    return 201, self.request_right(token or "", sid, str(body.get("kind", "")), str(body.get("request_id", "")))
            return 404, {"error": "not_found"}
        except DashboardError as exc:
            return exc.status, {"error": exc.code}
        except (LedgerError, RightsError) as exc:
            return 400, {"error": "invalid_request", "detail": str(exc)}
