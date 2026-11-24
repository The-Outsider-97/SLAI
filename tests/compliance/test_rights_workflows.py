"""GDPR Art. 12/15/16/20 workflows and the Privacy Dashboard API."""
from __future__ import annotations

import unittest

from src.agents.governance.privacy_dashboard import PrivacyDashboard
from src.agents.governance.privacy_ledger import PrivacyLedger
from src.agents.governance.subject_rights import DAY, RightsError, SubjectRights
from tests.compliance._fixtures import GovernanceTestCase


class Docs:
    name, portable = "documents", True

    def __init__(self):
        self.rows = {("t1", "s1"): {"title": "My letter"}, ("t2", "s1"): {"title": "OTHER TENANT"}, ("t1", "s2"): {"title": "OTHER SUBJECT"}}

    def export(self, tenant_id, subject_id):
        return dict(self.rows.get((tenant_id, subject_id), {}))


class Telemetry:
    name, portable = "telemetry", False

    def export(self, tenant_id, subject_id):
        return {"events": 3}


class Broken:
    name, portable = "broken", True

    def export(self, tenant_id, subject_id):
        raise RuntimeError("store down")


def ver(t="t1", s="s1"):
    return {"verified": True, "method": "session", "verified_by": "idp", "tenant_id": t, "subject_id": s}


class RightsTests(unittest.TestCase):
    def setUp(self):
        self.now = [1_000_000.0]
        self.r = SubjectRights(":memory:", clock=lambda: self.now[0], processing_info=lambda t: {"purposes": ["document_editing"], "retention_days": 30})
        self.r.register_provider(Docs())
        self.r.register_provider(Telemetry())

    def test_verification_is_mandatory_and_bound(self):
        with self.assertRaises(RightsError):
            self.r.submit("a", "t1", "s1", "access", verification={})
        with self.assertRaises(RightsError):
            self.r.submit("a", "t1", "s1", "access", verification=ver("t2", "s1"))
        with self.assertRaises(RightsError):
            self.r.submit("a", "t1", "s1", "bogus", verification=ver())

    def test_access_returns_only_the_subjects_data_with_art15_information(self):
        self.r.submit("a", "t1", "s1", "access", verification=ver())
        out = self.r.fulfil("a")
        b = out["bundle"]
        self.assertEqual(b["sections"]["documents"], {"title": "My letter"})
        self.assertNotIn("OTHER", str(b))
        self.assertEqual(b["processing_information"]["purposes"], ["document_editing"])
        self.assertEqual(self.r.status("a")["status"], "completed")
        with self.assertRaises(RightsError):
            self.r.fulfil("a")  # not retained, not re-released

    def test_portability_contains_only_portable_providers(self):
        self.r.submit("p", "t1", "s1", "portability", verification=ver())
        b = self.r.fulfil("p")["bundle"]
        self.assertEqual(list(b["sections"]), ["documents"])
        self.assertEqual(b["format"], "application/json")

    def test_provider_failure_releases_nothing_and_stays_incomplete(self):
        self.r.register_provider(Broken())
        self.r.submit("a", "t1", "s1", "access", verification=ver())
        with self.assertRaises(RightsError):
            self.r.fulfil("a")
        self.assertEqual(self.r.status("a")["status"], "incomplete")

    def test_rectification_records_hashes_not_values(self):
        self.r.submit("r", "t1", "s1", "rectification", verification=ver())
        seen = {}

        def apply(t, s, changes):
            seen.update(t=t, s=s, c=changes)
            return {"updated": 1}

        res = self.r.rectify("r", {"email": "new@example.nl"}, apply=apply)
        self.assertEqual(seen["t"], "t1")
        self.assertEqual(res["fields"], ["email"])
        self.assertNotIn("new@example.nl", str(res))
        self.assertEqual(self.r.status("r")["status"], "completed")

    def test_rectification_failure_keeps_request_open(self):
        self.r.submit("r", "t1", "s1", "rectification", verification=ver())
        with self.assertRaises(RightsError):
            self.r.rectify("r", {"x": 1}, apply=lambda *a: (_ for _ in ()).throw(RuntimeError("down")))
        self.assertEqual(self.r.status("r")["status"], "incomplete")

    def test_deadline_extension_rules_and_overdue(self):
        due = self.r.submit("a", "t1", "s1", "access", verification=ver())
        self.assertEqual(due, 1_000_000.0 + 30 * DAY)
        self.now[0] = due + 1
        self.assertEqual(self.r.overdue(), ["a"])
        with self.assertRaises(RightsError):
            self.r.extend("a", reason="complex", subject_notified=False)
        with self.assertRaises(RightsError):
            self.r.extend("a", reason="", subject_notified=True)
        new_due = self.r.extend("a", reason="large volume of requests", subject_notified=True)
        self.assertEqual(new_due, due + 60 * DAY)
        with self.assertRaises(RightsError):
            self.r.extend("a", reason="again", subject_notified=True)


class DashboardTests(GovernanceTestCase):
    def setUp(self):
        super().setUp()
        self.ledger = PrivacyLedger(":memory:", tombstone_salt=b"s" * 32)
        self.rights = SubjectRights(":memory:")
        self.rights.register_provider(Docs())
        self.register(activity_id="act-1")
        self.dash = PrivacyDashboard(self.core.identity, self.ledger, self.rights, self.core.activities)

    def token(self, principal, tenant="t1", actor="human", roles=()):
        return self.core.identity.issue(principal, tenant, actor_type=actor, roles=roles, ttl_seconds=600)

    def test_subject_sees_only_self_and_unauthenticated_is_refused(self):
        self.ledger.record_consent("t1", "s1", "document_editing", explicit=False, notice_version="v1", evidence_ref="e")
        ok = self.dash.handle("GET", "/subjects/s1", self.token("s1"))
        self.assertEqual(ok[0], 200)
        self.assertTrue(ok[1]["consents"][0]["active"])
        self.assertEqual(ok[1]["processing_activities"][0]["purposes"], ["document_editing"])
        self.assertEqual(self.dash.handle("GET", "/subjects/s1", self.token("s2"))[0], 403)
        self.assertEqual(self.dash.handle("GET", "/subjects/s1", self.token("s1", tenant="t2"))[1]["consents"], [])
        self.assertEqual(self.dash.handle("GET", "/subjects/s1", None)[0], 401)
        self.assertEqual(self.dash.handle("GET", "/subjects/s1", "forged.token")[0], 401)

    def test_service_accounts_cannot_use_subject_endpoints(self):
        self.assertEqual(self.dash.handle("GET", "/subjects/svc", self.token("svc", actor="service"))[0], 403)

    def test_staff_act_within_their_tenant_only(self):
        staff = self.token("dpo-1", roles=("dpo",))
        self.assertEqual(self.dash.handle("GET", "/subjects/s1", staff)[0], 200)
        self.ledger.record_consent("t1", "s1", "p", explicit=False, notice_version="v1", evidence_ref="e")
        other = self.token("dpo-2", tenant="t2", roles=("dpo",))
        self.assertEqual(self.dash.handle("GET", "/subjects/s1", other)[1]["consents"], [])

    def test_withdraw_objection_restriction_take_effect_in_the_ledger(self):
        t = self.token("s1")
        self.ledger.record_consent("t1", "s1", "document_editing", explicit=False, notice_version="v1", evidence_ref="e")
        self.assertEqual(self.dash.handle("POST", "/subjects/s1/consents/document_editing/withdraw", t)[0], 200)
        self.assertFalse(self.ledger.consent_snapshot("t1", "s1", "document_editing")["active"])
        self.dash.handle("POST", "/subjects/s1/objection", t, {"purpose": "marketing"})
        self.dash.handle("POST", "/subjects/s1/restriction", t, {"restricted": True})
        st = self.ledger.subject_state("t1", "s1")
        self.assertTrue(st["restricted"])
        self.assertEqual(st["objected_purposes"], ["marketing"])

    def test_requests_need_no_reason_and_are_tenant_bound(self):
        t = self.token("s1")
        code, body = self.dash.handle("POST", "/subjects/s1/requests", t, {"kind": "access", "request_id": "a1", "reason": "ignored"})
        self.assertEqual(code, 201)
        code, _ = self.dash.handle("POST", "/subjects/s1/requests", t, {"kind": "erasure", "request_id": "e1"})
        self.assertEqual(code, 201)
        self.assertTrue(self.ledger.subject_state("t1", "s1")["erasure_pending"])
        self.assertEqual(self.dash.handle("POST", "/subjects/s1/requests", t, {"kind": "nonsense", "request_id": "x"})[0], 400)
        self.assertEqual(self.dash.handle("GET", "/nope", t)[0], 404)


if __name__ == "__main__":
    unittest.main()
