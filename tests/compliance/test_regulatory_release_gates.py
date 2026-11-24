import hashlib
import sqlite3
import unittest

from src.agents.governance.compliance_engine import DEFAULT_CONTROLS, ComplianceEngine, ControlStatus, DeploymentProfile
from src.agents.governance.regulatory_evidence import EvidenceError
from src.agents.governance.regulatory_reporting import RegulatoryReporter
from tests.compliance._fixtures import GovernanceTestCase

DAY = 86400.0


def digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class ReleaseGateTests(GovernanceTestCase):
    def setUp(self):
        super().setUp()
        self.ci = self.core.identity.issue("ci_pipeline", None, ttl_seconds=3600)
        self.gov = self.core.identity.issue("governance_agent", None, ttl_seconds=3600)
        self.profile = DeploymentProfile(name="test")

    def add(self, control, etype, *, token=None, result="pass", now=None):
        return self.core.evidence.append(producer_token=token or self.ci, control_id=control, evidence_type=etype,
                                         artifact_digest=digest(f"{control}{etype}{result}"), artifact_ref=f"ci://run/{control}/{etype}", result=result, now=now)

    def fill_technical(self, **kw):
        for c in DEFAULT_CONTROLS:
            if c.category == "technical" and c.applies(self.profile):
                for req in c.requirements:
                    self.add(c.control_id, req.evidence_type, **kw)

    def assess(self, profile=None):
        return self.core.compliance.assess(profile or self.profile)

    def test_missing_evidence_is_not_compliance(self):
        a = self.assess()
        self.assertFalse(a.release_allowed())
        for r in a.results:
            if r.control.category == "technical" and r.status is not ControlStatus.NOT_APPLICABLE:
                self.assertEqual(r.status, ControlStatus.MISSING, r.control.control_id)

    def test_independent_complete_evidence_allows_software_release_but_not_deployment(self):
        self.fill_technical()
        a = self.assess()
        self.assertTrue(a.release_allowed(), [r.to_dict() for r in a.blocking(include_organisational=False)])
        self.assertFalse(a.release_allowed(include_organisational=True))  # DPA/notices/ROPA/literacy are organisational
        self.assertTrue(any(r.status is ControlStatus.DEPENDS_ON_ORGANISATION for r in a.results))

    def test_governance_agent_cannot_self_certify(self):
        self.fill_technical(token=self.gov)
        a = self.assess()
        self.assertFalse(a.release_allowed())
        self.assertTrue(all(r.status is ControlStatus.UNVERIFIED for r in a.results if r.control.control_id == "TC-LAW-01"))

    def test_one_failed_mandatory_control_blocks_release_regardless_of_the_rest(self):
        self.fill_technical()
        self.add("TC-ISO-05", "isolation_test", result="fail")
        a = self.assess()
        self.assertFalse(a.release_allowed())
        self.assertEqual(next(r for r in a.results if r.control.control_id == "TC-ISO-05").status, ControlStatus.FAILED)
        ok, reasons = self.core.compliance.release_gate(self.profile)
        self.assertFalse(ok)
        self.assertIn("TC-ISO-05:failed", reasons)

    def test_latest_result_wins_so_a_fix_can_restore_the_control(self):
        self.fill_technical()
        self.add("TC-ISO-05", "isolation_test", result="fail")
        self.add("TC-ISO-05", "isolation_test", result="pass")
        self.assertTrue(self.assess().release_allowed())

    def test_stale_evidence_is_not_current(self):
        import time
        self.fill_technical(now=time.time() - 400 * DAY)
        a = self.assess()
        self.assertFalse(a.release_allowed())
        self.assertTrue(any("stale" in f for r in a.results for f in r.findings))

    def test_tampered_evidence_chain_invalidates_everything(self):
        self.fill_technical()
        self.assertTrue(self.assess().release_allowed())
        db = self.core.evidence._db
        db.execute("UPDATE evidence SET result='pass', control_id='TC-LAW-01' WHERE seq=2")
        db.execute("UPDATE evidence SET summary='{\"x\": 1}' WHERE seq=1")
        a = self.assess()
        self.assertFalse(a.chain_ok)
        self.assertFalse(a.release_allowed())

    def test_deleted_evidence_is_detected(self):
        self.fill_technical()
        self.core.evidence._db.execute("DELETE FROM evidence WHERE seq=3")
        self.assertFalse(self.assess().chain_ok)

    def test_evidence_requires_authenticated_producer_and_metadata_only(self):
        with self.assertRaises(EvidenceError):
            self.core.evidence.append(producer_token="forged.token", control_id="TC-LAW-01", evidence_type="test_run",
                                      artifact_digest=digest("x"), artifact_ref="ci://x", result="pass")
        with self.assertRaises(EvidenceError):
            self.core.evidence.append(producer_token=self.ci, control_id="TC-LAW-01", evidence_type="test_run", artifact_digest=digest("x"),
                                      artifact_ref="ci://x", result="pass", summary={"payload": {"name": "Jan"}})
        with self.assertRaises(EvidenceError):
            self.core.evidence.append(producer_token=self.ci, control_id="TC-LAW-01", evidence_type="made_up", artifact_digest=digest("x"), artifact_ref="ci://x", result="pass")

    def test_not_applicable_requires_profile_and_is_reported_with_justification(self):
        a = self.assess(DeploymentProfile(name="calc-only", processes_personal_data=False, justification={"TC-TRF-12": "no personal data processed"}))
        r = next(x for x in a.results if x.control.control_id == "TC-TRF-12")
        self.assertEqual(r.status, ControlStatus.NOT_APPLICABLE)
        self.assertIn("no personal data", r.findings[0])

    def test_high_risk_deployment_adds_controls(self):
        hr = DeploymentProfile(name="hr", high_risk_ai_in_scope=True)
        ids = {r.control.control_id for r in self.assess(hr).results if r.status is not ControlStatus.NOT_APPLICABLE}
        self.assertTrue({"TC-HR-16", "ORG-CONF", "ORG-DPIA"} <= ids)

    def test_report_has_no_scores_and_never_claims_compliance_without_evidence(self):
        report = RegulatoryReporter().report(self.assess())
        flat = str(report).lower()
        self.assertNotIn("overall_score", flat)
        self.assertFalse(report["software_release_allowed"])
        self.assertIn("not legal advice", report["disclaimer"])
        for row in report["traceability"]:
            self.assertNotEqual(row["reported_status"], ControlStatus.VERIFIED.value, row["article"])  # no evidence yet

    def test_traceability_status_is_computed_from_evidence_not_copied(self):
        self.fill_technical()
        report = RegulatoryReporter().report(self.assess())
        by_article = {r["article"]: r for r in report["traceability"]}
        self.assertEqual(by_article["GDPR Art. 6"]["reported_status"], ControlStatus.VERIFIED.value)
        self.assertEqual(by_article["GDPR Art. 5(1)(a)-(b)"]["reported_status"], "partial")  # declared partial: evidence cannot upgrade it
        self.assertEqual(by_article["GDPR Art. 12-14"]["reported_status"], ControlStatus.DEPENDS_ON_ORGANISATION.value)
        self.assertEqual(by_article["GDPR Art. 15-16"]["reported_status"], "partial")  # workflow implemented; no store registers a RightsProvider yet

    def test_health_is_truthful_about_missing_dependencies(self):
        self.assertTrue(self.core.health()["ready"])
        self.core.engine.consent = None
        self.core._consent = None
        h = self.core.health()
        self.assertFalse(h["ready"])
        self.assertIn("consent_provider", h["problems"])
        self.assertTrue(h["enforcing"])


if __name__ == "__main__":
    unittest.main()


class SafetyCheckerDoesNotClaimLegalCompliance(unittest.TestCase):
    """Audit 4.10: the operational score must carry an explicit not-determined legal status."""

    def test_source_marks_legal_compliance_not_determined(self):
        from pathlib import Path

        text = (Path(__file__).resolve().parents[2] / "src" / "agents" / "safety" / "compliance_checker.py").read_text(encoding="utf-8")
        self.assertIn('"legal_compliance": "not_determined"', text)
        self.assertIn("NOT a legal compliance determination", text)
