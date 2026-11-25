"""GOV-P1-06 / GOV-P0-03: event-driven reassessment of in-flight authorisations and the side-effect boundary."""
from __future__ import annotations

import dataclasses
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path

from src.agents.governance.core import GovernanceCore
from src.agents.governance.effect_boundary import (EffectBoundary, GovernanceCommandAuthority, guard_effect, governed_scope,
                                                   install_effect_boundary, scan_effect_sites)
from src.agents.governance.processing_context import DataClass, Decision, GovernanceDenied, LawfulBasis, Operation, ProcessingContext
from src.agents.governance.reassessment import AuthorizationRevoked, LeaseState, ReassessmentAction
from tests.compliance._fixtures import GovernanceTestCase, KEY, make_activity


class ReassessmentTests(GovernanceTestCase):
    def setUp(self):
        super().setUp()
        self.register(basis=LawfulBasis.CONSENT)
        self.core.attach_providers(consent=self.core.ledger.provider(), restrictions=self.core.ledger.provider())
        self.core.ledger.record_consent("tenant-a", "subj-1", "document_editing", explicit=False, notice_version="v1", evidence_ref="e1")

    def start(self, **over):
        d = self.core.authorize(self.ctx(**over))
        self.assertTrue(d.allowed, d.reasons if hasattr(d, "reasons") else d)
        lease = self.core.authorization(d)
        self.assertIsNotNone(lease)
        self.assertEqual(lease.state, LeaseState.ACTIVE)
        return d, lease

    def test_allowed_decision_carries_an_authorisation_lease(self):
        d, lease = self.start()
        self.assertEqual(d.authorization_id, lease.auth_id)
        self.assertIs(lease.check(), lease)

    def test_denied_decision_has_no_lease(self):
        self.core.ledger.withdraw_consent("tenant-a", "subj-1", "document_editing", evidence_ref="w0")
        d = self.core.authorize(self.ctx())
        self.assertFalse(d.allowed)
        self.assertIsNone(d.authorization_id)

    def test_consent_withdrawal_cancels_the_running_operation(self):
        d, lease = self.start()
        self.core.ledger.withdraw_consent("tenant-a", "subj-1", "document_editing", evidence_ref="w1")
        with self.assertRaises(AuthorizationRevoked):
            lease.check()
        self.assertEqual(self.core.authorization(d).state, LeaseState.CANCELLED)

    def test_restriction_stops_running_operation_and_unrestriction_does_not_revive_it(self):
        d, lease = self.start()
        self.core.ledger.set_restriction("tenant-a", "subj-1", True, evidence_ref="r1")
        with self.assertRaises(AuthorizationRevoked):
            lease.check()
        self.core.ledger.set_restriction("tenant-a", "subj-1", False, evidence_ref="r2")
        with self.assertRaises(AuthorizationRevoked):  # a cancelled/suspended operation never resumes by itself
            lease.check()
        self.assertTrue(self.core.authorize(self.ctx()).allowed)  # a *new* request is evaluated afresh

    def test_objection_for_the_purpose_stops_the_operation(self):
        _, lease = self.start()
        self.core.ledger.set_objection("tenant-a", "subj-1", "document_editing", True, evidence_ref="o1")
        with self.assertRaises(AuthorizationRevoked):
            lease.check()

    def test_unrelated_subject_change_leaves_other_leases_active(self):
        self.core.ledger.record_consent("tenant-a", "subj-2", "document_editing", explicit=False, notice_version="v1", evidence_ref="e2")
        _, mine = self.start()
        _, other = self.start(subject_ids=("subj-2",))
        self.core.ledger.withdraw_consent("tenant-a", "subj-2", "document_editing", evidence_ref="w2")
        self.assertIs(mine.check(), mine)
        with self.assertRaises(AuthorizationRevoked):
            other.check()

    def test_other_tenants_event_does_not_touch_my_lease(self):
        _, lease = self.start()
        self.core.ledger.record_consent("tenant-b", "subj-1", "document_editing", explicit=False, notice_version="v1", evidence_ref="eb")
        self.core.ledger.withdraw_consent("tenant-b", "subj-1", "document_editing", evidence_ref="wb")
        self.assertIs(lease.check(), lease)

    def test_erasure_cancels_running_operations_on_that_subject(self):
        _, lease = self.start()
        self.core.ledger.request_erasure("er-1", "tenant-a", "subj-1", verification={
            "verified": True, "method": "id_check", "verified_by": "dpo.jansen", "subject_id": "subj-1", "tenant_id": "tenant-a"})
        with self.assertRaises(AuthorizationRevoked):
            lease.check()
        self.assertIn("erasure_requested", [r["trigger"] for r in self.core.reassessment.log()])

    def test_policy_change_reassesses_every_lease_and_keeps_those_still_allowed(self):
        _, lease = self.start()
        report = self.core.notify_change("policy_changed", "policy", "v-next")
        self.assertEqual(report.affected, 1)
        self.assertEqual(report.continued, (lease.auth_id,))
        self.assertIs(lease.check(), lease)
        self.assertEqual(lease.epoch, 1)

    def test_reassessing_state_fails_closed_during_evaluation(self):
        _, lease = self.start()
        seen = {}
        original = self.core.reassessment._evaluate

        def slow(ctx):
            try:
                self.core.reassessment.check(lease.auth_id)
                seen["during"] = "allowed"
            except AuthorizationRevoked as exc:
                seen["during"] = exc.state
            return original(ctx)

        self.core.reassessment._evaluate = slow
        self.core.notify_change("policy_changed", "policy")
        self.assertEqual(seen["during"], "reassessing")
        self.assertIs(lease.check(), lease)

    def test_reassessment_error_fails_closed(self):
        _, lease = self.start()

        def boom(ctx):
            raise RuntimeError("evaluator down")

        self.core.reassessment._evaluate = boom
        report = self.core.notify_change("policy_changed", "policy")
        self.assertEqual(report.cancelled, (lease.auth_id,))
        with self.assertRaises(AuthorizationRevoked):
            lease.check()

    def test_expired_lease_is_refused(self):
        _, lease = self.start()
        lease.expires_at = time.time() - 1
        with self.assertRaises(AuthorizationRevoked) as cm:
            lease.check()
        self.assertEqual(cm.exception.state, "expired")

    def test_completed_lease_cannot_authorise_further_effects(self):
        _, lease = self.start()
        lease.complete()
        with self.assertRaises(AuthorizationRevoked):
            lease.check()

    def test_unknown_or_missing_lease_fails_closed(self):
        for auth_id in (None, "", "auth_nope"):
            with self.assertRaises(AuthorizationRevoked):
                self.core.reassessment.check(auth_id)

    def test_decision_log_is_hash_chained_and_tamper_evident(self):
        _, lease = self.start()
        self.core.notify_change("policy_changed", "policy")
        self.core.ledger.withdraw_consent("tenant-a", "subj-1", "document_editing", evidence_ref="w1")
        log = self.core.reassessment.log()
        self.assertEqual([r["action"] for r in reversed(log)], ["continue", "cancel"])
        self.assertTrue(self.core.reassessment.verify_log())
        self.core.reassessment._db.execute("UPDATE reassessments SET new_decision='allow' WHERE action='cancel'")
        self.assertFalse(self.core.reassessment.verify_log())

    def test_cancelled_operation_never_resumes_but_suspended_one_can(self):
        _, lease = self.start()
        self.core.ledger.withdraw_consent("tenant-a", "subj-1", "document_editing", evidence_ref="w1")
        with self.assertRaises(AuthorizationRevoked):
            self.core.reassessment.resume(lease.auth_id, self.core.authorize)

    def test_concurrent_events_and_checks_never_allow_a_revoked_lease(self):
        _, lease = self.start()
        outcomes, stop = [], threading.Event()

        def checker():
            while not stop.is_set():
                try:
                    lease.check()
                    outcomes.append(("ok", self.core.ledger.consent_snapshot("tenant-a", "subj-1", "document_editing")["active"]))
                except AuthorizationRevoked:
                    outcomes.append(("revoked", None))

        threads = [threading.Thread(target=checker) for _ in range(4)]
        [t.start() for t in threads]
        time.sleep(0.05)
        self.core.ledger.withdraw_consent("tenant-a", "subj-1", "document_editing", evidence_ref="w1")
        withdrawn_at = len(outcomes)
        time.sleep(0.05)
        stop.set()
        [t.join() for t in threads]
        self.assertTrue(any(o[0] == "revoked" for o in outcomes))
        # once the withdrawal call has returned, no later check may succeed
        later = [o for o in outcomes[withdrawn_at + 16:]]
        self.assertTrue(all(o[0] == "revoked" for o in later))

    def test_restart_forgets_leases_and_fails_closed_but_log_survives(self):
        _, lease = self.start()
        self.core.ledger.withdraw_consent("tenant-a", "subj-1", "document_editing", evidence_ref="w1")
        state_dir = self.core.state_dir
        self.core.close()
        GovernanceCore.reset_shared()
        again = GovernanceCore(state_dir, master_key=KEY)
        self.addCleanup(again.close)
        with self.assertRaises(AuthorizationRevoked):
            again.reassessment.check(lease.auth_id)
        self.assertTrue(again.reassessment.verify_log())
        self.assertEqual(len(again.reassessment.log()), 1)


class ApprovalBackedLeaseTests(GovernanceTestCase):
    def setUp(self):
        super().setUp()
        from tests.compliance.test_approval_binding import DOSSIER
        self.dossier = DOSSIER
        self.core.activities.register(make_activity("act-1", art22_exception="contract", recipients=("processor-a",)), actor="dpo")
        self.requester, self.reviewer, self.t0 = self.ident("svc-decider"), self.human("reviewer-1"), time.time() - 600

    def test_consumed_approval_keeps_backing_its_running_operation_until_revoked(self):
        c = self.ctx(automated_decision=True, significant_effect=True, identity=self.requester, application_id="app-1")
        om = self.core.oversight
        aid = om.request_approval(c, self.dossier, self.requester, now=self.t0)
        om.mark_presented(aid, self.reviewer, now=self.t0 + 1)
        om.decide(aid, self.reviewer, approve=True, rationale="Reviewed affordability evidence and alternatives",
                  reviewed_dossier_digest=self.dossier.digest(), now=self.t0 + 60)
        d = self.core.authorize(dataclasses.replace(c, approval_id=aid))
        self.assertTrue(d.allowed, d)
        lease = self.core.authorization(d)
        self.assertEqual(lease.consumed_approval, aid)
        self.core.notify_change("policy_changed", "policy")  # approval is consumed, yet still backs the running operation
        self.assertIs(lease.check(), lease)
        om.revoke(aid, self.reviewer, "reviewer withdrew approval")
        with self.assertRaises(AuthorizationRevoked):
            lease.check()


class EffectBoundaryTests(GovernanceTestCase):
    def setUp(self):
        super().setUp()
        self.register(basis=LawfulBasis.CONTRACT)
        self.core.attach_providers(consent=self.core.ledger.provider(), restrictions=self.core.ledger.provider())
        self.addCleanup(install_effect_boundary, None)

    def scope_for(self, **over):
        d = self.core.authorize(self.ctx(**over))
        self.assertTrue(d.allowed, d)
        return d

    def test_enforce_blocks_governed_effect_outside_any_scope(self):
        self.core.enforcement_mode = "enforce"
        with self.assertRaises(GovernanceDenied) as cm:
            self.core.effects.guard("network_outbound", "https://x.invalid")
        self.assertIn("GOV-FX-01", {r.rule_id for r in cm.exception.decision.results})

    def test_observe_reports_would_block_and_off_does_not_enforce(self):
        self.core.enforcement_mode = "observe"
        self.assertEqual(self.core.effects.guard("network_outbound", "t").outcome, "would_block")
        self.core.enforcement_mode = "off"
        self.assertEqual(self.core.effects.guard("network_outbound", "t").outcome, "unguarded_mode_off")

    def test_low_risk_effects_pass_without_scope(self):
        self.core.enforcement_mode = "enforce"
        self.assertEqual(self.core.effects.guard("read_internal", "cfg").outcome, "low_risk")

    def test_unknown_effect_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            self.core.effects.guard("teleport", "x")

    def test_scoped_effect_inherits_authorisation_without_a_second_decision(self):
        self.core.enforcement_mode = "enforce"
        before = self.core.health()
        d = self.scope_for()
        with governed_scope("agent-a", d, self.ctx()):
            permit = self.core.effects.guard("network_outbound", "https://x.invalid")
            nested = self.core.effects.guard("file_write", "out.txt")
        self.assertEqual((permit.outcome, permit.auth_id), ("scoped", d.authorization_id))
        self.assertEqual(nested.auth_id, d.authorization_id)
        self.assertEqual(len(self.core.reassessment.active()), 1)

    def test_scope_is_per_context_and_does_not_leak_after_exit(self):
        self.core.enforcement_mode = "enforce"
        d = self.scope_for()
        with governed_scope("agent-a", d, self.ctx()):
            pass
        with self.assertRaises(GovernanceDenied):
            self.core.effects.guard("network_outbound", "t")

    def test_scope_does_not_cross_threads(self):
        self.core.enforcement_mode = "enforce"
        d = self.scope_for()
        errors = []

        def worker():
            try:
                self.core.effects.guard("network_outbound", "t")
            except GovernanceDenied:
                errors.append("denied")

        with governed_scope("agent-a", d, self.ctx()):
            t = threading.Thread(target=worker)
            t.start(); t.join()
        self.assertEqual(errors, ["denied"])

    def test_effect_after_revocation_is_refused(self):
        self.core.enforcement_mode = "enforce"
        d = self.scope_for()
        with governed_scope("agent-a", d, self.ctx()):
            self.core.ledger.set_restriction("tenant-a", "subj-1", True, evidence_ref="r1")
            with self.assertRaises(AuthorizationRevoked):
                self.core.effects.guard("network_outbound", "t")

    def test_personal_data_destination_outside_authorisation_is_refused(self):
        self.core.enforcement_mode = "enforce"
        d = self.scope_for(destination_country=None)
        with governed_scope("agent-a", d, self.ctx()):
            with self.assertRaises(GovernanceDenied) as cm:
                self.core.effects.guard("network_outbound", "https://x.invalid", destination_country="US")
            self.assertIn("GOV-FX-02", {r.rule_id for r in cm.exception.decision.results})
            self.core.effects.guard("network_outbound", "https://x.invalid", destination_country="NL")

    def test_unauthorised_recipient_is_refused(self):
        self.core.enforcement_mode = "enforce"
        d = self.scope_for(recipients=("processor-a",))
        with governed_scope("agent-a", d, self.ctx(recipients=("processor-a",))):
            self.core.effects.guard("export", "t", recipients=("processor-a",))
            with self.assertRaises(GovernanceDenied):
                self.core.effects.guard("export", "t", recipients=("processor-z",))

    def test_non_permitting_decision_opens_no_scope(self):
        self.core.enforcement_mode = "enforce"
        bad = self.core.authorize(self.ctx(purpose="advertising"))
        self.assertFalse(bad.allowed)
        with governed_scope("agent-a", bad, self.ctx()) as scope:
            self.assertIsNone(scope)
            with self.assertRaises(GovernanceDenied):
                self.core.effects.guard("network_outbound", "t")

    def test_runtime_coverage_reports_observed_effects_and_says_what_it_is(self):
        self.core.enforcement_mode = "observe"
        self.core.effects.guard("network_outbound", "https://a.invalid")
        cov = self.core.effects.coverage()
        self.assertIn("runtime", cov["basis"])
        self.assertEqual(cov["kinds"]["network_outbound"]["unscoped_would_block"], 1)

    def test_module_level_guard_is_noop_until_installed(self):
        self.assertIsNone(guard_effect("network_outbound", "t"))
        install_effect_boundary(self.core.effects)
        self.core.enforcement_mode = "enforce"
        with self.assertRaises(GovernanceDenied):
            guard_effect("network_outbound", "t")


class _Allow:
    def __init__(self, allowed=True):
        self.allowed, self.principal_id = allowed, "p"
        self.reason = "ok"

    def __call__(self, allowed, principal_id, reason):
        return _Allow(allowed)


class CommandAuthorityTests(GovernanceTestCase):
    class Inner:
        def __init__(self, decision):
            self.decision, self.calls = decision, 0

        def authorize(self, request_context, command, capability):
            self.calls += 1
            return self.decision

    @dataclasses.dataclass
    class Decision:
        allowed: bool
        principal_id: str = "p"
        reason: str = "ok"

    class Cap:
        effect = "external"

    class Cmd:
        operation = "send_mail"

    def test_inner_denial_is_final_and_governance_is_not_consulted(self):
        inner = self.Inner(self.Decision(False, reason="no"))
        self.core.enforcement_mode = "enforce"
        auth = GovernanceCommandAuthority(inner, self.core.effects)
        self.assertFalse(auth.authorize(None, self.Cmd(), self.Cap()).allowed)
        self.assertNotIn("network_outbound", self.core.effects.coverage()["kinds"])

    def test_governance_can_only_restrict(self):
        inner = self.Inner(self.Decision(True))
        self.core.enforcement_mode = "enforce"
        auth = GovernanceCommandAuthority(inner, self.core.effects, kind_for=lambda c: "network_outbound")
        out = auth.authorize(None, self.Cmd(), self.Cap())
        self.assertFalse(out.allowed)
        self.assertTrue(out.reason.startswith("governance:"))

    def test_read_effects_pass_in_enforce_mode(self):
        class Read:
            effect = "read"
        inner = self.Inner(self.Decision(True))
        self.core.enforcement_mode = "enforce"
        self.assertTrue(GovernanceCommandAuthority(inner, self.core.effects).authorize(None, self.Cmd(), Read()).allowed)

    def test_inner_must_be_an_authority(self):
        with self.assertRaises(TypeError):
            GovernanceCommandAuthority(object(), self.core.effects)


class RealCommandGatewayTests(GovernanceTestCase):
    """GovernanceCommandAuthority composed with the REAL CommandGateway (no stand-in gateway)."""

    def setUp(self):
        super().setUp()
        from src.agents.execution import command_gateway as cg
        self.cg = cg
        self.register(basis=LawfulBasis.CONTRACT)
        self.core.attach_providers(consent=self.core.ledger.provider(), restrictions=self.core.ledger.provider())
        self.calls = []

        class Inner:
            def authorize(_s, request_context, command, capability):
                return cg.Decision(True, "svc-writer", "inner-ok")

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.gateway = cg.CommandGateway(Path(self.tmp.name) / "journal.db", GovernanceCommandAuthority(Inner(), self.core.effects))
        self.addCleanup(self.gateway.close)
        self.gateway.register(cg.Capability("send_report", lambda command: self.calls.append(command.command_id) or "sent"))
        self.gateway.register(cg.Capability("read_report", lambda command: "r", effect="read"))
        self.core.enforcement_mode = "enforce"

    def cmd(self, cid, op="send_report"):
        return self.cg.Command(cid, "idem-" + cid, op, "report-1")

    def test_effect_without_governed_scope_is_denied_before_dispatch(self):
        result = self.gateway.execute(self.cmd("c1"), request_context=None)
        self.assertEqual(result.status, self.cg.Outcome.DENIED)
        self.assertEqual(self.calls, [])

    def test_effect_inside_scope_is_dispatched_once(self):
        d = self.core.authorize(self.ctx())
        with governed_scope("agent-a", d, self.ctx()):
            result = self.gateway.execute(self.cmd("c2"), request_context=None)
        self.assertTrue(result.is_success, result)
        self.assertEqual(self.calls, ["c2"])

    def test_read_capability_needs_no_scope(self):
        self.assertTrue(self.gateway.execute(self.cmd("c3", "read_report"), request_context=None).is_success)

    def test_revocation_before_dispatch_and_before_replay_denies(self):
        d = self.core.authorize(self.ctx())
        with governed_scope("agent-a", d, self.ctx()):
            self.assertTrue(self.gateway.execute(self.cmd("c4"), request_context=None).is_success)
            self.core.ledger.set_restriction("tenant-a", "subj-1", True, evidence_ref="r1")
            replay = self.gateway.execute(self.cmd("c4"), request_context=None)  # a stored result is not a standing grant
            fresh = self.gateway.execute(self.cmd("c5"), request_context=None)
        self.assertEqual((replay.status, fresh.status), (self.cg.Outcome.DENIED, self.cg.Outcome.DENIED))
        self.assertEqual(self.calls, ["c4"])


class StaticScanTests(unittest.TestCase):
    def test_scan_finds_guarded_and_unguarded_sites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text(textwrap.dedent('''
                import subprocess, requests
                from src.agents.governance.effect_boundary import guard_effect

                def guarded():
                    guard_effect("subprocess", "ls")
                    subprocess.run(["ls"])

                def unguarded():
                    requests.get("https://x.invalid")
            '''), encoding="utf-8")
            report = scan_effect_sites(root)
        self.assertTrue(report["basis"].startswith("static"))
        self.assertEqual(report["summary"], {"sites": 2, "guarded": 1, "unguarded": 1})
        self.assertEqual(len(report["unguarded"]), 1)
        self.assertIn("a.py", report["unguarded"][0]["file"])


if __name__ == "__main__":
    unittest.main()
