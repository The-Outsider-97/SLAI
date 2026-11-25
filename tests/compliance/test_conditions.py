"""GOV-P1-01: ALLOW_WITH_CONTROLS is a tracked set of obligations (pre / runtime / release / post), not a free-text hint."""
from __future__ import annotations

import dataclasses
import time
import unittest

from src.agents.governance.agent_gate import AgentGate
from src.agents.governance.conditions import ConditionError, ConditionsUnmet, Phase, Status
from src.agents.governance.core import GovernanceCore
from src.agents.governance.effect_boundary import governed_scope
from src.agents.governance.human_oversight import ApprovalStatus
from src.agents.governance.policy_engine import _r
from src.agents.governance.processing_context import Decision, GovernanceDenied, LawfulBasis, Operation
from src.agents.governance.reassessment import AuthorizationRevoked
from tests.compliance._fixtures import GovernanceTestCase, KEY, make_activity
from tests.compliance.test_approval_binding import DOSSIER


def controls_rule(*controls, when=lambda ctx: ctx.purpose == "document_editing"):
    def rule(ctx, env):
        return [_r("GOV-TEST-CTL", Decision.ALLOW_WITH_CONTROLS, "test_controls", "test", controls=controls)] if when(ctx) else []
    return rule


class ConditionTestCase(GovernanceTestCase):
    def setUp(self):
        super().setUp()
        self.register(basis=LawfulBasis.CONTRACT)
        self.requester = self.ident("svc-requester")
        self.dpo = self.human("dpo-1", roles=("dpo",))
        self.reviewer = self.human("reviewer-1", roles=("oversight_reviewer",))

    def add(self, *controls):
        self.core.engine.add_rule(controls_rule(*controls))

    def request(self, **over):
        return self.ctx(identity=self.requester, **over)

    def only_pending(self):
        pending = self.core.conditions.pending()
        self.assertEqual(len(pending), 1, pending)
        return pending[0]


class PreConditionTests(ConditionTestCase):
    def test_unknown_control_fails_closed(self):
        self.add("mystery_control")
        d = self.core.authorize(self.request())
        self.assertEqual(d.decision, Decision.BLOCK)
        self.assertIn("GOV-CND-01", self.rule_ids(d))
        self.assertIsNone(d.authorization_id)

    def test_unmet_mandatory_pre_condition_blocks_before_any_lease_exists(self):
        self.add("apply_supplementary_measures")
        d = self.core.authorize(self.request())
        self.assertEqual(d.decision, Decision.REVIEW_REQUIRED)
        self.assertEqual(self.rule_ids(d), {"GOV-CND-02"})
        self.assertIsNone(d.authorization_id)
        self.assertEqual(self.only_pending().condition, "apply_supplementary_measures")

    def test_attestation_satisfies_exactly_this_request_binding(self):
        self.add("apply_supplementary_measures")
        c = self.request()
        self.core.authorize(c)
        ob = self.only_pending()
        self.core.conditions.satisfy(ob.obligation_id, self.dpo, "tia-annex-3", "encryption + key custody evidenced")
        self.assertTrue(self.core.authorize(c).allowed)
        # a different request (other resource) is a different binding: the attestation does not carry over
        other = self.core.authorize(dataclasses.replace(c, resource="another-resource"))
        self.assertEqual(other.decision, Decision.REVIEW_REQUIRED)

    def test_requester_cannot_attest_its_own_condition(self):
        self.add("apply_supplementary_measures")
        self.core.authorize(self.ctx(identity=self.dpo))
        with self.assertRaises(ConditionError):
            self.core.conditions.satisfy(self.only_pending().obligation_id, self.dpo, "ref-1")

    def test_only_a_verified_human_owner_can_attest(self):
        self.add("apply_supplementary_measures")
        self.core.authorize(self.request())
        oid = self.only_pending().obligation_id
        service_with_role = self.ident("svc-x", roles=("dpo",))  # a service, however privileged, is not an owner
        wrong_role = self.human("hr-1", roles=("oversight_reviewer",))
        forged = dataclasses.replace(self.dpo, token=self.dpo.token[:-3] + "AAA")
        other_tenant = self.human("dpo-2", roles=("dpo",), tenant="tenant-b")
        for who in (service_with_role, wrong_role, forged, other_tenant):
            with self.assertRaises(ConditionError):
                self.core.conditions.satisfy(oid, who, "ref-1")
        with self.assertRaises(ConditionError):
            self.core.conditions.satisfy(oid, self.dpo, "  ")
        self.assertEqual(self.core.conditions.get(oid).status, Status.PENDING)

    def test_unmet_condition_does_not_consume_the_single_use_approval(self):
        from src.agents.governance.processing_context import DataClass
        self.core.activities.register(make_activity("act-g", art22_exception="contract"), actor="dpo")
        self.add("apply_supplementary_measures")
        c = self.request(activity_id="act-g", automated_decision=True, significant_effect=True, application_id="app-1", purpose="document_editing")
        t0 = time.time() - 600
        om = self.core.oversight
        aid = om.request_approval(c, DOSSIER, self.requester, now=t0)
        om.mark_presented(aid, self.reviewer, now=t0 + 1)
        om.decide(aid, self.reviewer, approve=True, rationale="Reviewed affordability evidence and alternatives",
                  reviewed_dossier_digest=DOSSIER.digest(), now=t0 + 60)
        with_approval = dataclasses.replace(c, approval_id=aid)
        d = self.core.authorize(with_approval)
        self.assertEqual((d.decision, om.status(aid)), (Decision.REVIEW_REQUIRED, ApprovalStatus.APPROVED))  # NOT consumed
        self.core.conditions.satisfy(self.only_pending().obligation_id, self.dpo, "tia-annex-3")
        d = self.core.authorize(with_approval)
        self.assertTrue(d.allowed, d)
        self.assertEqual(om.status(aid), ApprovalStatus.CONSUMED)

    def test_governance_fulfils_its_own_conditions_with_signed_evidence(self):
        self.core.activities.update(make_activity("act-1", operations=(Operation.PROCESS, Operation.TRANSFER),
                                                  transfers=({"country": "JP", "mechanism": "adequacy", "ref": "ec-2019-419"},)), expected_version=1, actor="dpo")
        before = len(self.core.evidence.records("GOV-TRF-09"))
        d = self.core.authorize(self.request(operation=Operation.TRANSFER, destination_country="JP"))
        self.assertEqual(d.decision, Decision.ALLOW_WITH_CONTROLS)
        self.assertEqual(len(self.core.evidence.records("GOV-TRF-09")), before + 1)
        self.assertEqual(self.core.evidence.records("GOV-TRF-09")[-1].evidence_type, "transfer_record")
        self.assertTrue(self.core.evidence.verify_chain()[0])
        self.assertEqual(self.core.conditions.pending(), [])

    def test_satisfaction_expires(self):
        clock = [time.time() + 10]
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            core = GovernanceCore(tmp, master_key=KEY, clock=lambda: clock[0])
            self.addCleanup(core.close)
            core.activities.register(make_activity("act-1"), actor="dpo")
            core.engine.add_rule(controls_rule("apply_supplementary_measures"))
            ident = core.identity.verify(core.identity.issue("svc-requester", "tenant-a", actor_type="service", ttl_seconds=100000, now=clock[0]), now=clock[0])
            dpo = core.identity.verify(core.identity.issue("dpo-1", "tenant-a", actor_type="human", roles=("dpo",), ttl_seconds=100000, now=clock[0]), now=clock[0])
            ctx = dataclasses.replace(self.ctx(), identity=ident)
            core.authorize(ctx)
            core.conditions.satisfy(core.conditions.pending()[0].obligation_id, dpo, "tia-1", now=clock[0])
            self.assertTrue(core.authorize(ctx).allowed)
            clock[0] += 901
            self.assertEqual(core.authorize(ctx).decision, Decision.REVIEW_REQUIRED)  # stale: re-attest

    def test_refused_requests_leave_no_obligations(self):
        self.add("apply_supplementary_measures")
        self.core.authorize(self.request(purpose="advertising"))  # blocked by lawful-basis rules; the test rule keys on purpose
        self.assertEqual(self.core.conditions.pending(), [])


class ReleaseRuntimePostTests(ConditionTestCase):
    def test_release_condition_blocks_release_until_satisfied(self):
        self.add("art50_transparency_obligations")
        d = self.core.authorize(self.request())
        self.assertTrue(d.allowed)
        self.assertEqual(self.core.conditions.release_unmet(d.authorization_id), ["art50_transparency_obligations"])
        with self.assertRaises(ConditionsUnmet):
            self.core.conditions.release_gate(d.authorization_id)
        self.core.conditions.satisfy_system(d.authorization_id, "art50_transparency_obligations", "marking:abc")
        self.core.conditions.release_gate(d.authorization_id)

    def test_gate_enforces_release_conditions_and_marking_satisfies_them(self):
        self.add("art50_transparency_obligations")
        spec = {"language": {"system_id": "sys-1", "name": "Writer", "generates_synthetic_content": True, "interacts_with_natural_persons": True}}
        gate = AgentGate(self.core, mode="enforce")
        d = self.core.authorize(self.request())
        with self.assertRaises(ConditionsUnmet):  # no marking configured for this agent: the output is not released
            gate.after("plain", {"x": 1}, {"text": "out"}, d)
        self.assertEqual(gate.status()["stats"]["release_blocked"], 1)
        d2 = self.core.authorize(self.request())
        marking = AgentGate(self.core, mode="enforce", marking_agents=spec)
        out = marking.after("language", {"x": 1}, {"text": "out"}, d2)
        self.assertIn("ai_content_marking", out)
        self.assertEqual(self.core.conditions.release_unmet(d2.authorization_id), [])

    def test_observe_mode_counts_but_releases(self):
        self.add("art50_transparency_obligations")
        gate = AgentGate(self.core, mode="observe")
        d = self.core.authorize(self.request())
        self.assertEqual(gate.after("plain", {}, {"text": "out"}, d), {"text": "out"})
        self.assertEqual(gate.status()["stats"]["release_would_block"], 1)

    def test_release_obligation_survives_a_restart(self):
        self.add("art50_transparency_obligations")
        d = self.core.authorize(self.request())
        state = self.core.state_dir
        self.core.close()
        GovernanceCore.reset_shared()
        again = GovernanceCore(state, master_key=KEY)
        self.addCleanup(again.close)
        self.assertEqual(again.conditions.release_unmet(d.authorization_id), ["art50_transparency_obligations"])

    def approved(self, *conditions):
        self.core.activities.register(make_activity("act-g", art22_exception="contract"), actor="dpo")
        self.add()  # no-op, keeps helper symmetrical
        c = self.request(activity_id="act-g", automated_decision=True, significant_effect=True, application_id="app-1")
        t0 = time.time() - 600
        om = self.core.oversight
        aid = om.request_approval(c, DOSSIER, self.requester, now=t0)
        om.mark_presented(aid, self.reviewer, now=t0 + 1)
        om.decide(aid, self.reviewer, approve=True, rationale="Reviewed affordability evidence and alternatives",
                  reviewed_dossier_digest=DOSSIER.digest(), conditions=conditions, now=t0 + 60)
        return dataclasses.replace(c, approval_id=aid)

    def test_reviewer_conditions_become_release_obligations_by_default(self):
        d = self.core.authorize(self.approved("check-output-for-bias"))
        self.assertTrue(d.allowed, d)
        (ob,) = self.core.conditions.obligations(d.authorization_id)
        self.assertEqual((ob.condition, ob.phase, ob.mandatory), ("approved_condition:check-output-for-bias", Phase.RELEASE, True))
        with self.assertRaises(ConditionsUnmet):
            self.core.conditions.release_gate(d.authorization_id)
        self.core.conditions.satisfy(ob.obligation_id, self.dpo, "review-note-12")
        self.core.conditions.release_gate(d.authorization_id)

    def test_reviewer_condition_can_be_waived_with_a_rationale_but_mandatory_controls_cannot(self):
        d = self.core.authorize(self.approved("check-output-for-bias"))
        (ob,) = self.core.conditions.obligations(d.authorization_id)
        with self.assertRaises(ConditionError):
            self.core.conditions.waive(ob.obligation_id, self.dpo, "too short")
        self.core.conditions.waive(ob.obligation_id, self.dpo, "Output is purely numeric; bias review is not meaningful here")
        self.assertEqual(self.core.conditions.get(ob.obligation_id).status, Status.WAIVED)
        self.add("apply_supplementary_measures")
        self.core.authorize(self.request())
        with self.assertRaises(ConditionError):
            self.core.conditions.waive(self.only_pending().obligation_id, self.dpo, "we would rather not do this one at all")

    def test_runtime_condition_is_verified_on_every_effect(self):
        self.core.enforcement_mode = "enforce"
        d = self.core.authorize(self.approved("runtime:egress-only-to-allowlist"))
        self.assertTrue(d.allowed, d)
        name = "approved_condition:runtime:egress-only-to-allowlist"
        with governed_scope("agent-a", d, self.request()):
            with self.assertRaises(ConditionsUnmet):  # nobody can verify it: fail closed
                self.core.effects.guard("network_outbound", "t")
            allowed = {"value": True}
            self.core.conditions.register_runtime_verifier(name, lambda auth, kind, target: (allowed["value"], "ok"))
            self.assertEqual(self.core.effects.guard("network_outbound", "t").outcome, "scoped")
            allowed["value"] = False
            with self.assertRaises(ConditionsUnmet):
                self.core.effects.guard("network_outbound", "t")
            self.assertEqual(self.core.effects.guard("read_internal", "cfg").outcome, "scoped")  # low-risk effects are not gated

    def test_failed_mandatory_condition_stops_the_running_operation(self):
        d = self.core.authorize(self.approved("check-output-for-bias"))
        lease = self.core.authorization(d)
        (ob,) = self.core.conditions.obligations(d.authorization_id)
        self.core.conditions.fail(ob.obligation_id, self.dpo, "bias review found disparate impact")
        with self.assertRaises(AuthorizationRevoked):
            lease.check()
        self.assertIn("terminated", [r["trigger"] for r in self.core.reassessment.log()])

    def test_post_conditions_are_tracked_escalated_once_and_never_block(self):
        self.add("open_question:art22-exception-scope")
        d = self.core.authorize(self.request())
        self.assertTrue(d.allowed)
        (ob,) = self.core.conditions.obligations(d.authorization_id)
        self.assertEqual((ob.phase, ob.mandatory, ob.status), (Phase.POST, False, Status.PENDING))
        self.core.conditions.release_gate(d.authorization_id)  # not a release condition
        self.assertEqual(self.core.conditions.overdue(), [])
        later = ob.due_at + 1
        self.assertEqual([o.obligation_id for o in self.core.conditions.overdue(later)], [ob.obligation_id])
        seen = []
        self.core.changes.subscribe(lambda kind, key, detail: seen.append(kind))
        self.assertEqual(self.core.conditions.escalate_overdue(later), 1)
        self.assertEqual(self.core.conditions.escalate_overdue(later), 0)
        self.assertEqual(seen.count("condition_overdue"), 1)

    def test_event_log_is_hash_chained_and_tamper_evident(self):
        self.add("apply_supplementary_measures")
        self.core.authorize(self.request())
        oid = self.only_pending().obligation_id
        self.core.conditions.satisfy(oid, self.dpo, "tia-1")
        self.assertEqual([e["kind"] for e in self.core.conditions.events(oid)], ["created", "satisfied"])
        self.assertTrue(self.core.conditions.verify_log())
        self.core.conditions._db.execute("UPDATE condition_events SET actor='someone-else' WHERE kind='satisfied'")
        self.assertFalse(self.core.conditions.verify_log())
        self.assertFalse(self.core.health()["ready"])  # a broken condition log is a readiness problem


if __name__ == "__main__":
    unittest.main()
