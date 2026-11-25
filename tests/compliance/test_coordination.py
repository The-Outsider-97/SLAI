"""GOV-P1-02: governance-aware inter-agent coordination (real GovernanceCore; specialists are the only test doubles)."""

import time
import unittest

from src.agents.governance.coordination import Coordinator, DEFAULT_CONTRACTS, SpecialistContract
from src.agents.governance.policy_resolution import BASES
from src.agents.governance.processing_context import DataClass, Decision, Operation
from src.agents.governance.risk_classification import AISystemProfile
from tests.compliance._fixtures import GovernanceTestCase

SECRET = "ssn-123-45-6789-do-not-retain"
TAG = "emotion_recognition_workplace_education"  # allowed only through a complete registered exception


class Specialists:
    """Specialist double implementing the *port*: records calls and returns configurable reports."""

    def __init__(self, verdicts=None):
        self.calls = []
        self.verdicts = verdicts or {}
        self.mutate = {}

    def __call__(self, name, action, request):
        self.calls.append((name, action))
        if name in self.mutate:
            return self.mutate[name](request)
        v = self.verdicts.get(name, "clear")
        if isinstance(v, Exception):
            raise v
        kinds = {"privacy": "privacy_assessment", "safety": "safety_assessment", "provenance": "provenance_attestation", "alignment": "alignment_assessment"}
        return {"specialist": name, "evidence_kind": kinds[name], "verdict": v, "request_fingerprint": request["request_fingerprint"], "summary": "ok"}


class CoordinationTests(GovernanceTestCase):
    def setUp(self):
        super().setUp()
        self.reg_profile(AISystemProfile(
            "med-1", "fatigue", intended_purpose="fatigue detection", practices=(TAG,),
            practice_details={TAG: {"medical_or_safety_purpose": True}}))

    def safety_ctx(self, **over):
        return self.np_ctx(Operation.PROCESS, "analysis", ai_system_id="med-1", ai_practices=(TAG,), identity=self.ident(), **over)

    # --- selectivity --------------------------------------------------------------
    def test_request_that_needs_no_specialist_invokes_none(self):
        sp = Specialists()
        d = self.core.authorize_coordinated(self.np_ctx(Operation.PROCESS, "analysis", identity=self.ident()), sp)
        self.assertEqual(sp.calls, [])
        self.assertEqual(self.core.coordinator.stats["invoked"], 0)
        self.assertIsNotNone(d.resolution)

    def test_refused_request_invokes_nobody(self):
        sp = Specialists()
        d = self.core.authorize_coordinated(self.ctx(data_class=DataClass.UNCLASSIFIED, activity_id="nope"), sp)
        self.assertIn(d.decision, (Decision.BLOCK, Decision.QUARANTINE))
        self.assertEqual(sp.calls, [])

    def test_verified_classification_does_not_re_invoke_privacy(self):
        sp = Specialists()
        self.core.authorize_coordinated(self.np_ctx(Operation.PROCESS, "analysis", identity=self.ident()), sp)
        self.assertNotIn("privacy", [c[0] for c in sp.calls])

    def test_only_the_triggered_specialist_is_invoked(self):
        sp = Specialists()
        d = self.core.authorize_coordinated(self.safety_ctx(), sp)
        self.assertEqual([c[0] for c in sp.calls], ["safety"])
        self.assertTrue(d.allowed, d.to_dict())

    # --- evidence contract and monotonicity --------------------------------------
    def test_clear_never_relaxes_and_never_changes_the_outcome(self):
        c = self.safety_ctx()
        baseline = self.core.engine.evaluate(c).decision
        sp = Specialists()
        outcome = self.core.coordinator.coordinate(self.core.engine.resolve(c), baseline, sp)
        self.assertEqual(outcome.results, ())
        self.assertEqual(self.core.engine.evaluate(c, outcome.results).decision, baseline)

    def test_violation_blocks(self):
        d = self.core.authorize_coordinated(self.safety_ctx(), Specialists({"safety": "violation"}))
        self.assertEqual(d.decision, Decision.BLOCK)
        self.assertIn("GOV-COORD-01", self.rule_ids(d))

    def test_concern_requires_review_and_is_overridable_like_any_review(self):
        c = self.safety_ctx()
        outcome = self.core.coordinator.coordinate(self.core.engine.resolve(c), Decision.ALLOW, Specialists({"safety": "concern"}))
        self.assertEqual(outcome.results[0].decision, Decision.REVIEW_REQUIRED)
        self.assertTrue(outcome.results[0].overridable)
        d = self.core.authorize(c, coordination=outcome)
        self.assertEqual(d.decision, Decision.REVIEW_REQUIRED)
        self.assertEqual(d.resolution["kind"], "review")

    def test_unknown_verdict_quarantines_and_resolves_to_clarify(self):
        d = self.core.authorize_coordinated(self.safety_ctx(), Specialists({"safety": "unknown"}))
        self.assertEqual(d.decision, Decision.QUARANTINE)
        self.assertEqual(d.resolution["kind"], "clarify")

    def test_required_specialist_failure_fails_closed(self):
        for failure in (RuntimeError("boom"), PermissionError("denied by its own gate")):
            with self.subTest(failure=type(failure).__name__):
                d = self.core.authorize_coordinated(self.safety_ctx(), Specialists({"safety": failure}))
                self.assertEqual(d.decision, Decision.QUARANTINE)
                self.assertIn("GOV-COORD-03", {r.rule_id for r in d.results})

    def test_invalid_evidence_is_rejected_not_trusted(self):
        bad = {
            "wrong_specialist": lambda r: {"specialist": "privacy", "evidence_kind": "safety_assessment", "verdict": "clear", "request_fingerprint": r["request_fingerprint"]},
            "wrong_kind": lambda r: {"specialist": "safety", "evidence_kind": "marketing", "verdict": "clear", "request_fingerprint": r["request_fingerprint"]},
            "other_request": lambda r: {"specialist": "safety", "evidence_kind": "safety_assessment", "verdict": "clear", "request_fingerprint": "0" * 64},
            "bad_verdict": lambda r: {"specialist": "safety", "evidence_kind": "safety_assessment", "verdict": "approved!", "request_fingerprint": r["request_fingerprint"]},
            "not_a_mapping": lambda r: "all good",
        }
        for label, fn in bad.items():
            with self.subTest(label):
                sp = Specialists()
                sp.mutate["safety"] = fn
                d = self.core.authorize_coordinated(self.safety_ctx(), sp)
                self.assertEqual(d.decision, Decision.QUARANTINE, d.to_dict())

    def test_optional_specialist_failure_is_recorded_and_changes_nothing(self):
        c = self.np_ctx(Operation.LEARN, "analysis", identity=self.ident())
        outcome = self.core.coordinator.coordinate(self.core.engine.resolve(c), Decision.ALLOW, Specialists({"provenance": RuntimeError("x")}))
        self.assertIn(("provenance", "error:RuntimeError"), outcome.failures)
        self.assertEqual(outcome.results, ())  # optional: no restriction, and above all no relaxation

    def test_timeout_is_a_failure(self):
        slow = SpecialistContract("safety", "assess_safety", frozenset({"safety_assessment"}), lambda f: True, "t", timeout_seconds=0.05)
        coord = Coordinator((slow,))
        out = coord.coordinate(self.core.engine.resolve(self.safety_ctx()), Decision.ALLOW, lambda n, a, r: time.sleep(0.5))
        self.assertEqual(out.failures, (("safety", "timeout"),))
        self.assertEqual(out.results[0].rule_id, "GOV-COORD-03")

    def test_evidence_for_another_request_cannot_authorise_this_one(self):
        a, b = self.safety_ctx(), self.safety_ctx()
        outcome = self.core.coordinator.coordinate(self.core.engine.resolve(a), Decision.ALLOW, Specialists())
        # b is a different request (distinct classification evidence/request id): the report is not bound to it
        d = self.core.authorize(b, coordination=type(outcome)("0" * 64, results=()))
        self.assertEqual(d.decision, Decision.BLOCK)
        self.assertIn("GOV-COORD-05", self.rule_ids(d))

    # --- duplication and recursion ------------------------------------------------
    def test_same_request_is_not_re_asked(self):
        c = self.core.engine.resolve(self.safety_ctx())
        sp = Specialists()
        self.core.coordinator.coordinate(c, Decision.ALLOW, sp)
        again = self.core.coordinator.coordinate(c, Decision.ALLOW, sp)
        self.assertEqual(len(sp.calls), 1)
        self.assertEqual(again.reused, ("safety",))

    def test_policy_change_invalidates_reuse(self):
        c = self.core.engine.resolve(self.safety_ctx())
        sp = Specialists()
        self.core.coordinator.coordinate(c, Decision.ALLOW, sp)
        self.core.notify_change("policy_changed", "policy", "GOV-AI5-03")  # bumps the change epoch
        self.core.coordinator.coordinate(c, Decision.ALLOW, sp)
        self.assertEqual(len(sp.calls), 2)

    def test_specialist_calling_back_into_governance_does_not_recurse(self):
        c = self.safety_ctx()
        sp = Specialists()
        inner_results = []

        def reentrant(request):
            inner_results.append(self.core.authorize_coordinated(self.safety_ctx(), sp))  # the specialist asks governance again
            return {"specialist": "safety", "evidence_kind": "safety_assessment", "verdict": "clear", "request_fingerprint": request["request_fingerprint"]}

        sp.mutate["safety"] = reentrant
        outer = self.core.authorize_coordinated(c, sp)
        self.assertEqual(len([x for x in sp.calls if x[0] == "safety"]), 1)  # no second entry into the same specialist
        self.assertGreaterEqual(self.core.coordinator.stats["recursion_prevented"], 1)
        self.assertEqual(inner_results[0].decision, Decision.QUARANTINE)  # the re-entrant request failed closed
        self.assertTrue(outer.allowed, outer.to_dict())

    def test_a_specialist_is_never_asked_to_assess_its_own_request(self):
        sp = Specialists()
        out = self.core.coordinator.coordinate(self.core.engine.resolve(self.safety_ctx()), Decision.ALLOW, sp, requester="safety")
        self.assertEqual(sp.calls, [])
        self.assertIn(("safety", "recursion_prevented"), out.skipped)

    # --- data minimisation ---------------------------------------------------------
    def test_payload_is_handed_to_the_port_only_and_never_retained(self):
        seen = []
        sp = Specialists()
        orig = sp.__call__

        def spy(name, action, request):
            seen.append(request["payload"])
            return orig(name, action, request)

        d = self.core.authorize_coordinated(self.safety_ctx(), spy, payload={"text": SECRET})
        self.assertEqual(seen, [{"text": SECRET}])
        blob = repr(d.to_dict()) + repr(self.core.coordinator.stats) + repr(self.core.coordinator._cache)
        self.assertNotIn(SECRET, blob)

    def test_every_coordination_rule_has_a_registered_basis(self):
        for rid in ("GOV-COORD-01", "GOV-COORD-02", "GOV-COORD-03", "GOV-COORD-04", "GOV-COORD-05"):
            self.assertIn(rid, BASES)

    def test_default_contracts_are_unique_and_cover_the_specialist_agents(self):
        names = [c.name for c in DEFAULT_CONTRACTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), {"privacy", "safety", "provenance", "alignment"})


class AdapterTests(GovernanceTestCase):
    """The real agents cannot be imported in CI-light environments (torch); their documented result shapes are covered
    here through fake agent classes behind the REAL factory contract and the REAL delegation guard."""

    def test_adapters_map_each_agents_result_shape_restrictively(self):
        from src.agents.governance.coordination import ADAPTERS
        cases = {
            "privacy": [({"decision": "allow"}, "clear"), ({"decision": "modify"}, "concern"), ({"decision": "escalate"}, "concern"),
                        ({"decision": "block"}, "violation"), ({"decision": "???"}, "unknown"), (None, "unknown"), ({}, "unknown")],
            "safety": [({"decision": "allow", "approved": True}, "clear"), ({"decision": "allow", "approved": False}, "unknown"),
                       ({"decision": "review"}, "concern"), ({"decision": "block"}, "violation"), ({"decision": "x"}, "unknown")],
            "alignment": [({"decision": {"approved": True, "requires_review": False}}, "clear"),
                          ({"decision": {"approved": True, "requires_review": True}}, "concern"),
                          ({"decision": {"approved": False, "decision": "review_required"}}, "concern"), ({"decision": "approved"}, "unknown"), ({}, "unknown")],
            "provenance": [({"artifact_id": "a1", "lineage": [1]}, "clear"), ({}, "unknown"), ({"error": "x", "lineage": [1]}, "unknown"), (None, "unknown")],
        }
        for name, rows in cases.items():
            for raw, expected in rows:
                with self.subTest(agent=name, raw=raw):
                    self.assertEqual(ADAPTERS[name](raw)[0], expected)

    def test_unrecognised_output_can_never_be_clear(self):
        from src.agents.governance.coordination import ADAPTERS
        for name, fn in ADAPTERS.items():
            for junk in (None, 0, "", "ok", [], {}, {"decision": None}, {"approved": True}, {"status": "fine"}):
                self.assertNotEqual(fn(junk)[0], "clear", (name, junk))

    def test_invoker_goes_through_the_delegation_guard_and_the_factory(self):
        from src.agents.governance.coordination import agent_invoker
        calls = []

        class FakeSafety:
            def validate_action(self, action_params, action_context=None):
                calls.append(action_params)
                return {"decision": "allow", "approved": True}

            def _private(self):  # never delegable
                return {}

        class Factory:
            def normalize_agent_type(self, n): return n
            def create(self, n, mem, **kw): return FakeSafety()

        out = agent_invoker(Factory(), None)("safety", "validate_action", {"request_fingerprint": "f" * 64, "payload": {"a": 1}, "purpose": "p", "operation": "process"})
        self.assertEqual(out["verdict"], "clear")
        self.assertEqual(out["evidence_kind"], "safety_assessment")
        self.assertEqual(calls[0]["action_name"], "process:p")
        with self.assertRaises(Exception):  # a non-delegable action is refused by the guard, not called
            agent_invoker(Factory(), None)("safety", "_private", {"request_fingerprint": "f" * 64})

    def test_without_an_agent_factory_required_specialists_fail_closed(self):
        from src.agents.governance.coordination import agent_invoker
        self.reg_profile(AISystemProfile("med-1", "fatigue", intended_purpose="x", practices=(TAG,), practice_details={TAG: {"medical_or_safety_purpose": True}}))
        c = self.np_ctx(Operation.PROCESS, "analysis", ai_system_id="med-1", ai_practices=(TAG,), identity=self.ident())
        d = self.core.authorize_coordinated(c, agent_invoker(None, None))
        self.assertEqual(d.decision, Decision.QUARANTINE)

    def test_contract_actions_exist_on_the_real_agent_sources(self):
        """Static check against the repository's own agent modules (no import needed): every contract action is a real method."""
        import ast
        from pathlib import Path
        agents = Path(__file__).resolve().parents[2] / "src" / "agents"
        files = {"privacy": "privacy_agent.py", "safety": "safety_agent.py", "alignment": "alignment_agent.py", "provenance": "provenance_agent.py"}
        for c in DEFAULT_CONTRACTS:
            tree = ast.parse((agents / files[c.name]).read_text(encoding="utf-8"))
            methods = {n.name for cls in tree.body if isinstance(cls, ast.ClassDef) for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            self.assertIn(c.action, methods, f"{c.name}.{c.action} does not exist")


if __name__ == "__main__":
    unittest.main()
