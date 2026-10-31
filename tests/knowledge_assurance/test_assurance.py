"""Deterministic, offline, stdlib-only tests for the proposed A–C components.

Run: PYTHONPATH=src/agents/knowledge python -m unittest discover -s tests/knowledge_assurance -v
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from assurance import (AccessContext, AuthorizationDenied, DeclarativeRuleEvaluator, KnowledgeRepository,
                       Lifecycle, RecordDraft, RevisionConflict, RuleOutcome, RulesRegistry, Scope,
                       StandardsRegistry, ValidationFailure)
from assurance.contracts import normalize_time
from assurance.legacy_rules import UnsafeLegacyRule, interpret_legacy_rule_code


VALID_RULE = {
    "schema": "slai.rule_eval.v1",
    "when": {"op": "eq", "left": {"field": "category"}, "right": {"const": "public", "unit": "text"}},
    "assert": {"op": "ge", "left": {"field": "clearance"}, "right": {"const": "1.2", "unit": "m"}},
    "input_units": {"clearance": "mm", "category": "text"},
    "description": "Demonstration rule, not a legal requirement",
}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = KnowledgeRepository(Path(self.tmp.name) / "evidence.sqlite3")
        self.admin = AccessContext("global-admin", can_write=True,
                                   can_publish_global=True, can_review=True, can_attest=True,
                                   project_ids=frozenset({"P-1", "P-2"}),
                                   organization_ids=frozenset({"O-1"}))
        self.a = AccessContext("alice", project_ids=frozenset({"P-1"}), can_write=True, can_review=True)
        self.b = AccessContext("bob", project_ids=frozenset({"P-2"}), can_write=True)
        self.viewer = AccessContext("viewer", project_ids=frozenset({"P-1"}))
        self.reviewer = AccessContext("reviewer", can_write=True, can_publish_global=True,
                                      can_review=True, can_attest=True)

    def draft(self, payload=None, scope=Scope.PROJECT, status=Lifecycle.DRAFT):
        return RecordDraft(kind="project_knowledge", scope=scope,
                           project_id="P-1" if scope == Scope.PROJECT else None,
                           payload=payload or {"claim": "A"}, status=status)


class CoreStorageTests(Fixture):
    def test_global_read_unauthed_project_denied(self):
        self.store.create(RecordDraft("assertion", Scope.GLOBAL, {"fact": "public"}), self.admin, record_id="global:one")
        self.store.create(self.draft({"claim": "secret"}), self.a, record_id="project:one")
        self.assertIsNone(self.store.get("project:one", None))
        self.assertIsNone(self.store.get("project:one", self.b))
        self.assertEqual("secret", self.store.get("project:one", self.viewer).draft.payload["claim"])
        self.assertEqual(["global:one"], [r.record_id for r in self.store.query(None)])
        self.assertEqual(["global:one"], [r.record_id for r in self.store.query(self.b)])
        self.assertEqual({"project:one", "global:one"}, {r.record_id for r in self.store.query(self.a)})

    def test_project_write_requires_member(self):
        with self.assertRaises(AuthorizationDenied):
            self.store.create(self.draft(), self.b)
        with self.assertRaises(AuthorizationDenied):
            self.store.create(self.draft(), self.viewer)

    def test_requires_approval_for_active(self):
        active = self.draft(status=Lifecycle.ACTIVE)
        with self.assertRaises(AuthorizationDenied):
            self.store.create(active, self.b)
        rec = self.store.create(active, self.a)
        self.assertEqual("active", rec.draft.status.value)

    def test_revision_conflict_preservation_and_history(self):
        first = self.store.create(self.draft(), self.a, record_id="project:rev")
        second = self.store.revise(first.record_id, 1, replace(first.draft, payload={"claim": "B"}), self.a, reason="changed")
        self.assertEqual(second.revision, 2)
        with self.assertRaises(RevisionConflict):
            self.store.revise(first.record_id, 1, first.draft, self.a, reason="stale")
        self.assertEqual("A", self.store.get("project:rev", self.a, as_known=first.recorded_at).draft.payload["claim"])
        self.assertEqual("B", self.store.get("project:rev", self.a).draft.payload["claim"])
        self.assertNotEqual(first.digest, second.digest)
        with sqlite3.connect(self.store.path) as db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("UPDATE knowledge_revisions SET reason='tampered' WHERE record_id=?", (first.record_id,))

    def test_revision_cannot_reassign_project(self):
        a = self.store.create(self.draft(), self.a)
        with self.assertRaises(ValidationFailure):
            self.store.revise(a.record_id, 1, replace(a.draft, project_id="P-2"), self.admin, reason="leak")

    def test_effective_vs_known_time(self):
        t0 = "2026-01-01T00:00:00Z"
        t1 = "2026-02-01T00:00:00Z"
        record = self.store.create(replace(self.draft(), valid_from=t0, valid_until=t1), self.a)
        self.assertIsNone(self.store.get(record.record_id, self.a, as_valid="2025-12-31T23:00:00Z"))
        self.assertIsNotNone(self.store.get(record.record_id, self.a, as_valid="2026-01-15T00:00:00Z"))
        self.assertIsNone(self.store.get(record.record_id, self.a, as_valid="2026-02-01T00:00:00Z"))
        self.assertIsNone(self.store.get(record.record_id, self.a, as_known="2025-01-01T00:00:00Z"))

    def test_lifecycle_query(self):
        x = self.store.create(self.draft(), self.a)
        self.store.revise(x.record_id, x.revision, replace(x.draft, status=Lifecycle.SUPERSEDED), self.a, reason="replaced")
        self.assertEqual([], self.store.query(self.a))
        self.assertEqual(1, len(self.store.query(self.a, include_inactive=True)))

    def test_dependency_impact_and_resolution(self):
        source = self.store.create(self.draft({"a": "source"}), self.a, record_id="A")
        derived = self.store.create(self.draft({"b": "derived"}), self.a, record_id="B")
        self.store.link("B", "A", self.a)
        self.store.revise("A", 1, replace(source.draft, payload={"a": "source2"}), self.a, reason="changed source")
        impacted = self.store.impacts("A", self.a)
        self.assertEqual(1, len(impacted))
        self.assertEqual("B", impacted[0]["record_id"])
        self.store.revise("B", 1, replace(derived.draft, payload={"b": "recomputed"}), self.a, reason="revalidated")
        self.assertEqual([], self.store.impacts("A", self.a))

    def test_cross_project_and_global_cannot_depend_on_private(self):
        one = self.store.create(self.draft(), self.a)
        other = self.store.create(RecordDraft("project_knowledge", Scope.PROJECT, {"secret": True}, project_id="P-2"), self.b)
        with self.assertRaises(ValidationFailure):
            self.store.link(one.record_id, other.record_id, self.admin)
        glob = self.store.create(RecordDraft("assertion", Scope.GLOBAL, {"a": 1}), self.admin)
        with self.assertRaises(ValidationFailure):
            self.store.link(glob.record_id, one.record_id, self.admin)

    def test_dependency_cycles_prohibited(self):
        self.store.create(self.draft(), self.a, record_id="A")
        self.store.create(self.draft(), self.a, record_id="B")
        self.store.create(self.draft(), self.a, record_id="C")
        self.store.link("B", "A", self.a)
        self.store.link("C", "B", self.a)
        with self.assertRaises(ValidationFailure):
            self.store.link("A", "C", self.a)

    def test_no_cross_project_search_text(self):
        self.store.create(self.draft({"api_key": "TOP_SECRET_P1"}), self.a)
        self.assertEqual([], self.store.query(self.b, text="TOP_SECRET_P1"))
        self.assertEqual([], self.store.query(None, text="TOP_SECRET_P1"))
        self.assertEqual(1, len(self.store.query(self.a, text="TOP_SECRET_P1")))

    def test_sql_injection_as_search_data(self):
        self.store.create(self.draft({"text": "john"}), self.a)
        self.assertEqual([], self.store.query(self.a, text="%' OR 1=1 --"))

    def test_id_requires_safe_characters(self):
        with self.assertRaises(ValidationFailure):
            self.store.create(self.draft(), self.a, record_id="../escape")

    def test_persistence_reopens(self):
        self.store.create(self.draft({"persisted": True}), self.a, record_id="persist:1")
        another = KnowledgeRepository(self.store.path)
        self.assertTrue(another.get("persist:1", self.a).draft.payload["persisted"])


class StandardsTests(Fixture):
    def setUp(self):
        super().setUp()
        self.registry = StandardsRegistry(self.store)

    def _standard(self, **kwargs):
        opts = {"standard_id": "ISO-TEST", "edition": "2026", "title": "Test publication metadata",
                "publisher": "Example Agency", "source_id": "prov:src123",
                "access": self.admin, "publication_status": "published",
                "applicability": {"jurisdiction": ["NL"], "domain": ["general"]}}
        opts.update(kwargs)
        return self.registry.register_edition(**opts)

    def test_standards_are_metadata_only_and_unverified(self):
        std = self._standard()
        self.assertEqual({}, std.draft.payload["verification"])
        self.assertEqual("metadata_only", std.draft.payload["licence"])
        self.assertEqual("draft", std.draft.status.value)

    def test_standard_independent_attestation(self):
        std = self._standard()
        with self.assertRaises(AuthorizationDenied):
            self.registry.attest(std.record_id, 1, dimension="edition", verdict="supported",
                                 source_id="prov:src123", evidence_id="prov:check001",
                                 method="publisher index", access=self.a)
        verified = self.registry.attest(std.record_id, 1, dimension="edition", verdict="supported",
                                        source_id="prov:src123", evidence_id="prov:check001",
                                        method="publisher index", access=self.reviewer)
        self.assertEqual("supported", verified.draft.payload["verification"]["edition"]["verdict"])
        self.assertEqual({}, self.store.get(std.record_id, self.admin, as_known=std.recorded_at).draft.payload["verification"])

    def test_independent_reviewer_cannot_be_original_author(self):
        std = self._standard()
        with self.assertRaises(AuthorizationDenied):
            self.registry.attest(std.record_id, 1, dimension="edition", verdict="supported",
                                 source_id="prov:src123", evidence_id="prov:check001",
                                 method="publisher index", access=self.admin)

    def test_requirements_applicability_unresolved_and_matching(self):
        std = self._standard()
        req = self.registry.register_requirement(standard_record_id=std.record_id,
                    clause_locator="§3.4", paraphrase="Example dimensional criterion",
                    modality="conditional", access=self.admin)
        unknown = self.registry.applicable_requirements(self.admin, context={})
        self.assertEqual("undetermined", unknown[0]["applicability"])
        eligible = self.registry.applicable_requirements(self.admin,
                    context={"jurisdiction": "NL", "domain": "general"})
        self.assertEqual("applicable_candidate", eligible[0]["applicability"])
        excluded = self.registry.applicable_requirements(self.admin,
                    context={"jurisdiction": "US", "domain": "general"})
        self.assertEqual("not_applicable", excluded[0]["applicability"])
        self.assertEqual("§3.4", req.draft.payload["clause_locator"])

    def test_supersession_reference_no_silent_overwrite(self):
        old = self._standard(edition="2020")
        new = self._standard(edition="2026", supersedes_record_id=old.record_id)
        self.assertIsNotNone(self.store.get(old.record_id, self.admin))
        self.assertNotEqual(old.record_id, new.record_id)

    def test_cannot_claim_unlisted_publication_state(self):
        with self.assertRaises(ValidationFailure):
            self._standard(publication_status="officially_compliant")


class RuleTests(Fixture):
    def setUp(self):
        super().setUp()
        self.evaluator = DeclarativeRuleEvaluator()

    def test_six_outcomes(self):
        e = lambda inp, spec=VALID_RULE: self.evaluator.evaluate(rule_id="rule:1", rule_revision=1, spec=spec, inputs=inp).outcome
        self.assertEqual(RuleOutcome.SATISFIED, e({"category": "public", "clearance": 1200}))
        self.assertEqual(RuleOutcome.VIOLATED, e({"category": "public", "clearance": 1199}))
        self.assertEqual(RuleOutcome.NOT_APPLICABLE, e({"category": "private", "clearance": 100}))
        self.assertEqual(RuleOutcome.UNDETERMINED, e({"category": "public"}))
        invalid = dict(VALID_RULE, **{"assert": {"op": "exec", "command": "rm -rf /"}})
        self.assertEqual(RuleOutcome.UNSUPPORTED, e({"category": "public"}, invalid))
        self.assertEqual(RuleOutcome.EVALUATION_ERROR, e({"category": "public", "clearance": "not a number"}))

    def test_strict_units_and_dimension(self):
        invalid_units = dict(VALID_RULE, **{"input_units": {"clearance": "kg", "category": "text"}})
        result = self.evaluator.evaluate(rule_id="rule:x", rule_revision=1, spec=invalid_units,
                                         inputs={"category": "public", "clearance": 1200})
        self.assertEqual(RuleOutcome.EVALUATION_ERROR, result.outcome)

    def test_invalid_and_oversized_expressions(self):
        with self.assertRaises(ValidationFailure):
            self.evaluator.validate_rule({"schema": "slai.rule_eval.v1", "assert": {"op": "__import__"}})
        huge = {"op": "all", "args": [{"op": "exists", "field": str(i)} for i in range(200)]}
        with self.assertRaises(ValidationFailure):
            self.evaluator.validate_rule({"schema": "slai.rule_eval.v1", "assert": huge})

    def test_draft_rule_cannot_evaluate_as_reviewed(self):
        rules = RulesRegistry(self.store)
        rec = rules.register_rule(spec=VALID_RULE, access=self.admin)
        result = rules.evaluate(rec.record_id, inputs={"category": "public", "clearance": 1200}, access=self.admin)
        self.assertEqual(RuleOutcome.UNSUPPORTED, result.outcome)
        reviewed = self.store.revise(rec.record_id, 1, replace(rec.draft, status=Lifecycle.REVIEWED), self.admin,
                                     reason="reviewed by authorized human")
        result = rules.evaluate(rec.record_id, inputs={"category": "public", "clearance": 1200}, access=self.admin)
        self.assertEqual(RuleOutcome.SATISFIED, result.outcome)
        self.assertEqual(2, result.rule_revision)
        persisted = rules.record_evaluation(result, self.admin)
        self.assertFalse(persisted.draft.payload["compliance_certification"])
        self.assertEqual("verification_outcome", persisted.draft.kind)

    def test_eval_requires_acl(self):
        rules = RulesRegistry(self.store)
        rec = rules.register_rule(spec=VALID_RULE, access=self.a, scope=Scope.PROJECT, project_id="P-1")
        with self.assertRaises(ValidationFailure):
            rules.evaluate(rec.record_id, inputs={}, access=self.b)


class LegacyInterpreterTests(unittest.TestCase):
    def test_small_legacy_rule(self):
        rule = """if kb.get('danger', False) and not kb.get('mitigated', False):
    inferred['hazard_warning'] = 0.9
if kb.get('ok', False):
    inferred['passed'] = 1.0
"""
        self.assertEqual({"hazard_warning": 0.9}, interpret_legacy_rule_code(rule, {"danger": True}))
        self.assertEqual({"passed": 1.0}, interpret_legacy_rule_code(rule, {"ok": True}))

    def test_blank_rule_does_nothing(self):
        self.assertEqual({}, interpret_legacy_rule_code("", {}))

    def test_import_exec_file_access_and_loop_rejected(self):
        attacks = ["import os", "__import__('os').system('echo bad')",
                   "for x in range(2):\n    inferred['x'] = 1.0",
                   "inferred['x'] = kb.__class__", "open('/tmp/x','w')",
                   "inferred['x'] = (1).__class__", "def f():\n    pass",
                   "while True: pass", "inferred['x'] = 9**999999"]
        for attack in attacks:
            with self.subTest(attack=attack), self.assertRaises(UnsafeLegacyRule):
                interpret_legacy_rule_code(attack, {})

    def test_no_attribute_execution_through_kb_objects(self):
        class Evil:
            def __str__(self):
                raise AssertionError("should never be called")
        with self.assertRaises(ValidationFailure):
            interpret_legacy_rule_code("if kb.get('x'): inferred['x']=1", {"x": Evil()})

    def test_confidence_range(self):
        with self.assertRaises(UnsafeLegacyRule):
            interpret_legacy_rule_code("inferred['fake_confidence'] = 99.0", {})


if __name__ == "__main__":
    unittest.main()
