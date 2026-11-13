"""Additional negative and concurrency tests for bounded assurance contracts."""
from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

from assurance import (AccessContext, DeclarativeRuleEvaluator, KnowledgeRepository,
                       RecordDraft, RevisionConflict, RuleOutcome, Scope,
                       StandardsRegistry, ValidationFailure)
from assurance.legacy_rules import UnsafeLegacyRule, interpret_legacy_rule_code


class AdversarialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = KnowledgeRepository(Path(self.tmp.name) / "k.sqlite3")
        self.p1 = AccessContext("p1-author", project_ids=frozenset({"p1"}), can_write=True, can_review=True)
        self.p2 = AccessContext("p2-author", project_ids=frozenset({"p2"}), can_write=True, can_review=True)
        self.admin = AccessContext("publisher", can_write=True, can_publish_global=True, can_review=True)
        self.rules = DeclarativeRuleEvaluator()

    def _rec(self, project_id="p1"):
        access = self.p1 if project_id == "p1" else self.p2
        return self.db.create(RecordDraft("project_requirement", Scope.PROJECT,
                                {"description": "strictly confidential"}, project_id=project_id), access)

    def test_denied_historical_knowledge_cannot_be_recovered(self):
        rec = self._rec()
        self.assertIsNone(self.db.get(rec.record_id, self.p2, as_known=rec.recorded_at))
        self.assertEqual([], self.db.query(self.p2, as_known=rec.recorded_at, text="confidential"))

    def test_invalid_timezone_and_range_rejected(self):
        rec = self._rec()
        with self.assertRaises(ValidationFailure):
            self.db.get(rec.record_id, self.p1, as_known="2026-10-31T00:00:00")
        with self.assertRaises(ValidationFailure):
            RecordDraft("assertion", Scope.GLOBAL, {}, valid_from="2026-10-31T00:00:00Z",
                        valid_until="2026-10-30T00:00:00Z")

    def test_concurrent_writers_cannot_lose_revision(self):
        original = self._rec()
        def update(i):
            try:
                return self.db.revise(original.record_id, 1, replace(original.draft,
                         payload={"writer": i}), self.p1, reason=f"writer {i}").revision
            except RevisionConflict:
                return None
        with ThreadPoolExecutor(max_workers=12) as pool:
            outcomes = list(pool.map(update, range(12)))
        self.assertEqual(1, outcomes.count(2))
        self.assertEqual(11, outcomes.count(None))
        self.assertEqual(2, self.db.get(original.record_id, self.p1).revision)

    def test_project_filter_is_sql_filtered_before_limit(self):
        for i in range(10):
            self.db.create(RecordDraft("assertion", Scope.GLOBAL, {"name": f"public{i}"}), self.admin)
        target = self._rec()
        results = self.db.query(self.p1, kind="project_requirement", project_id="p1", limit=1)
        self.assertEqual([target.record_id], [r.record_id for r in results])

    def test_disallowed_scope_not_visible_through_project_filter(self):
        self._rec("p2")
        self.assertEqual([], self.db.query(self.p1, kind="project_requirement", project_id="p2"))

    def test_unknown_input_unit_is_unsupported_not_compliant(self):
        rule = {"schema": "slai.rule_eval.v1",
                "assert": {"op": "gt", "left": {"field": "m"}, "right": {"const": 2, "unit": "m"}},
                "input_units": {"m": "m"}}
        result = self.rules.evaluate(rule_id="test", rule_revision=1, spec=rule,
                        inputs={"m": {"value": 3, "unit": "made-up-unit"}})
        self.assertEqual(RuleOutcome.UNSUPPORTED, result.outcome)

    def test_nonfinite_input_produces_error_not_exception(self):
        rule = {"schema": "slai.rule_eval.v1",
                "assert": {"op": "gt", "left": {"field": "m"}, "right": {"const": 2, "unit": "m"}},
                "input_units": {"m": "m"}}
        result = self.rules.evaluate(rule_id="test", rule_revision=1, spec=rule, inputs={"m": float("nan")})
        self.assertEqual(RuleOutcome.EVALUATION_ERROR, result.outcome)
        self.assertEqual("", result.input_digest)

    def test_short_circuit_and_missing_input_reason(self):
        spec = {"schema": "slai.rule_eval.v1", "assert": {"op": "all", "args": [
            {"op": "exists", "field": "ready"},
            {"op": "ge", "left": {"field": "missing"}, "right": {"const": 3}}]}}
        r = self.rules.evaluate(rule_id="test", rule_revision=1, spec=spec, inputs={})
        self.assertEqual(RuleOutcome.VIOLATED, r.outcome)  # false AND missing -> false
        spec2 = {"schema": "slai.rule_eval.v1", "assert": {"op": "any", "args": [
            {"op": "exists", "field": "ready"},
            {"op": "ge", "left": {"field": "missing"}, "right": {"const": 3}}]}}
        r2 = self.rules.evaluate(rule_id="test", rule_revision=1, spec=spec2, inputs={})
        self.assertEqual(RuleOutcome.UNDETERMINED, r2.outcome)
        self.assertEqual(("missing",), r2.missing_inputs)

    def test_malicious_rule_ast_in_inactive_branch_is_never_executed(self):
        text = "if kb.get('is_active', False):\n    __import__('os').system('echo BAD')\n"
        self.assertEqual({}, interpret_legacy_rule_code(text, {"is_active": False}))
        with self.assertRaises(UnsafeLegacyRule):
            interpret_legacy_rule_code(text, {"is_active": True})

    def test_invalid_standard_selector_rejected_at_write(self):
        with self.assertRaises(ValidationFailure):
            StandardsRegistry(self.db).register_edition(
                standard_id="ST-1", edition="2026", title="Example", publisher="Example",
                source_id="source:1", access=self.admin,
                applicability={"jurisdiction": "NL"})

    def test_unknown_publication_requires_context_review(self):
        registry = StandardsRegistry(self.db)
        std = registry.register_edition(standard_id="ST-1", edition="2026", title="Example",
                                        publisher="Example", source_id="source:1", access=self.admin)
        registry.register_requirement(standard_record_id=std.record_id, clause_locator="A1",
                                      paraphrase="Optional example", modality="advisory", access=self.admin)
        results = registry.applicable_requirements(self.admin, context={})
        self.assertEqual("undetermined", results[0]["applicability"])


if __name__ == "__main__":
    unittest.main()
