"""Project-scoped requirement/decision and approval lifecycle regressions."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from assurance import (AccessContext, AuthorizationDenied, Lifecycle, KnowledgeRepository,
                       RecordDraft, Scope, RulesRegistry, ValidationFailure)
from assurance.project import ProjectKnowledgeService


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = KnowledgeRepository(Path(self.tmp.name) / "k.sqlite3")
        self.service = ProjectKnowledgeService(self.store)
        self.author = AccessContext("author", project_ids=frozenset({"A"}), can_write=True)
        self.reviewer = AccessContext("reviewer", project_ids=frozenset({"A"}), can_write=True, can_review=True)
        self.other = AccessContext("other", project_ids=frozenset({"B"}), can_write=True, can_review=True)

    def test_decision_requires_rationale(self):
        with self.assertRaises(ValidationFailure):
            self.service.register(project_id="A", kind="project_decision", statement="Choose option C",
                                  access=self.author)

    def test_approval_is_distinct_from_author(self):
        decision = self.service.register(project_id="A", kind="project_decision",
                statement="Choose option C", rationale="Evidence E favors C", access=self.author)
        with self.assertRaises(AuthorizationDenied):
            self.service.approve(decision.record_id, 1, self.author, approval_evidence_id="ev:1", reason="looks fine")
        approved = self.service.approve(decision.record_id, 1, self.reviewer,
                    approval_evidence_id="ev:1", reason="checked decision rationale")
        self.assertEqual(Lifecycle.ACTIVE, approved.draft.status)
        self.assertEqual("reviewer", approved.draft.payload["approval"]["reviewer"])

    def test_revision_invalidates_approval(self):
        decision = self.service.register(project_id="A", kind="project_decision",
                  statement="Choose C", rationale="A", access=self.author)
        approved = self.service.approve(decision.record_id, 1, self.reviewer,
                  approval_evidence_id="ev:1", reason="reviewed")
        updated = self.service.revise(decision.record_id, 2, "Choose D", self.author, reason="changed decision")
        self.assertEqual(Lifecycle.DRAFT, updated.draft.status)
        self.assertIsNone(updated.draft.payload["approval"])
        self.assertIsNotNone(self.store.get(decision.record_id, self.author, as_known=approved.recorded_at))

    def test_private_record_not_global(self):
        self.service.register(project_id="A", kind="project_assumption", statement="Confidential assumption",
                              access=self.author)
        self.assertEqual(0, len(self.service.project_records("A", self.other)))
        self.assertEqual(0, len(self.store.query(None)))

    def test_scope_cannot_change_on_revision(self):
        r = self.service.register(project_id="A", kind="project_requirement",
                                  statement="Provide assurance", access=self.author)
        with self.assertRaises(ValidationFailure):
            self.store.revise(r.record_id, 1, replace(r.draft, scope=Scope.GLOBAL, project_id=None),
                              AccessContext("global-admin", can_write=True, can_publish_global=True,
                                            project_ids=frozenset({"A"})), reason="attempt to promote")

    def test_inheritance_is_only_candidates_without_precedence(self):
        global_author = AccessContext("publisher", can_write=True, can_publish_global=True, can_review=True)
        spec = {"schema": "slai.rule_eval.v1", "assert": {"op": "exists", "field": "x"}}
        rules = RulesRegistry(self.store)
        r = rules.register_rule(spec=spec, access=global_author)
        self.store.revise(r.record_id, 1, replace(r.draft, status=Lifecycle.REVIEWED), global_author,
                          reason="manual review")
        found = self.service.inherited_rule_candidates("A", self.author)
        self.assertEqual(1, len(found))
        self.assertEqual("unresolved", found[0]["precedence"])


if __name__ == "__main__":
    unittest.main()
