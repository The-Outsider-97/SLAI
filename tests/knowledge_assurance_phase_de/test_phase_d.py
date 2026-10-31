"""Phase D stdlib-only regression suite; run with PYTHONPATH=src/agents/knowledge."""
from __future__ import annotations

from dataclasses import replace
import sqlite3
import tempfile
from pathlib import Path
import unittest

from assurance import (AccessContext, AuthorizationDenied, KnowledgeAssuranceGateway,
                       Lifecycle, RecordDraft, Scope, ValidationFailure)


SAMPLE_RULE = {
    "schema": "slai.rule_eval.v1",
    "assert": {"op": "ge", "left": {"field": "clearance"},
               "right": {"const": "1.2", "unit": "m"}},
    "input_units": {"clearance": "m"},
    "description": "Fictional test constraint only",
}
VALID_AT = "2026-10-01T00:00:00Z"


class Fixture(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.g = KnowledgeAssuranceGateway(Path(tmp.name)/"canonical.db")
        self.admin = AccessContext("admin", can_write=True, can_publish_global=True,
                                   can_review=True, can_attest=True,
                                   project_ids=frozenset({"P1", "P2"}),
                                   organization_ids=frozenset({"Org1"}))
        self.alice = AccessContext("alice", can_write=True,
                                   project_ids=frozenset({"P1"}))
        self.bob = AccessContext("bob", can_write=True,
                                 project_ids=frozenset({"P2"}))
        self.reviewer = AccessContext("reviewer", can_write=True,
                                      can_review=True, project_ids=frozenset({"P1", "P2"}))
        self.view1 = AccessContext("viewer1", project_ids=frozenset({"P1"}))
        self.view2 = AccessContext("viewer2", project_ids=frozenset({"P2"}))

    def project(self, proj="P1", writer=None, statement="A private design requirement"):
        return self.g.projects.register(project_id=proj, kind="project_requirement",
                                        statement=statement,
                                        access=writer or (self.alice if proj == "P1" else self.bob))

    def rule(self):
        rule = self.g.rules.register_rule(spec=SAMPLE_RULE, access=self.admin,
                                          scope=Scope.GLOBAL)
        return self.g.repository.revise(rule.record_id, rule.revision,
                     replace(rule.draft, status=Lifecycle.ACTIVE), self.admin,
                     reason="reviewed fictional rule")

    def assess(self, project="P1"):
        obj = self.project(proj=project)
        rule = self.rule()
        outcome = self.g.assessments.evaluate_and_record(
            project_id=project, object_record_id=obj.record_id,
            rule_record_id=rule.record_id, inputs={"clearance": "1.4"},
            access=self.alice if project == "P1" else self.bob,
            valid_at=VALID_AT, evidence_refs=("test:evidence",))
        return obj, rule, outcome


class AssessmentTests(Fixture):
    def test_rule_object_and_input_snapshots(self):
        obj, rule, result = self.assess()
        self.assertEqual("satisfied", result.outcome.value)
        self.assertEqual(rule.revision, result.rule_revision)
        self.assertEqual(obj.revision, result.object_revision)
        self.assertFalse(result.stale)
        self.assertEqual(64, len(result.assessment_digest))
        self.assertEqual(1, len(self.g.assessments.list_project("P1", self.view1)))

    def test_cross_project_denied_and_hidden(self):
        obj, rule, result = self.assess()
        self.assertIsNone(self.g.assessments.get(result.assessment_id, self.view2))
        self.assertEqual([], self.g.assessments.list_project("P1", self.view2))
        with self.assertRaises(AuthorizationDenied):
            self.g.assessments.evaluate_and_record(
                project_id="P1", object_record_id=obj.record_id,
                rule_record_id=rule.record_id, inputs={"clearance": 2},
                access=self.bob, valid_at=VALID_AT)

    def test_object_revision_causes_stale_without_mutating_outcome(self):
        obj, _, result = self.assess()
        old_known = result.recorded_at
        self.g.projects.revise(obj.record_id, 1, "Updated design requirement",
                               self.alice, reason="new constraint")
        current = self.g.assessments.get(result.assessment_id, self.alice)
        self.assertTrue(current.stale)
        self.assertEqual("satisfied", current.outcome.value)
        historical = self.g.assessments.get(result.assessment_id, self.alice,
                                            as_known=old_known)
        self.assertFalse(historical.stale)
        with sqlite3.connect(self.g.repository.path) as db:
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("DELETE FROM assessment_snapshots WHERE assessment_id=?",
                           (result.assessment_id,))

    def test_rule_revision_causes_stale(self):
        obj, rule, result = self.assess()
        self.g.repository.revise(rule.record_id, rule.revision,
                                 replace(rule.draft, payload={**rule.draft.payload, "new": "meaning changed"}),
                                 self.admin, reason="rule revision")
        current = self.g.assessments.get(result.assessment_id, self.alice)
        self.assertTrue(current.stale)

    def test_transitive_dependency_causes_stale(self):
        obj, rule, result = self.assess()
        baseline = self.project(statement="Critical source")
        self.g.repository.link(obj.record_id, baseline.record_id, self.alice)
        self.g.repository.revise(baseline.record_id, 1,
                                 replace(baseline.draft, payload={**baseline.draft.payload,
                                                                  "new": "changed"}),
                                 self.alice, reason="upstream changed")
        causes = self.g.assessments.get(result.assessment_id, self.alice).stale_causes
        self.assertTrue(any(baseline.record_id in x for x in causes))

    def test_concurrent_revision_rejected_for_assessment_dependency(self):
        obj, rule, _ = self.assess()
        revision = self.g.projects.revise(obj.record_id, 1, "New version",
                                          self.alice, reason="updated")
        self.assertEqual(2, revision.revision)
        with self.assertRaises(ValidationFailure):
            self.g.assessments.evaluate_and_record(
                project_id="P1", object_record_id=obj.record_id,
                rule_record_id=rule.record_id, inputs={"clearance": float("nan")},
                access=self.alice, valid_at=VALID_AT)

    def test_additional_dependency_cannot_cross_projects(self):
        obj, rule, _ = self.assess()
        another = self.project(proj="P2")
        with self.assertRaises(ValidationFailure):
            self.g.assessments.evaluate_and_record(
                project_id="P1", object_record_id=obj.record_id,
                rule_record_id=rule.record_id, inputs={"clearance": "2"},
                access=self.admin, valid_at=VALID_AT,
                additional_dependencies=(another.record_id,))

    def test_duplicate_assessment_id_rejected(self):
        obj, rule, _ = self.assess()
        params = dict(project_id="P1", object_record_id=obj.record_id,
                      rule_record_id=rule.record_id, inputs={"clearance": "2"},
                      access=self.alice, valid_at=VALID_AT,
                      assessment_id="assessment:fixed")
        self.g.assessments.evaluate_and_record(**params)
        from assurance.contracts import RevisionConflict
        with self.assertRaises(RevisionConflict):
            self.g.assessments.evaluate_and_record(**params)

    def test_stale_only_query(self):
        obj, _, result = self.assess()
        self.assertEqual([], self.g.assessments.list_project("P1", self.alice, stale_only=True))
        self.g.projects.revise(obj.record_id, 1, "Change", self.alice, reason="changed")
        self.assertEqual([result.assessment_id],
                         [a.assessment_id for a in self.g.assessments.list_project("P1", self.alice, stale_only=True)])


class AdoptionAndGovernanceTests(Fixture):
    def edition(self, edition):
        return self.g.standards.register_edition(
            standard_id="TEST-100", edition=edition, title="Fictional Standard",
            publisher="Fictional Publisher", source_id="source:fictional",
            access=self.admin, publication_status="published")

    def test_independent_project_adoption(self):
        e1, e2 = self.edition("2024"), self.edition("2026")
        a = self.g.adoptions.propose(project_id="P1", edition_record_id=e1.record_id,
             authority_evidence_id="contract:p1", basis="Reviewed local contract", access=self.alice)
        b = self.g.adoptions.propose(project_id="P2", edition_record_id=e2.record_id,
             authority_evidence_id="contract:p2", basis="Reviewed different contract", access=self.bob)
        self.g.adoptions.approve(a.record_id, 1, reviewer_evidence_id="review:p1",
                                 access=self.reviewer, reason="confirmed contract")
        self.g.adoptions.approve(b.record_id, 1, reviewer_evidence_id="review:p2",
                                 access=self.reviewer, reason="confirmed contract")
        ra = self.g.adoptions.resolve(project_id="P1", standard_id="TEST-100",
                                      access=self.view1, valid_at=VALID_AT)
        rb = self.g.adoptions.resolve(project_id="P2", standard_id="TEST-100",
                                      access=self.view2, valid_at=VALID_AT)
        self.assertEqual("adopted_candidate", ra.outcome)
        self.assertEqual((e1.record_id,), ra.edition_record_ids)
        self.assertEqual((e2.record_id,), rb.edition_record_ids)

    def test_conflicting_approved_editions_not_auto_resolved(self):
        for edition in ("2024", "2026"):
            e = self.edition(edition)
            adoption = self.g.adoptions.propose(project_id="P1", edition_record_id=e.record_id,
                           authority_evidence_id="contract:p1", basis="Proposed", access=self.alice)
            self.g.adoptions.approve(adoption.record_id, 1, reviewer_evidence_id="review:1",
                                     access=self.reviewer, reason="reviewed")
        value = self.g.adoptions.resolve(project_id="P1", standard_id="TEST-100",
                                         access=self.view1, valid_at=VALID_AT)
        self.assertEqual("conflict", value.outcome)
        self.assertEqual(2, len(value.edition_record_ids))

    def test_self_approval_rejected(self):
        e = self.edition("2024")
        a = self.g.adoptions.propose(project_id="P1", edition_record_id=e.record_id,
                 authority_evidence_id="contract:p1", basis="Proposed", access=self.reviewer)
        with self.assertRaises(AuthorizationDenied):
            self.g.adoptions.approve(a.record_id, 1, reviewer_evidence_id="review:x",
                                     access=self.reviewer, reason="my own record")

    def test_external_project_adoption_not_visible(self):
        e = self.edition("2024")
        a = self.g.adoptions.propose(project_id="P1", edition_record_id=e.record_id,
                 authority_evidence_id="contract:p1", basis="Private contract", access=self.alice)
        self.assertIsNone(self.g.repository.get(a.record_id, self.view2))
        with self.assertRaises(AuthorizationDenied):
            self.g.adoptions.resolve(project_id="P1", standard_id="TEST-100",
                                     access=self.view2, valid_at=VALID_AT)

    def test_validity_time_and_as_known_adoption(self):
        e = self.edition("2024")
        a = self.g.adoptions.propose(project_id="P1", edition_record_id=e.record_id,
                 authority_evidence_id="contract:p1", basis="Effective after October", access=self.alice,
                 valid_from="2026-10-15T00:00:00Z")
        self.g.adoptions.approve(a.record_id, 1, reviewer_evidence_id="review:1",
                                 access=self.reviewer, reason="approved")
        early = self.g.adoptions.resolve(project_id="P1", standard_id="TEST-100",
                     access=self.view1, valid_at="2026-10-01T00:00:00Z")
        later = self.g.adoptions.resolve(project_id="P1", standard_id="TEST-100",
                     access=self.view1, valid_at="2026-10-20T00:00:00Z")
        self.assertEqual("no_adoption", early.outcome)
        self.assertEqual("adopted_candidate", later.outcome)

    def test_historical_lineage_respects_access(self):
        parent = self.project()
        child = self.project()
        self.g.repository.link(child.record_id, parent.record_id, self.alice)
        outgoing = self.g.lineage.links(child.record_id, self.view1)
        self.assertEqual(parent.record_id, outgoing[0]["record_id"])
        self.assertEqual([], self.g.lineage.links(child.record_id, self.view2))

    def test_promotion_public_summary_with_private_audit(self):
        record = self.g.projects.register(project_id="P1", kind="project_decision",
            statement="Private confidential detail", rationale="Private reason", access=self.alice)
        approved = self.g.projects.approve(record.record_id, 1, self.reviewer,
                                           approval_evidence_id="approval:private", reason="independent")
        public = self.g.promotions.promote(origin_record_id=record.record_id,
              expected_revision=approved.revision, public_statement="Reviewed public summary",
              access=self.admin, privacy_evidence_id="privacy:approved",
              licence_evidence_id="licence:cleared")
        self.assertEqual("knowledge_summary", public.draft.kind)
        self.assertNotIn(record.record_id, str(public.draft.payload))
        self.assertIsNone(self.g.promotions.origin_for_reviewer(public.record_id,
                            AccessContext("public-viewer", can_review=True)))
        self.assertEqual(record.record_id,
                         self.g.promotions.origin_for_reviewer(public.record_id, self.admin)["origin_record_id"])

    def test_promotion_requires_independent_authorized_review(self):
        rec = self.project()
        with self.assertRaises(ValidationFailure):
            self.g.promotions.promote(origin_record_id=rec.record_id, expected_revision=1,
                 public_statement="public", access=self.admin,
                 privacy_evidence_id="privacy:ok", licence_evidence_id="licence:ok")

    def test_approved_exception_does_not_override_rule(self):
        e = self.g.projects.register(project_id="P1", kind="project_exception",
            statement="Exception requested", rationale="Explicit human review", access=self.alice)
        self.g.projects.approve(e.record_id, 1, self.reviewer,
                                approval_evidence_id="review:exception", reason="human approved")
        obj, rule, result = self.assess()
        self.assertEqual("satisfied", result.outcome.value)
        self.assertTrue(self.g.repository.get(e.record_id, self.view1))


class OutboxTests(Fixture):
    def test_revision_is_transactionally_enqueued(self):
        rec = self.project()
        self.g.projects.revise(rec.record_id, rec.revision,
                               "Updated requirement", self.alice, reason="change")
        events = self.g.outbox.lease()
        self.assertTrue(any(e.event_type == "knowledge.revised" for e in events))
        self.assertEqual(len({x.event_id for x in events}), len(events))

    def test_ack_is_lease_token_scoped(self):
        rec = self.project()
        self.g.projects.revise(rec.record_id, 1, "Another", self.alice, reason="new")
        evt = self.g.outbox.lease()[0]
        self.assertFalse(self.g.outbox.ack(evt.event_id, "different-token"))
        self.assertTrue(self.g.outbox.ack(evt.event_id, evt.lease_token))
        self.assertFalse(self.g.outbox.ack(evt.event_id, evt.lease_token))

    def test_no_duplicate_outbox_on_reprocessing(self):
        rec = self.project()
        self.g.projects.revise(rec.record_id, 1, "Updated", self.alice, reason="x")
        with self.g.repository._connect() as db:
            count = db.execute("SELECT COUNT(*) FROM knowledge_outbox WHERE event_type='knowledge.revised'").fetchone()[0]
        self.assertEqual(1, count)

    def test_failed_consumer_retries_without_lying_about_delivery(self):
        rec = self.project()
        self.g.projects.revise(rec.record_id, 1, "Updated", self.alice, reason="x")
        event = self.g.outbox.lease()[0]
        self.assertTrue(self.g.outbox.fail(event.event_id, event.lease_token))
        self.assertEqual(1, self.g.outbox.status()["pending"])
        self.assertEqual(0, self.g.outbox.status()["delivered"])


class DeactivationTests(Fixture):
    def test_relation_deactivation_is_historical_and_authorized(self):
        child, parent = self.project(), self.project()
        self.g.repository.link(child.record_id, parent.record_id, self.alice)
        snapshot_time = self.g.lineage.links(child.record_id, self.view1)[0]["recorded_at"]
        with self.assertRaises(AuthorizationDenied):
            self.g.repository.unlink(child.record_id, parent.record_id, self.bob)
        self.g.repository.unlink(child.record_id, parent.record_id, self.alice)
        self.assertEqual([], self.g.lineage.links(child.record_id, self.view1))
        old = self.g.lineage.links(child.record_id, self.view1, as_known=snapshot_time)
        self.assertEqual(parent.record_id, old[0]["record_id"])
        with self.assertRaises(Exception):
            self.g.repository.unlink(child.record_id, parent.record_id, self.alice)


class MultiApplicationTests(Fixture):
    def test_two_gateways_share_edition_but_not_other_project_facts(self):
        second = KnowledgeAssuranceGateway(self.g.repository.path)
        source = self.g.standards.register_edition(
            standard_id="TEST-200", edition="2026", title="Fictional shared edition",
            publisher="Fictional Publisher", source_id="source:fixture",
            access=self.admin, publication_status="published")
        protected = self.project(statement="Secret P1 assumption")
        self.assertIsNotNone(second.repository.get(source.record_id, self.view2))
        self.assertIsNone(second.repository.get(protected.record_id, self.view2))
        self.assertIsNotNone(second.repository.get(protected.record_id, self.view1))