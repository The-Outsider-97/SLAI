"""Phase E stdlib-only reliability, retrieval, QA and provider-update tests."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from assurance import (AccessContext, AuthorizationDenied, EnhancedKnowledgeGateway,
                       Lifecycle, RecordDraft, Scope, Sufficiency, ValidationFailure)
from assurance.gateway import RetrievalBenchmarkCase, run_benchmark, load_cases
from assurance.gateway import OutboxDispatcher


class Fixture(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.g = EnhancedKnowledgeGateway(self.tmp/"knowledge.db")
        self.admin = AccessContext("admin", project_ids=frozenset({"P1", "P2"}),
                                   can_write=True, can_review=True,
                                   can_publish_global=True, can_attest=True)
        self.alice = AccessContext("alice", project_ids=frozenset({"P1"}),
                                   can_write=True, can_review=True)
        self.bob = AccessContext("bob", project_ids=frozenset({"P2"}),
                                 can_write=True, can_review=True)
        self.viewer1 = AccessContext("view1", project_ids=frozenset({"P1"}))
        self.viewer2 = AccessContext("view2", project_ids=frozenset({"P2"}))
        self.anonymous = None

    def create(self, text, project_id="P1", *,
               kind="project_requirement", status=Lifecycle.DRAFT,
               scope=Scope.PROJECT):
        return self.g.repository.create(RecordDraft(
            kind=kind, scope=scope,
            project_id=project_id if scope == Scope.PROJECT else None,
            payload={"statement": text, "author": "fixture"}, status=status),
            self.admin)


class RetrievalTests(Fixture):
    def test_same_principal_two_projects_cannot_mix_contexts(self):
        p1 = self.create("Fluorescent copper panel", "P1")
        p2 = self.create("Fluorescent copper panel", "P2")
        for project, expected in (("P1", p1.record_id), ("P2", p2.record_id)):
            result = self.g.retrieval.search("Fluorescent copper panel", self.admin,
                                             project_id=project)
            self.assertEqual([expected], [x.record_id for x in result.hits])

    def test_other_project_cannot_see_secret_even_with_exact_search(self):
        self.create("EXCLUSIVELYCONFIDENTIALMECHANISM", "P1")
        found = self.g.retrieval.search("EXCLUSIVELYCONFIDENTIALMECHANISM", self.viewer2,
                                        project_id="P2")
        self.assertEqual((), found.hits)

    def test_unauthenticated_user_only_global_metadata(self):
        private = self.create("public query private value")
        public = self.create("public query public value", scope=Scope.GLOBAL)
        result = self.g.retrieval.search("public query", self.anonymous)
        self.assertEqual([public.record_id], [hit.record_id for hit in result.hits])

    def test_global_and_context_project_search_coexist(self):
        private = self.create("renewable thermal battery")
        public = self.create("renewable thermal reference", scope=Scope.GLOBAL)
        found = self.g.retrieval.search("renewable thermal", self.viewer1, project_id="P1")
        self.assertEqual({private.record_id, public.record_id}, {hit.record_id for hit in found.hits})

    def test_one_project_member_cannot_select_another(self):
        with self.assertRaises(AuthorizationDenied):
            self.g.retrieval.search("query", self.viewer2, project_id="P1")

    def test_revision_watermark_changes_without_stale_cache(self):
        rec = self.create("efficiency ratio")
        first = self.g.retrieval.search("efficiency", self.viewer1, project_id="P1")
        self.g.repository.revise(rec.record_id, 1,
              replace(rec.draft, payload={"statement": "revised efficiency ratio"}),
              self.admin, reason="revision")
        second = self.g.retrieval.search("efficiency", self.viewer1, project_id="P1")
        self.assertNotEqual(first.corpus_watermark, second.corpus_watermark)
        self.assertNotEqual(first.hits[0].revision, second.hits[0].revision)

    def test_unrelated_project_revision_does_not_change_watermark(self):
        self.create("same authorized document", "P1")
        other = self.create("hidden project document", "P2")
        first = self.g.retrieval.search("document", self.viewer1, project_id="P1")
        self.g.repository.revise(other.record_id, 1,
               replace(other.draft, payload={"statement": "new hidden document"}),
               self.admin, reason="P2 change")
        second = self.g.retrieval.search("document", self.viewer1, project_id="P1")
        self.assertEqual(first.corpus_watermark, second.corpus_watermark)

    def test_bitemporal_retrieval(self):
        rec = self.g.repository.create(RecordDraft(
            kind="project_requirement", scope=Scope.PROJECT, project_id="P1",
            payload={"statement": "historic leaf ventilation"},
            valid_from="2026-02-01T00:00:00Z",
            valid_until="2026-07-01T00:00:00Z"), self.admin)
        old = self.g.retrieval.search("historic leaf", self.viewer1, project_id="P1",
                     as_valid="2026-05-01T00:00:00Z")
        new = self.g.retrieval.search("historic leaf", self.viewer1, project_id="P1",
                     as_valid="2026-10-01T00:00:00Z")
        self.assertEqual(rec.record_id, old.hits[0].record_id)
        self.assertEqual([], list(new.hits))

    def test_bounded_retrieval_reports_truncation(self):
        retriever = type(self.g.retrieval)(self.g.repository, max_candidates=1)
        self.create("alpha first")
        self.create("alpha second")
        found = retriever.search("alpha", self.viewer1, project_id="P1")
        self.assertTrue(found.truncated)
        self.assertEqual(1, found.scanned)

    def test_invalid_query_and_k_fail(self):
        for query, k in (("", 2), ("!", 2), ("alpha", 0), ("alpha", 101)):
            with self.assertRaises(ValidationFailure):
                self.g.retrieval.search(query, self.viewer1, project_id="P1", k=k)

    def test_content_rights_metadata_only_blocks_candidate_paraphrase(self):
        rec = self.g.repository.create(RecordDraft(
            kind="requirement", scope=Scope.PROJECT, project_id="P1",
            payload={"candidate_paraphrase": "licensedfictionaltexttoken",
                     "clause_locator": "Cl 2.0"}), self.admin)
        found = self.g.retrieval.search("licensedfictionaltexttoken", self.viewer1, project_id="P1")
        self.assertEqual((), found.hits)
        self.g.repository.revise(rec.record_id, 1,
              replace(rec.draft, payload={**rec.draft.payload, "content_rights": "public_domain"}),
              self.admin, reason="rights reviewed")
        found2 = self.g.retrieval.search("licensedfictionaltexttoken", self.viewer1, project_id="P1")
        self.assertEqual(rec.record_id, found2.hits[0].record_id)


class SufficiencyAndBenchmarks(Fixture):
    def test_missing_evidence_not_reported_sufficient(self):
        self.create("weld geometry")
        result = self.g.retrieval.search("weld geometry", self.viewer1, project_id="P1")
        state = self.g.sufficiency.inspect(result, self.viewer1)
        self.assertEqual(Sufficiency.INCOMPLETE, state.state)
        self.assertIn("source_evidence", state.missing)

    def test_no_hits_is_not_found(self):
        result = self.g.retrieval.search("unknownphrase", self.viewer1, project_id="P1")
        state = self.g.sufficiency.inspect(result, self.viewer1)
        self.assertEqual(Sufficiency.NOT_FOUND, state.state)

    def test_evidence_dimension_is_unverified_without_proof(self):
        ed = self.g.standards.register_edition(
            standard_id="FAKE-1", edition="2024", title="Fictional thermal safety",
            publisher="Fictional publisher", source_id="test:source",
            publication_status="published", access=self.admin)
        result = self.g.retrieval.search("thermal", self.viewer1, project_id="P1")
        state = self.g.sufficiency.inspect(result, self.viewer1,
                                           required_dimensions=("authenticity",))
        self.assertEqual(Sufficiency.UNVERIFIED, state.state)
        self.assertIn("verified:authenticity", state.missing)

    def test_conflicted_authority_never_auto_overridden(self):
        result = self.g.retrieval.search("query", self.viewer1, project_id="P1")
        state = self.g.sufficiency.inspect(result, self.viewer1, conflict=True)
        self.assertEqual(Sufficiency.CONFLICTED, state.state)

    def test_retrieval_benchmark_scores_and_groups(self):
        expected = self.create("bacterial geometry module")
        unwanted = self.create("bacterial geometry hidden", project_id="P2")
        cases = [RetrievalBenchmarkCase("case1", "bacterial geometry", "viewer1",
                   (expected.record_id,), (unwanted.record_id,), project_id="P1",
                   language="en", domain="engineering")]
        report = run_benchmark(self.g.retrieval, cases, {"viewer1": self.viewer1}, k=1)
        self.assertEqual(1, report.evaluated)
        self.assertEqual(1.0, report.recall_at_k)
        self.assertEqual(1.0, report.ndcg_at_k)
        self.assertEqual(0, report.leak_count)
        self.assertIn("language:en", report.by_group)

    def test_benchmark_does_not_create_acl_from_case(self):
        cases = [RetrievalBenchmarkCase("case1", "query", "unknown", ("rec:one",))]
        with self.assertRaises(ValidationFailure):
            run_benchmark(self.g.retrieval, cases, {})

    def test_load_benchmark_jsonl(self):
        path = self.tmp/"cases.jsonl"
        path.write_text(json.dumps({"case_id": "one", "query": "hello", "principal_key": "user",
                                    "relevant_ids": ["rec:one"]}) + "\n", encoding="utf-8")
        items = load_cases(path)
        self.assertEqual(1, len(items))
        self.assertEqual("one", items[0].case_id)


class OperationTests(Fixture):
    def revision_event(self):
        obj = self.create("event subject")
        self.g.repository.revise(obj.record_id, 1, replace(obj.draft, payload={"statement": "updated"}),
                                 self.admin, reason="change")

    def test_dispatch_ack(self):
        self.revision_event()
        received = []
        worker = OutboxDispatcher(self.g.outbox, lambda e: received.append(e.event_id))
        report = worker.run_once()
        self.assertEqual(1, report.acknowledged)
        self.assertEqual(1, len(received))
        self.assertEqual(0, self.g.outbox.status()["pending"])

    def test_dispatch_exception_no_fake_delivery(self):
        self.revision_event()
        handled = []
        def reject(_):
            raise RuntimeError("private diagnostic SHOULD NOT BE LOGGED")
        worker = OutboxDispatcher(self.g.outbox, reject, on_failure=lambda event, code: handled.append(code))
        result = worker.run_once()
        self.assertEqual(1, result.failed)
        self.assertEqual(("consumer_exception",), tuple(handled))
        self.assertEqual(0, self.g.outbox.status()["delivered"])

    def test_diagnostics_do_not_claim_green_on_dead_letter(self):
        self.revision_event()
        evt = self.g.outbox.lease()[0]
        self.g.outbox.max_attempts = 1
        self.g.outbox.fail(evt.event_id, evt.lease_token)
        report = self.g.diagnostics.inspect(min_free_bytes=1)
        self.assertEqual("degraded", report.status)
        self.assertGreaterEqual(report.outbox_dead_letter, 1)

    def test_online_backup_consistent_and_validated(self):
        entry = self.create("backup data")
        backup = self.g.backup.create(self.tmp/"backup.sqlite3")
        self.assertTrue(self.g.backup.validate(backup))
        with sqlite3.connect(backup) as db:
            self.assertEqual(1, db.execute("SELECT COUNT(*) FROM knowledge_heads").fetchone()[0])
        with self.assertRaises(ValidationFailure):
            self.g.backup.create(backup)

    def test_publisher_update_must_be_reviewed_no_automatic_standard_change(self):
        edition = self.g.standards.register_edition(standard_id="FSA-100", edition="2024",
            title="Fictional Standard", publisher="Fictional Publishing House",
            source_id="source:fixture", access=self.admin, publication_status="published")
        proposal = self.g.publisher_updates.propose(
             edition_record_id=edition.record_id, observed_status="withdrawn",
             observation_source_id="source:observation", observation_evidence_id="evidence:meta",
             observation_method="authorized provider fixture", access=self.admin)
        self.assertEqual("published", self.g.repository.get(edition.record_id, self.admin).draft.payload["publication_status"])
        second = self.g.publisher_updates.review(proposal.record_id, 1,
            reviewer_evidence_id="review:observation", access=AccessContext("independent",
                can_write=True, can_publish_global=True, can_review=True),
            decision="accept_observation", reason="verified metadata separately")
        self.assertEqual(Lifecycle.REVIEWED, second.draft.status)
        self.assertEqual(edition.revision, self.g.repository.get(edition.record_id, self.admin).revision)

    def test_observer_cannot_review_own_submission(self):
        ed = self.g.standards.register_edition(standard_id="FSA-200", edition="2024",
            title="Fictional", publisher="Fictional", source_id="source:fixture", access=self.admin)
        proposal = self.g.publisher_updates.propose(
            edition_record_id=ed.record_id, observed_status="withdrawn",
            observation_source_id="source:obs", observation_evidence_id="evidence:obs",
            observation_method="manual", access=self.admin)
        with self.assertRaises(AuthorizationDenied):
            self.g.publisher_updates.review(proposal.record_id, 1,
                reviewer_evidence_id="review:bad", access=self.admin,
                decision="accept_observation", reason="same actor")


class PerformanceAndInteropTests(Fixture):
    def test_concurrent_writes_are_serialized_without_lost_updates(self):
        from _de_fixture_utils import run_load_probe
        outcome = run_load_probe(worker_count=5, read_operations=15)
        self.assertTrue(outcome.integrity_ok)
        self.assertEqual(1, outcome.successful_conflicting_writes)
        self.assertEqual(4, outcome.correctly_rejected_conflicts)
        self.assertEqual((), outcome.errors)

    def test_reviewed_relation_alignment_is_constrained(self):
        from _de_fixture_utils import RelationAlignment, validate_alignments
        record = RelationAlignment("depends_on", "https://example.org/ontology/dependsOn",
                                   "candidate_equivalence", "review:2026")
        result = validate_alignments({"depends_on": record})
        self.assertEqual(record, result[0])
        with self.assertRaises(ValidationFailure):
            validate_alignments({"unsafe_relation": record})


class DurableRecoveryTests(Fixture):
    def test_expired_lease_reacquirable_with_new_token(self):
        from assurance.contracts import utc_now
        from assurance.repository import _plus_seconds
        self.create("outbox source")
        source = self.create("outbox object")
        self.g.repository.revise(source.record_id, 1,
             replace(source.draft, payload={"statement": "new"}), self.admin, reason="change")
        start = utc_now()
        lease = self.g.outbox.lease(lease_seconds=1, now=start)[0]
        expired = _plus_seconds(start, 3)
        self.assertFalse(self.g.outbox.ack(lease.event_id, lease.lease_token, now=expired))
        repeat = self.g.outbox.lease(now=expired)[0]
        self.assertEqual(lease.event_id, repeat.event_id)
        self.assertNotEqual(lease.lease_token, repeat.lease_token)
        self.assertEqual(2, repeat.attempts)