"""The shared commit host: idempotency, the revision check, honest receipts, and its use through SLAI's real ``CommitReconciliationStore``.

The "application" here is a small document with a revision counter, written for the test; the kit itself holds none of that logic.
"""
from __future__ import annotations

import importlib.util
import tempfile
import threading
import unittest
from pathlib import Path

from ..commit_host import Applied, CommitHost, CommitRefused, FileCommitLedger, planning_types
from ..errors import BadRequest

HAVE_PLANNING = importlib.util.find_spec("src.agents.planning.commit_reconciliation") is not None


class Doc:
    """An application's document: one artifact with a revision, and a change that its validator may refuse."""

    def __init__(self) -> None:
        self.revision, self.text, self.applied = 0, "", []
        self.lock = threading.RLock()

    def revisions(self, project: str) -> dict[str, str]:
        return {"doc": str(self.revision)} if project == "p1" else {}

    def apply(self, request) -> Applied:
        if request.proposal_ref == "refused":
            raise CommitRefused("the person has not agreed to this proposal")
        if request.proposal_ref == "invalid":
            raise BadRequest("the title is too long")
        if request.proposal_ref == "boom":
            raise RuntimeError("disk vanished")
        self.revision += 1
        self.applied.append(request.proposal_ref)
        return Applied(str(self.revision), f"change-{self.revision}")


@unittest.skipUnless(HAVE_PLANNING, "needs src/agents/planning/project_adapter.py")
class Host(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.doc = Doc()
        self.ledger = FileCommitLedger(Path(self.tmp.name) / "commits.json")
        self.host = CommitHost(revisions=self.doc.revisions, apply=self.doc.apply, ledger=self.ledger, lock=self.doc.lock)
        self.T = planning_types()

    def request(self, ref="ok", key="k1", revision=None, artifact="doc", project="p1"):
        return self.T.HostCommitRequest(project, artifact, str(self.doc.revision) if revision is None else revision, ref, key)

    def test_it_satisfies_the_adapter_protocol(self) -> None:
        self.assertIsInstance(self.host, self.T.AuthoritativeProjectAdapter)

    def test_a_commit_at_the_expected_revision_is_confirmed_with_the_applications_revision(self) -> None:
        receipt = self.host.commit(self.request())
        self.assertEqual((receipt.status, receipt.committed_revision, receipt.evidence_id), ("confirmed", "1", "change-1"))
        self.assertEqual(self.doc.applied, ["ok"])

    def test_a_stale_revision_is_rejected_and_nothing_is_applied(self) -> None:
        receipt = self.host.commit(self.request(revision="7"))
        self.assertEqual(receipt.status, "rejected"); self.assertIn("revision 7", receipt.reason); self.assertIn("revision 0", receipt.reason)
        self.assertEqual(self.doc.applied, [])

    def test_an_unknown_artifact_or_project_is_rejected(self) -> None:
        self.assertEqual(self.host.commit(self.request(artifact="nope", key="a")).status, "rejected")
        self.assertEqual(self.host.commit(self.request(project="other", key="b")).status, "rejected")
        self.assertIsNone(self.host.lookup_commit("p1", "a"), "a refusal about something the application does not hold is not recorded")
        self.assertIsNone(self.host.lookup_commit("other", "b"))

    def test_the_same_key_is_applied_once_and_answers_identically(self) -> None:
        first = self.host.commit(self.request())
        again = self.host.commit(self.request(revision="0"))                 # the same request as the first
        self.assertEqual(first, again); self.assertEqual(self.doc.applied, ["ok"])

    def test_a_key_reused_for_a_different_request_is_refused_and_does_not_replace_the_decision(self) -> None:
        first = self.host.commit(self.request(ref="a"))
        other = self.host.commit(self.request(ref="b"))
        self.assertEqual(other.status, "rejected"); self.assertIn("different commit request", other.reason)
        self.assertEqual(self.host.lookup_commit("p1", "k1"), first); self.assertEqual(self.doc.applied, ["a"])

    def test_the_applications_refusal_and_validation_errors_become_rejections_with_their_own_words(self) -> None:
        self.assertEqual(self.host.commit(self.request(ref="refused", key="r")).reason, "the person has not agreed to this proposal")
        self.assertEqual(self.host.commit(self.request(ref="invalid", key="i")).reason, "the title is too long")
        self.assertEqual(self.doc.revision, 0)

    def test_an_unclassifiable_failure_is_not_turned_into_a_receipt(self) -> None:
        with self.assertRaises(RuntimeError):
            self.host.commit(self.request(ref="boom", key="x"))
        self.assertIsNone(self.host.lookup_commit("p1", "x"), "nothing was decided, so nothing is claimed")

    def test_lookup_is_scoped_to_the_project_and_survives_a_restart(self) -> None:
        self.host.commit(self.request())
        self.assertIsNone(self.host.lookup_commit("p2", "k1"))
        reopened = CommitHost(revisions=self.doc.revisions, apply=self.doc.apply, ledger=FileCommitLedger(Path(self.tmp.name) / "commits.json"))
        self.assertEqual(reopened.lookup_commit("p1", "k1").status, "confirmed")

    def test_two_commits_for_the_same_revision_cannot_both_win(self) -> None:
        results = []
        def go(key: str) -> None:
            results.append(self.host.commit(self.request(key=key, revision="0")).status)
        threads = [threading.Thread(target=go, args=(f"k{i}",)) for i in range(8)]
        [t.start() for t in threads]; [t.join() for t in threads]
        self.assertEqual(sorted(results), ["confirmed"] + ["rejected"] * 7); self.assertEqual(len(self.doc.applied), 1)


@unittest.skipUnless(HAVE_PLANNING, "needs src/agents/planning/project_adapter.py")
class Atomic(unittest.TestCase):
    """The change and its record are one unit: if recording fails the change is taken back, and a refusal leaves nothing behind."""

    def setUp(self) -> None:
        import contextlib
        self.doc = Doc()
        self.T = planning_types()

        @contextlib.contextmanager
        def transaction():
            snapshot = (self.doc.revision, list(self.doc.applied))
            try:
                yield
            except BaseException:
                self.doc.revision, self.doc.applied = snapshot[0], snapshot[1]
                raise
        self.transaction = transaction

    def host(self, ledger):
        return CommitHost(revisions=self.doc.revisions, apply=self.doc.apply, ledger=ledger, atomic=self.transaction)

    def test_if_the_record_cannot_be_written_the_change_is_taken_back(self) -> None:
        class Broken:
            def get(self, key): return None
            def put(self, key, entry): raise OSError("disk full")
        with self.assertRaises(OSError):
            self.host(Broken()).commit(self.T.HostCommitRequest("p1", "doc", "0", "ok", "k1"))
        self.assertEqual((self.doc.revision, self.doc.applied), (0, []), "applied and unrecorded can never happen")

    def test_a_refusal_after_part_of_the_change_leaves_nothing_behind_and_is_recorded(self) -> None:
        stored = {}
        class Memory:
            def get(self, key): return stored.get(key)
            def put(self, key, entry): stored[key] = entry
        def partial(request):
            self.doc.revision += 5                                    # part of the change is done ...
            raise CommitRefused("a later step was refused")           # ... and then the application says no
        host = CommitHost(revisions=self.doc.revisions, apply=partial, ledger=Memory(), atomic=self.transaction)
        receipt = host.commit(self.T.HostCommitRequest("p1", "doc", "0", "ok", "k2"))
        self.assertEqual((receipt.status, receipt.reason), ("rejected", "a later step was refused"))
        self.assertEqual(self.doc.revision, 0); self.assertEqual(stored["k2"]["receipt"]["status"], "rejected")


@unittest.skipUnless(HAVE_PLANNING, "needs src/agents/planning/commit_reconciliation.py")
class WithTheOutbox(unittest.TestCase):
    """The host driven by SLAI's own durable outbox, including a crash after the change was applied."""

    def setUp(self) -> None:
        from src.agents.planning.commit_reconciliation import CommitReconciliationStore
        from src.agents.planning.workflow_runtime import AuthorizationContext
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.doc = Doc()
        self.host = CommitHost(revisions=self.doc.revisions, apply=self.doc.apply, ledger=FileCommitLedger(Path(self.tmp.name) / "l.json"))
        self.store = CommitReconciliationStore(Path(self.tmp.name) / "outbox.sqlite3", self.host); self.addCleanup(self.store.close)
        self.auth = AuthorizationContext("person", "p1", frozenset({"workflow:commit", "workflow:commit:reconcile"}))
        self.T = planning_types()

    def test_prepare_dispatch_confirms_through_the_outbox(self) -> None:
        req = self.T.HostCommitRequest("p1", "doc", "0", "ok", "k1")
        self.store.prepare(req, auth=self.auth)
        self.assertEqual(self.store.dispatch("k1", auth=self.auth)["status"], "confirmed")
        self.assertEqual(self.doc.applied, ["ok"])

    def test_a_crash_after_the_change_leaves_it_unknown_and_reconciling_finds_it_without_applying_twice(self) -> None:
        req = self.T.HostCommitRequest("p1", "doc", "0", "ok", "k1")
        real = self.host.commit
        def crash_after_commit(request):
            real(request)
            raise ConnectionError("the reply was lost")
        self.host.commit = crash_after_commit                                  # the host did commit; the outbox never heard
        self.store.prepare(req, auth=self.auth)
        self.assertEqual(self.store.dispatch("k1", auth=self.auth)["status"], "outcome_unknown")
        self.host.commit = real
        self.assertEqual(self.store.reconcile("k1", auth=self.auth)["status"], "confirmed")
        self.assertEqual(self.doc.applied, ["ok"], "the change was applied once")

    def test_a_rejection_is_recorded_as_rejected(self) -> None:
        self.store.prepare(self.T.HostCommitRequest("p1", "doc", "5", "ok", "k9"), auth=self.auth)
        done = self.store.dispatch("k9", auth=self.auth)
        self.assertEqual(done["status"], "rejected"); self.assertIn("Stale", done["receipt"]["reason"])


if __name__ == "__main__":
    unittest.main()
