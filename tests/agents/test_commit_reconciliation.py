"""Durable host-owned commits, uncertain outcomes and authorization checks."""

import pytest

from src.agents.planning.commit_reconciliation import (
    CommitReconciliationError, CommitReconciliationStore,
)
from src.agents.planning.project_adapter import (
    CallbackProjectAdapter, HostCommitReceipt, HostCommitRequest,
)
from src.agents.planning.workflow_runtime import AuthorizationContext, WorkflowAuthorizationError
from src.agents.provenance.provenance_store import ProvenanceStore


class Host:
    """Test-only revision owner supporting atomic CAS and durable idempotency lookup."""

    def __init__(self):
        self.revisions = {"artifact": "v1"}
        self.receipts = {}
        self.commit_calls = 0
        self.lose_ack = False

    def current_revisions(self, project_id):
        return dict(self.revisions)

    def commit(self, request):
        self.commit_calls += 1
        if request.idempotency_key in self.receipts:
            return self.receipts[request.idempotency_key]
        if self.revisions[request.artifact_id] == request.expected_revision:
            revision = f"v{self.commit_calls + 1}"
            self.revisions[request.artifact_id] = revision
            receipt = HostCommitReceipt(request.project_id, request.artifact_id,
                                        request.idempotency_key, "confirmed", revision,
                                        f"host-tx-{request.idempotency_key}")
        else:
            receipt = HostCommitReceipt(request.project_id, request.artifact_id,
                                        request.idempotency_key, "rejected", reason="base revision conflict")
        self.receipts[request.idempotency_key] = receipt
        if self.lose_ack:
            raise TimeoutError("host committed but response was lost")
        return receipt

    def lookup_commit(self, project_id, idempotency_key):
        return self.receipts.get(idempotency_key)


def auth(*scopes, project="project"):
    return AuthorizationContext("trusted", project, frozenset(scopes))


def request(key, *, expected="v1"):
    return HostCommitRequest("project", "artifact", expected, "proposal://draft", key)


class ProvenanceCapture:
    def __init__(self):
        self.events = {}

    def record_event(self, value):
        key = value["event_id"]
        self.events.setdefault(key, value)


def test_host_conflicts_and_provenance_only_after_confirmation(tmp_path):
    host = Host()
    store = CommitReconciliationStore(tmp_path / "commits.db", host)
    publisher = ProvenanceCapture()
    credentials = auth("workflow:commit", "workflow:provenance:write")
    store.prepare(request("one"), auth=credentials)
    store.prepare(request("two"), auth=credentials)
    assert store.publish_confirmed("one", publisher, auth=credentials) is False
    assert store.dispatch("one", auth=credentials)["status"] == "confirmed"
    assert store.dispatch("two", auth=credentials)["status"] == "rejected"
    assert store.publish_confirmed("two", publisher, auth=credentials) is False
    assert store.publish_confirmed("one", publisher, auth=credentials) is True
    assert store.publish_confirmed("one", publisher, auth=credentials) is True
    assert list(publisher.events) == ["commit:one"]
    assert publisher.events["commit:one"]["payload"]["revision_id"] == "v2"


def test_lost_acknowledgment_reconciles_without_reexecution(tmp_path):
    path = tmp_path / "commits.db"
    host = Host()
    host.lose_ack = True
    store = CommitReconciliationStore(path, host)
    permissions = auth("workflow:commit", "workflow:commit:reconcile")
    store.prepare(request("lost"), auth=permissions)
    assert store.dispatch("lost", auth=permissions)["status"] == "outcome_unknown"
    store.close()
    recovered = CommitReconciliationStore(path, host)
    assert [item["idempotency_key"] for item in recovered.unresolved("project")] == ["lost"]
    assert recovered.dispatch("lost", auth=permissions)["status"] == "outcome_unknown"
    assert host.commit_calls == 1
    assert recovered.reconcile("lost", auth=permissions)["status"] == "confirmed"
    assert recovered.reconcile("lost", auth=permissions)["status"] == "confirmed"
    assert host.commit_calls == 1


def test_no_host_evidence_is_not_a_failed_or_successful_commit(tmp_path):
    host = Host()
    store = CommitReconciliationStore(tmp_path / "commits.db", host)
    permissions = auth("workflow:commit", "workflow:commit:reconcile")
    store.prepare(request("pending"), auth=permissions)
    assert store._transition("pending", "pending", "outcome_unknown")  # interrupted after outbox claim
    assert store.reconcile("pending", auth=permissions)["status"] == "outcome_unknown"
    assert host.commit_calls == 0
    assert store.dispatch("pending", auth=permissions)["status"] == "outcome_unknown"


def test_commit_security_and_conflicting_idempotency_key(tmp_path):
    host = Host()
    store = CommitReconciliationStore(tmp_path / "commits.db", host)
    with pytest.raises(WorkflowAuthorizationError):
        store.prepare(request("one"), auth=auth("workflow:execute"))
    assert host.commit_calls == 0
    with pytest.raises(WorkflowAuthorizationError):
        store.prepare(request("one"), auth=auth("workflow:commit", project="another"))
    store.prepare(request("one"), auth=auth("workflow:commit"))
    with pytest.raises(CommitReconciliationError, match="reused"):
        store.prepare(request("one", expected="v3"), auth=auth("workflow:commit"))
    with pytest.raises(WorkflowAuthorizationError):
        store.reconcile("one", auth=auth("workflow:commit"))


def test_callback_adapter_uses_real_host_contract():
    host = Host()
    adapter = CallbackProjectAdapter(revisions=host.current_revisions, commit=host.commit,
                                     lookup=host.lookup_commit)
    assert adapter.current_revisions("project") == {"artifact": "v1"}
    receipt = adapter.commit(request("adapter"))
    assert receipt.status == "confirmed"
    assert adapter.lookup_commit("project", "adapter") == receipt


def test_confirmed_host_receipt_reaches_real_provenance_graph(tmp_path):
    host = Host()
    store = CommitReconciliationStore(tmp_path / "outbox.db", host)
    provenance = ProvenanceStore(storage_path=tmp_path / "facts.json")

    class ProvenanceReceiver:
        def record_event(self, event):
            return provenance.save_revision_event(event["payload"])

    credentials = auth("workflow:commit", "workflow:provenance:write")
    store.prepare(request("host-tx-1"), auth=credentials)
    assert store.dispatch("host-tx-1", auth=credentials)["status"] == "confirmed"
    assert store.publish_confirmed("host-tx-1", ProvenanceReceiver(), auth=credentials)
    assert store.publish_confirmed("host-tx-1", ProvenanceReceiver(), auth=credentials)
    assert provenance.get_revision_graph("artifact")["committed_transitions"] == [{
        "previous_revision_id": "v1", "revision_id": "v2",
        "commit_id": "host-tx-host-tx-1", "event_id": "commit:host-tx-1",
    }]
