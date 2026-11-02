"""Durable revision evidence and Planning→Provenance event replay."""

import pytest

from src.agents.planning.revision_workflow import RevisionTask, RevisionWorkflowStore
from src.agents.provenance.provenance_store import ProvenanceStore
from src.agents.provenance.provenance_types import RevisionEvidence
from src.agents.provenance.utils.provenance_errors import (
    ProvenanceConflictError, ProvenanceValidationError,
)


def test_revision_events_are_immutable_and_queryable_after_restart(tmp_path):
    path = tmp_path / "provenance.json"
    store = ProvenanceStore(storage_path=path)
    event = RevisionEvidence(event_id="event-1", event_type="task.completed",
                             project_id="project-1", plan_id="plan-1", task_id="task-1",
                             artifact_id="model", revision_id="v2")
    expected = store.save_revision_event(event)
    assert store.save_revision_event(event) == expected
    assert store.get_revision_events(artifact_id="model", plan_id="plan-1") == [expected]
    assert store.get_provenance("model")["revision_events"] == [expected]
    modified = store.get_revision_events(artifact_id="model")[0]
    modified["metadata"]["unexpected"] = "mutation"
    assert store.get_revision_events(artifact_id="model") == [expected]
    assert ProvenanceStore(storage_path=path).get_revision_events(plan_id="plan-1") == [expected]
    with pytest.raises(ProvenanceConflictError):
        store.save_revision_event(RevisionEvidence(event_id="event-1", event_type="task.failed",
                                                   project_id="project-1"))


def test_commit_evidence_cannot_be_claimed_without_host_acknowledgment(tmp_path):
    store = ProvenanceStore(storage_path=tmp_path / "provenance.json")
    with pytest.raises(ProvenanceValidationError):
        store.save_revision_event(RevisionEvidence(event_id="c1", event_type="commit.confirmed",
                                                   project_id="project", artifact_id="drawing", revision_id="v3"))
    recorded = store.save_revision_event(RevisionEvidence(
        event_id="c1", event_type="commit.confirmed", project_id="project",
        artifact_id="drawing", revision_id="v3", previous_revision_id="v2",
        commit_id="host-transaction-98",
    ))
    assert recorded["commit_id"] == "host-transaction-98"
    assert len(store.get_revision_events(artifact_id="drawing")) == 1
    assert store.get_revision_graph("drawing")["committed_transitions"] == [{
        "previous_revision_id": "v2", "revision_id": "v3",
        "commit_id": "host-transaction-98", "event_id": "c1",
    }]
    with pytest.raises(ProvenanceValidationError, match="own parent"):
        RevisionEvidence(event_id="bad", event_type="commit.confirmed", project_id="project",
                         artifact_id="drawing", revision_id="v3",
                         previous_revision_id="v3", commit_id="bad-commit")


def test_older_schema_stores_gain_revision_table_without_losing_records(tmp_path):
    store = ProvenanceStore(storage_path=tmp_path / "provenance.json")
    store.save_entity({"entity_id": "legacy-artifact"})
    state = store.snapshot()
    del state["revision_events"]
    from src.agents.provenance.utils.provenance_helpers import atomic_write_provenance_json
    atomic_write_provenance_json(store.storage_path, state)
    reopened = ProvenanceStore(storage_path=store.storage_path)
    assert reopened.get_entity("legacy-artifact")["entity_id"] == "legacy-artifact"
    assert reopened.get_revision_events() == []


def test_workflow_event_replay_is_idempotent(tmp_path):
    planning = RevisionWorkflowStore(tmp_path / "plans.db")
    provenance = ProvenanceStore(storage_path=tmp_path / "provenance.json")

    class Receiver:
        def record_event(self, event):
            return provenance.save_revision_event(event["payload"])

    state = planning.create_plan("project", [RevisionTask("render", reads={"document": "v1"})],
                                 {"document": "v1"})
    plan_id = state["plan_id"]
    attempt = planning.claim_task(plan_id, "render")
    planning.complete_task(plan_id, "render", attempt["run_id"], output_refs={"preview": "v2"})
    sent = planning.publish_provenance_events(plan_id, Receiver())
    assert sent == 3
    assert planning.publish_provenance_events(plan_id, Receiver()) == 3
    events = provenance.get_revision_events(plan_id=plan_id)
    assert len(events) == 3
    assert provenance.get_revision_events(artifact_id="preview")[0]["revision_id"] == "v2"
    assert not any(event["event_type"] == "commit.confirmed" for event in events)


def test_provenance_agent_record_event_and_query_facade(tmp_path, monkeypatch):
    import src.agents.provenance_agent as module

    monkeypatch.setattr(module, "ProvenanceStore", lambda: ProvenanceStore(storage_path=tmp_path / "facts.json"))
    agent = module.ProvenanceAgent(None, None, config={"publish_shared_memory": False})
    try:
        event = {"event_type": "revision_event", "event_id": "stable-1", "payload": {
            "event_type": "task.completed", "project_id": "project", "plan_id": "p1",
            "task_id": "t1", "artifact_id": "drawing", "revision_id": "r2",
            "timestamp": "2026-11-02T12:00:00Z",
        }}
        result = agent.record_event(event)
        assert result["event_id"] == "stable-1"
        assert agent.record_event(event) == result
        assert agent.perform_task({"operation": "get_revision_evidence", "payload": {
            "artifact_id": "drawing",
        }}) == [result]
        assert agent.perform_task({"operation": "get_revision_graph", "payload": {
            "artifact_id": "drawing",
        }})["committed_transitions"] == []
        planning = RevisionWorkflowStore(tmp_path / "plans.db")
        workflow = planning.create_plan("project", [RevisionTask("analyze")], {"document": "r1"})
        assert planning.publish_provenance_events(workflow["plan_id"], agent) == 1
        assert planning.publish_provenance_events(workflow["plan_id"], agent) == 1
        assert len(agent.get_revision_evidence(plan_id=workflow["plan_id"])) == 1
    finally:
        agent.shutdown()
