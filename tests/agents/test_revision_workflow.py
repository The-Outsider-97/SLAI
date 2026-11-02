"""Revision-aware planning contracts (no application or model dependencies)."""

import pytest

from src.agents.planning.revision_workflow import (
    RevisionTask, RevisionWorkflowError, RevisionWorkflowStore,
)


def test_rejects_cycles_missing_dependencies_and_duplicate_ids(tmp_path):
    store = RevisionWorkflowStore(tmp_path / "plans.db")
    with pytest.raises(RevisionWorkflowError, match="missing dependencies"):
        store.create_plan("project", [RevisionTask("x", depends_on=("missing",))], {})
    with pytest.raises(RevisionWorkflowError, match="cyclic"):
        store.create_plan("project", [RevisionTask("a", ("b",)), RevisionTask("b", ("a",))], {})
    with pytest.raises(RevisionWorkflowError, match="distinct"):
        store.create_plan("project", [RevisionTask("a"), RevisionTask("a")], {})


def test_parallel_groups_respect_conflicting_writes(tmp_path):
    store = RevisionWorkflowStore(tmp_path / "plans.db")
    tasks = [
        RevisionTask("a", reads={"x": "r1"}, writes=("z",)),
        RevisionTask("b", reads={"y": "r1"}, writes=("w",)),
        RevisionTask("c", reads={"x": "r1"}, writes=("z",)),
        RevisionTask("d", depends_on=("a",)),
    ]
    plan = store.create_plan("project", tasks, {"x": "r1", "y": "r1"})
    groups = store.ready_groups(plan["plan_id"])
    assert groups == [["a", "b"], ["c"]]
    a = store.claim_task(plan["plan_id"], "a")
    store.complete_task(plan["plan_id"], "a", a["run_id"])
    assert "d" in [item for group in store.ready_groups(plan["plan_id"]) for item in group]


def test_selective_invalidation_replan_and_history(tmp_path):
    store = RevisionWorkflowStore(tmp_path / "plans.db")
    tasks = [
        RevisionTask("geometry", reads={"hinge": "v1"}, writes=("geometry",)),
        RevisionTask("trajectory", depends_on=("geometry",), reads={"hinge": "v1"}),
        RevisionTask("documentation", reads={"title": "v1"}),
    ]
    plan = store.create_plan("project", tasks, {"hinge": "v1", "title": "v1"})
    plan_id = plan["plan_id"]
    for name in ("geometry", "trajectory", "documentation"):
        attempt = store.claim_task(plan_id, name)
        store.complete_task(plan_id, name, attempt["run_id"])
    changed = store.apply_change_set(plan_id, {"hinge": "v2"})
    assert changed["status"] == {"geometry": "stale", "trajectory": "stale", "documentation": "succeeded"}
    assert [group for group in store.ready_groups(plan_id)] == []
    replanned = store.replan_stale(plan_id)
    assert replanned["plan_revision"] == 3
    assert replanned["tasks"]["geometry"]["reads"] == {"hinge": "v2"}
    assert replanned["status"]["documentation"] == "succeeded"
    assert len(replanned["attempts"]["geometry"]) == 1
    assert store.ready_groups(plan_id) == [["geometry"]]
    assert any(event["event_type"] == "plan.invalidated" for event in store.events(plan_id))


def test_running_result_rejected_after_input_change(tmp_path):
    store = RevisionWorkflowStore(tmp_path / "plans.db")
    plan = store.create_plan("project", [RevisionTask("simulation", reads={"model": "1"})], {"model": "1"})
    plan_id = plan["plan_id"]
    attempt = store.claim_task(plan_id, "simulation")
    store.apply_change_set(plan_id, {"model": "2"})
    with pytest.raises(RevisionWorkflowError, match="not running"):
        store.complete_task(plan_id, "simulation", attempt["run_id"])
    assert store.get_plan(plan_id)["attempts"]["simulation"][0]["status"] == "stale"


def test_verification_is_required_before_downstream_task(tmp_path):
    store = RevisionWorkflowStore(tmp_path / "plans.db")
    plan = store.create_plan("project", [
        RevisionTask("calculation", verification_required=True),
        RevisionTask("report", depends_on=("calculation",)),
    ], {"document": "1"})
    plan_id = plan["plan_id"]
    attempt = store.claim_task(plan_id, "calculation")
    store.complete_task(plan_id, "calculation", attempt["run_id"])
    assert store.ready_groups(plan_id) == []
    store.verify_task(plan_id, "calculation", accepted=True, evidence_id="check-42")
    assert store.ready_groups(plan_id) == [["report"]]


def test_workflow_reopens_with_attempts_without_reexecution(tmp_path):
    path = tmp_path / "plans.db"
    store = RevisionWorkflowStore(path)
    plan = store.create_plan("project", [RevisionTask("task")], {"project": "1"})
    task_id = plan["plan_id"]
    attempt = store.claim_task(task_id, "task")
    store.complete_task(task_id, "task", attempt["run_id"], output_refs={"artifact": "v2"})
    store.close()
    recovered = RevisionWorkflowStore(path)
    assert recovered.get_plan(task_id)["status"]["task"] == "succeeded"
    assert recovered.ready_groups(task_id) == []
    assert [event["event_type"] for event in recovered.events(task_id)] == [
        "plan.created", "task.started", "task.completed",
    ]
    recovered.close()


def test_invalid_transitions_do_not_commit_partial_state(tmp_path):
    store = RevisionWorkflowStore(tmp_path / "plans.db")
    plan = store.create_plan("project", [RevisionTask("a")], {"p": "r1"})
    plan_id = plan["plan_id"]
    before = store.get_plan(plan_id)
    with pytest.raises(RevisionWorkflowError):
        store.complete_task(plan_id, "a", "nonexistent")
    assert store.get_plan(plan_id) == before
    assert len(store.events(plan_id)) == 1


def test_duplicate_change_events_are_noops(tmp_path):
    store = RevisionWorkflowStore(tmp_path / "plans.db")
    state = store.create_plan("project", [RevisionTask("task", reads={"a": "1"})], {"a": "1"})
    plan_id = state["plan_id"]
    before = store.get_plan(plan_id)
    assert store.apply_change_set(plan_id, {"a": "1"}) == before
    assert store.events(plan_id)[0]["event_type"] == "plan.created"
    assert len(store.events(plan_id)) == 1


def test_interrupted_execution_is_not_automatically_retried(tmp_path):
    path = tmp_path / "plans.db"
    store = RevisionWorkflowStore(path)
    plan = store.create_plan("project", [RevisionTask("task")], {"project": "r1"})
    plan_id = plan["plan_id"]
    claim = store.claim_task(plan_id, "task")
    store.close()
    recovery = RevisionWorkflowStore(path)
    recovered = recovery.mark_interrupted_tasks(plan_id)
    assert recovered["status"]["task"] == "outcome_unknown"
    assert recovery.ready_groups(plan_id) == []
    with pytest.raises(RevisionWorkflowError, match="not running"):
        recovery.complete_task(plan_id, "task", claim["run_id"])
    assert recovery.resolve_interrupted_task(plan_id, "task", claim["run_id"],
                                             succeeded=True, evidence_id="host-result-1")["status"]["task"] == "succeeded"
