"""End-to-end workflow orchestration against the existing delegation contract."""

import threading

import pytest

from src.agents.planning.revision_workflow import RevisionTask, RevisionWorkflowStore
from src.agents.planning.workflow_runtime import (
    AuthorizationContext, CapabilityBinding, OrchestratorCapabilityAdapter,
    RevisionWorkflowRunner, WorkflowAuthorizationError,
)


class FakeOrchestrator:
    def __init__(self, on_task=None):
        self.calls = []
        self._lock = threading.Lock()
        self.on_task = on_task

    def perform_task(self, request):
        with self._lock:
            self.calls.append(request)
        task = request["task_data"]["task_data"]
        if self.on_task:
            self.on_task(task)
        return {"status": "success", "output_refs": {task["task_id"]: "result-r1"}}


def runner(store, orchestrator, *, registered=frozenset({"stem", "reader"})):
    adapter = OrchestratorCapabilityAdapter(
        orchestrator, {"calculate": CapabilityBinding("stem"),
                       "extract": CapabilityBinding("reader")},
        registered_specialties=registered,
    )
    return RevisionWorkflowRunner(store, adapter)


def auth(*scopes, project="project"):
    return AuthorizationContext("trusted-user", project, frozenset(scopes))


def test_real_delegation_contract_and_parallel_dependencies(tmp_path):
    store = RevisionWorkflowStore(tmp_path / "workflow.db")
    tasks = [RevisionTask("a", reads={"x": "1"}, capability="calculate"),
             RevisionTask("b", reads={"y": "1"}, capability="extract"),
             RevisionTask("combine", depends_on=("a", "b"), capability="calculate")]
    plan = store.create_plan("project", tasks, {"x": "1", "y": "1"})
    orchestrator = FakeOrchestrator()
    result = runner(store, orchestrator).run_until_blocked(
        plan["plan_id"], {"a": {"value": 2}}, auth=auth("workflow:execute")
    )
    assert result["complete"] is True
    assert result["rounds"] == 2
    assert sorted(call["task_data"]["task_data"]["task_id"] for call in orchestrator.calls) == [
        "a", "b", "combine"]
    assert all(call["action"] == "perform_task" for call in orchestrator.calls)
    assert store.get_plan(plan["plan_id"])["status"]["combine"] == "succeeded"


def test_verification_and_auth_authority_are_real_gates(tmp_path):
    store = RevisionWorkflowStore(tmp_path / "workflow.db")
    plan = store.create_plan("project", [RevisionTask("a", capability="calculate", verification_required=True),
                                          RevisionTask("b", depends_on=("a",), capability="calculate")], {})
    orchestrator = FakeOrchestrator()
    runtime = runner(store, orchestrator)
    with pytest.raises(WorkflowAuthorizationError):
        runtime.run_until_blocked(plan["plan_id"], {}, auth=auth("workflow:execute", project="other"))
    assert not orchestrator.calls
    result = runtime.run_until_blocked(plan["plan_id"], {}, auth=auth("workflow:execute"))
    assert result["complete"] is False
    assert store.get_plan(plan["plan_id"])["status"]["a"] == "awaiting_verification"
    assert len(orchestrator.calls) == 1
    store.verify_task(plan["plan_id"], "a", accepted=True, evidence_id="qa-accepted")
    assert runtime.run_until_blocked(plan["plan_id"], {}, auth=auth("workflow:execute"))["complete"]


def test_unknown_specialty_does_not_dynamic_onboard(tmp_path):
    store = RevisionWorkflowStore(tmp_path / "workflow.db")
    plan = store.create_plan("project", [RevisionTask("a", capability="calculate")], {})
    orchestrator = FakeOrchestrator()
    result = runner(store, orchestrator, registered=frozenset()).run_until_blocked(
        plan["plan_id"], {}, auth=auth("workflow:execute"))
    assert result["task_results"][0]["status"] == "blocked"
    assert not orchestrator.calls
    assert store.get_plan(plan["plan_id"])["status"]["a"] == "pending"


def test_blocked_capability_does_not_stop_independent_tasks(tmp_path):
    store = RevisionWorkflowStore(tmp_path / "workflow.db")
    plan = store.create_plan("project", [RevisionTask("a", capability="missing"),
                                          RevisionTask("b", capability="calculate")], {})
    orchestrator = FakeOrchestrator()
    result = runner(store, orchestrator).run_until_blocked(
        plan["plan_id"], {}, auth=auth("workflow:execute"))
    assert result["remaining_tasks"] == ["a"]
    assert [call["task_data"]["task_data"]["task_id"] for call in orchestrator.calls] == ["b"]


def test_revision_change_mid_delegation_discards_stale_result(tmp_path):
    store = RevisionWorkflowStore(tmp_path / "workflow.db")
    plan = store.create_plan("project", [RevisionTask("a", reads={"input": "v1"}, capability="calculate")],
                             {"input": "v1"})
    orchestrator = FakeOrchestrator(on_task=lambda task: store.apply_change_set(
        task["plan_id"], {"input": "v2"}))
    result = runner(store, orchestrator).run_until_blocked(
        plan["plan_id"], {}, auth=auth("workflow:execute"))
    assert result["task_results"][0]["status"] == "stale"
    assert store.get_plan(plan["plan_id"])["status"]["a"] == "stale"
    assert not result["complete"]


def test_arbitrary_specialist_methods_not_exposed():
    with pytest.raises(ValueError, match="perform_task"):
        CapabilityBinding("execution", action="delete_project")
