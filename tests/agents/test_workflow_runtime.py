"""End-to-end workflow orchestration against the existing delegation contract."""

import threading

import pytest

from src.agents.planning.revision_workflow import RevisionTask, RevisionWorkflowStore
from src.agents.planning.workflow_runtime import (
    AuthorizationContext, CapabilityBinding, CapabilityContractError,
    OrchestratorCapabilityAdapter, RevisionWorkflowRunner, WorkflowAuthorizationError,
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


def test_typed_capability_contract_and_authorized_discovery(tmp_path):
    binding = CapabilityBinding(
        "stem", version="2.0", description="Compute a numeric result",
        input_schema={"type": "object", "properties": {"value": {"type": "integer"}},
                      "required": ["value"], "additionalProperties": False},
        result_schema={"type": "object", "properties": {"status": {"const": "success"},
                      "output_refs": {"type": "object"}}, "required": ["status", "output_refs"]},
    )
    adapter = OrchestratorCapabilityAdapter(FakeOrchestrator(), {"calculate": binding},
                                            registered_specialties={"stem"})
    assert adapter.discover(auth("workflow:execute"))["calculate"]["version"] == "2.0"
    assert adapter.discover(auth()) == {}
    assert adapter.execute("calculate", {"input": {"value": 7}, "task_id": "x", "capability_version": "2.0"},
                           auth("workflow:execute")) == {"x": "result-r1"}
    with pytest.raises(CapabilityContractError, match="input schema"):
        adapter.execute("calculate", {"input": {"value": "7"}}, auth("workflow:execute"))
    with pytest.raises(CapabilityContractError, match="version"):
        adapter.execute("calculate", {"input": {"value": 7}, "capability_version": "3.0"},
                        auth("workflow:execute"))


def test_invalid_output_fails_execution_and_preserves_provenance(tmp_path):
    class BadOrchestrator:
        def perform_task(self, request):
            return {"status": "success", "output_refs": {}, "verified": False}

    binding = CapabilityBinding("stem", result_schema={"type": "object",
                                "properties": {"verified": {"const": True}}, "required": ["verified"]})
    adapter = OrchestratorCapabilityAdapter(BadOrchestrator(), {"calculate": binding},
                                            registered_specialties={"stem"})
    store = RevisionWorkflowStore(tmp_path / "workflow.db")
    plan = store.create_plan("project", [RevisionTask("a", capability="calculate", capability_version="1.0")], {})
    result = RevisionWorkflowRunner(store, adapter).run_until_blocked(plan["plan_id"], {},
                                                                     auth=auth("workflow:execute"))
    assert result["task_results"][0]["status"] == "failed"
    attempt = store.get_plan(plan["plan_id"])["attempts"]["a"][0]
    assert attempt["status"] == "failed"
    assert attempt["run_id"] == result["task_results"][0]["run_id"]


def test_invalid_contracts_and_mutation_bindings_are_refused():
    with pytest.raises(ValueError, match="input_schema"):
        CapabilityBinding("stem", input_schema={"type": "not-a-json-schema-type"})
    with pytest.raises(ValueError, match="mutations require host commit"):
        CapabilityBinding("stem", side_effects="write")
    with pytest.raises(ValueError, match="capability_version requires"):
        RevisionTask("x", capability_version="2.0")


def test_capability_dispatch_rejects_cross_project_and_actor_impersonation():
    orchestrator = FakeOrchestrator()
    adapter = OrchestratorCapabilityAdapter(orchestrator,
        {"read": CapabilityBinding("reader")}, registered_specialties={"reader"})
    principal = auth("workflow:execute")
    with pytest.raises(WorkflowAuthorizationError, match="project boundary"):
        adapter.execute("read", {"project_id": "another-project"}, principal)
    with pytest.raises(WorkflowAuthorizationError, match="actor identity"):
        adapter.execute("read", {"actor_id": "impersonated-user"}, principal)
    assert not orchestrator.calls
    adapter.execute("read", {"task_id": "a"}, principal)
    assert orchestrator.calls[0]["task_data"]["task_data"]["actor_id"] == "trusted-user"
    assert orchestrator.calls[0]["task_data"]["task_data"]["project_id"] == "project"


def test_capability_version_mismatch_blocks_dispatch_before_claim(tmp_path):
    store = RevisionWorkflowStore(tmp_path / "workflow.db")
    plan = store.create_plan("project", [RevisionTask("a", capability="calculate", capability_version="2.0")], {})
    orchestrator = FakeOrchestrator()
    result = runner(store, orchestrator).run_until_blocked(plan["plan_id"], {}, auth=auth("workflow:execute"))
    assert result["task_results"][0]["reason"] == "capability version mismatch"
    assert store.get_plan(plan["plan_id"])["status"]["a"] == "pending"
    assert not orchestrator.calls


def test_live_authorization_revocation_blocks_result_release(tmp_path):
    allowed = {"workflow:execute"}
    credentials = AuthorizationContext(
        "alice", "project", frozenset(allowed),
        live_scope_check=lambda actor, project, scope: scope in allowed,
    )
    store = RevisionWorkflowStore(tmp_path / "workflow.db")
    plan = store.create_plan("project", [RevisionTask("a", capability="calculate")], {})
    orchestrator = FakeOrchestrator(on_task=lambda task: allowed.clear())
    result = runner(store, orchestrator).run_until_blocked(plan["plan_id"], {}, auth=credentials)
    assert result["task_results"][0]["status"] == "failed"
    assert "output_refs" not in result["task_results"][0]
    assert len(orchestrator.calls) == 1
