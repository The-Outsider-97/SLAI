"""Authorized, capability-bound execution of revision workflow computations.

The existing OrchestratorAgent performs specialist delegation. This adapter
adds durable plan lifecycle checks without copying its agent-factory logic.
Computations produce proposals; external document commits are never inferred.
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .revision_workflow import RevisionWorkflowError, RevisionWorkflowStore


class WorkflowAuthorizationError(PermissionError):
    """A caller is missing a required project-scoped authority."""


class WorkflowDispatchError(RuntimeError):
    """A specialist failed or returned an ambiguous execution result."""


@dataclass(frozen=True)
class AuthorizationContext:
    """Trusted host-issued authorization; never construct from agent output."""

    actor_id: str
    project_id: str
    scopes: frozenset[str]
    expires_at: Optional[float] = None

    def require(self, project_id: str, *scopes: str) -> None:
        if not self.actor_id.strip() or self.project_id != project_id:
            raise WorkflowAuthorizationError("actor or project authorization mismatch")
        if self.expires_at is not None and time.time() >= self.expires_at:
            raise WorkflowAuthorizationError("authorization has expired")
        missing = set(scopes) - self.scopes
        if missing:
            raise WorkflowAuthorizationError(f"required authorization scopes missing: {sorted(missing)}")


@dataclass(frozen=True)
class CapabilityBinding:
    """Trusted, preconfigured mapping from a capability to a registered agent."""

    specialty: str
    action: str = "perform_task"
    required_scope: str = "workflow:execute"

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip()
                   for value in (self.specialty, self.action, self.required_scope)):
            raise ValueError("capability binding fields must be nonempty strings")
        if self.action != "perform_task":
            raise ValueError("capability binding only permits the audited perform_task entrypoint")


class OrchestratorCapabilityAdapter:
    """Strict adapter over the actual OrchestratorAgent.perform_task boundary.

    Unknown capabilities and unregistered specialties are rejected, instead of
    using the Orchestrator's dynamic specialist-generation fallback.
    """

    def __init__(self, orchestrator: Any, bindings: Mapping[str, CapabilityBinding], *,
                 registered_specialties: set[str] | frozenset[str]) -> None:
        self.orchestrator = orchestrator
        self.bindings = dict(bindings)
        self.registered_specialties = frozenset(registered_specialties)
        for capability, binding in self.bindings.items():
            if not capability or not isinstance(binding, CapabilityBinding):
                raise ValueError("invalid capability registry binding")

    def execute(self, capability: str, request: Mapping[str, Any], auth: AuthorizationContext) -> dict[str, str]:
        binding = self.bindings.get(capability)
        if binding is None or binding.specialty not in self.registered_specialties:
            raise WorkflowDispatchError("no approved registered specialist for capability")
        auth.require(auth.project_id, binding.required_scope)
        payload = {
            "specialty": binding.specialty,
            "action": binding.action,
            "task_data": {"task_data": dict(request)},
        }
        result = self.orchestrator.perform_task(payload)
        if not isinstance(result, Mapping):
            raise WorkflowDispatchError("specialist did not return a structured result")
        status = str(result.get("status", "")).lower()
        if status not in {"success", "succeeded", "completed", "ok"}:
            raise WorkflowDispatchError(f"specialist result not confirmed successful: {status or 'unknown'}")
        refs = result.get("output_refs", {})
        if not isinstance(refs, Mapping) or any(not isinstance(key, str) or not isinstance(value, str)
                                                or not key or not value for key, value in refs.items()):
            raise WorkflowDispatchError("specialist returned invalid artifact revision references")
        return dict(refs)


class RevisionWorkflowRunner:
    """Bounded ready-task executor using a trusted specialist adapter.

    The runner never calls an external commit API, never interprets output
    strings as commands, and does not skip verification or approval gates.
    """

    def __init__(self, workflows: RevisionWorkflowStore, dispatcher: OrchestratorCapabilityAdapter,
                 *, max_parallel: int = 4, max_rounds: int = 100) -> None:
        if max_parallel < 1 or max_rounds < 1:
            raise ValueError("execution limits must be positive")
        self.workflows = workflows
        self.dispatcher = dispatcher
        self.max_parallel = max_parallel
        self.max_rounds = max_rounds

    def _run_task(self, plan_id: str, task_id: str, payload: Mapping[str, Any],
                  auth: AuthorizationContext) -> dict[str, Any]:
        state = self.workflows.get_plan(plan_id)
        specification = state["tasks"][task_id]
        capability = specification.get("capability")
        if not capability:
            return {"task_id": task_id, "status": "blocked", "reason": "no capability specified"}
        binding = self.dispatcher.bindings.get(capability)
        if binding is None or binding.specialty not in self.dispatcher.registered_specialties:
            return {"task_id": task_id, "status": "blocked", "reason": "capability not registered"}
        # Authorize before claiming. An untrusted task payload cannot grant
        # permission to dispatch or change the trusted specialty/action.
        try:
            auth.require(state["project_id"], "workflow:execute", binding.required_scope)
        except WorkflowAuthorizationError:
            return {"task_id": task_id, "status": "blocked", "reason": "not authorized"}
        try:
            attempt = self.workflows.claim_task(plan_id, task_id)
        except RevisionWorkflowError:
            return {"task_id": task_id, "status": "blocked", "reason": "task claim rejected"}
        request = {"project_id": state["project_id"], "plan_id": plan_id,
                   "task_id": task_id, "run_id": attempt["run_id"],
                   "plan_revision": attempt["plan_revision"],
                   "input_revisions": attempt["input_revisions"],
                   "input": dict(payload)}
        try:
            refs = self.dispatcher.execute(capability, request, auth)
            updated = self.workflows.complete_task(plan_id, task_id, attempt["run_id"], output_refs=refs)
            return {"task_id": task_id, "run_id": attempt["run_id"], "status": updated["status"][task_id],
                    "output_refs": refs}
        except RevisionWorkflowError:
            # Stale results cannot be certified by the execution runner.
            return {"task_id": task_id, "run_id": attempt["run_id"], "status": "stale"}
        except Exception as exc:
            try:
                updated = self.workflows.complete_task(
                    plan_id, task_id, attempt["run_id"], error=f"{type(exc).__name__}: delegated task failed"
                )
            except RevisionWorkflowError:
                return {"task_id": task_id, "run_id": attempt["run_id"], "status": "stale"}
            return {"task_id": task_id, "run_id": attempt["run_id"],
                    "status": updated["status"][task_id], "error_type": type(exc).__name__}

    def run_until_blocked(self, plan_id: str, payloads: Mapping[str, Mapping[str, Any]],
                          *, auth: AuthorizationContext) -> dict[str, Any]:
        state = self.workflows.get_plan(plan_id)
        auth.require(state["project_id"], "workflow:execute")
        results: list[dict[str, Any]] = []
        rounds = 0
        blocked_this_run: set[str] = set()
        for _ in range(self.max_rounds):
            groups = self.workflows.ready_groups(plan_id, max_parallel=self.max_parallel)
            if not groups:
                break
            # Only the first safe group executes this round. Recalculate graph
            # readiness and revision state before subsequent work starts.
            group = next(([task_id for task_id in batch if task_id not in blocked_this_run]
                          for batch in groups if any(task_id not in blocked_this_run for task_id in batch)), [])
            if not group:
                break
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_parallel, len(group))) as pool:
                futures = [pool.submit(self._run_task, plan_id, task_id, payloads.get(task_id, {}), auth)
                           for task_id in group]
                for future in futures:
                    results.append(future.result())
            rounds += 1
            # Unsupported or unauthorized tasks remain pending. Continue
            # unrelated branches but do not spin on those same tasks.
            blocked_this_run.update(item["task_id"] for item in results[-len(group):]
                                    if item["status"] == "blocked")
        final = self.workflows.get_plan(plan_id)
        pending = [task_id for task_id, status in final["status"].items()
                   if status not in {"succeeded", "verified"}]
        return {"plan_id": plan_id, "plan_revision": final["plan_revision"],
                "rounds": rounds, "task_results": results, "remaining_tasks": sorted(pending),
                "complete": not pending}


__all__ = ["AuthorizationContext", "CapabilityBinding", "OrchestratorCapabilityAdapter",
           "RevisionWorkflowRunner", "WorkflowAuthorizationError", "WorkflowDispatchError"]
