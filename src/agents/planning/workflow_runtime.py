"""Authorized, capability-bound execution of revision workflow computations.

The existing OrchestratorAgent performs specialist delegation. This adapter
adds durable plan lifecycle checks without copying its agent-factory logic.
Computations produce proposals; external document commits are never inferred.
"""

from __future__ import annotations

import concurrent.futures
import time

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional
from jsonschema import Draft202012Validator, ValidationError  # type: ignore

from .utils.planning_errors import *
from .utils.planning_helpers import *
from .revision_workflow import RevisionWorkflowStore
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Revision Workflow Runner")


@dataclass(frozen=True)
class AuthorizationContext:
    """Trusted host-issued authorization; never construct from agent output."""

    actor_id: str
    project_id: str
    scopes: frozenset[str]
    expires_at: Optional[float] = None
    live_scope_check: Optional[Callable[[str, str, str], bool]] = field(default=None, repr=False, compare=False)

    def require(self, project_id: str, *scopes: str) -> None:
        if not self.actor_id.strip() or self.project_id != project_id:
            raise WorkflowAuthorizationError(
                "actor or project authorization mismatch",
                actor_id=self.actor_id,
                project_id=project_id,
                required_scopes=list(scopes),
                denial_reason="actor_project_mismatch",
            )
        if self.expires_at is not None and time.time() >= self.expires_at:
            raise WorkflowAuthorizationError(
                "authorization has expired",
                actor_id=self.actor_id,
                project_id=project_id,
                required_scopes=list(scopes),
                denial_reason="expired",
            )
        missing = set(scopes) - self.scopes
        if missing:
            raise WorkflowAuthorizationError(
                f"required authorization scopes missing: {sorted(missing)}",
                actor_id=self.actor_id,
                project_id=project_id,
                required_scopes=sorted(missing),
                denial_reason="missing_scopes",
            )
        if self.live_scope_check is not None:
            for scope in scopes:
                if not self.live_scope_check(self.actor_id, project_id, scope):
                    raise WorkflowAuthorizationError(
                        "current project policy denied requested scope",
                        actor_id=self.actor_id,
                        project_id=project_id,
                        required_scopes=[scope],
                        denial_reason="live_policy_denied",
                    )


@dataclass(frozen=True)
class CapabilityBinding:
    """Trusted, versioned *computational* capability bound to a factory agent.

    Schemas validate the application-supplied ``input`` and the complete
    specialist result respectively. Mutation capabilities belong to the
    authoritative host commit adapter, not this computation runner.
    """

    specialty: str
    action: str = "perform_task"
    required_scope: str = "workflow:execute"
    version: str = "1.0"
    description: str = ""
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    result_schema: Mapping[str, Any] = field(default_factory=dict)
    side_effects: str = "none"

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip()
                   for value in (self.specialty, self.action, self.required_scope, self.version)):
            raise ValueError("capability binding fields must be nonempty strings")
        if self.action != "perform_task":
            raise ValueError("capability binding only permits the audited perform_task entrypoint")
        if self.side_effects != "none":
            raise ValueError("workflow capability must be computational; mutations require host commit")
        for name, schema in (("input_schema", self.input_schema), ("result_schema", self.result_schema)):
            if not isinstance(schema, Mapping):
                raise ValueError(f"{name} must be a JSON Schema mapping")
            try:
                Draft202012Validator.check_schema(dict(schema))
            except Exception as exc:
                raise ValueError(f"invalid {name}: {getattr(exc, 'message', type(exc).__name__)}") from exc


def _validate_payload(schema: Mapping[str, Any], value: Any, boundary: str, *,
                      capability: Optional[str] = None,
                      specialty: Optional[str] = None) -> None:
    if not schema:
        return
    try:
        Draft202012Validator(dict(schema)).validate(value)
    except ValidationError as exc:
        location = "/".join(str(segment) for segment in exc.absolute_path) or "<root>"
        raise CapabilityContractError(
            f"{boundary} schema rejected {location}: {exc.message}",
            boundary=boundary,
            location=location,
            capability=capability,
            specialty=specialty,
            failure_kind="schema_violation",
        ) from exc


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
            if not isinstance(capability, str) or not capability.strip() or not isinstance(binding, CapabilityBinding):
                raise ValueError("invalid capability registry binding")

    def _available(self, binding: CapabilityBinding) -> bool:
        """Enforce both the configured allowlist and live factory registration."""
        if binding.specialty not in self.registered_specialties:
            return False
        factory = getattr(self.orchestrator, "agent_factory", None)
        if factory is None:
            return True  # Standalone/test adapters use the explicit trusted allowlist.
        return binding.specialty in set(factory.get_registered_agent_types())

    def discover(self, auth: AuthorizationContext) -> dict[str, dict[str, Any]]:
        """Expose only currently eligible, authorized computational contracts."""
        descriptions = {}
        for name, binding in sorted(self.bindings.items()):
            if not self._available(binding):
                continue
            try:
                auth.require(auth.project_id, binding.required_scope)
            except WorkflowAuthorizationError:
                continue
            descriptions[name] = {
                "specialty": binding.specialty, "version": binding.version,
                "description": binding.description,
                "required_scope": binding.required_scope,
                "input_schema": dict(binding.input_schema),
                "result_schema": dict(binding.result_schema),
                "side_effects": binding.side_effects,
            }
        logger.debug("Capabilities discovered actor=%s project=%s count=%d",
                     auth.actor_id, auth.project_id, len(descriptions))
        return descriptions

    def execute(self, capability: str, request: Mapping[str, Any],
                auth: AuthorizationContext) -> dict[str, str]:
        if not isinstance(request, Mapping):
            raise CapabilityContractError(
                "capability request must be a mapping",
                capability=capability,
                boundary="request",
                failure_kind="not_mapping",
            )
        binding = self.bindings.get(capability)
        if binding is None or not self._available(binding):
            raise WorkflowDispatchError(
                "no approved registered specialist for capability",
                capability=capability,
                failure_kind="unregistered_or_unavailable",
            )
        auth.require(auth.project_id, binding.required_scope)
        if request.get("project_id", auth.project_id) != auth.project_id:
            raise WorkflowAuthorizationError(
                "capability request crosses the authorized project boundary",
                actor_id=auth.actor_id,
                project_id=auth.project_id,
                required_scopes=[binding.required_scope],
                denial_reason="project_boundary_violation",
            )
        if request.get("actor_id", auth.actor_id) != auth.actor_id:
            raise WorkflowAuthorizationError(
                "capability request actor identity mismatch",
                actor_id=auth.actor_id,
                project_id=auth.project_id,
                required_scopes=[binding.required_scope],
                denial_reason="actor_identity_mismatch",
            )
        if request.get("capability_version", binding.version) != binding.version:
            raise CapabilityContractError(
                "requested capability version is incompatible",
                capability=capability,
                specialty=binding.specialty,
                boundary="capability_version",
                failure_kind="version_mismatch",
            )
        if not isinstance(request.get("input", {}), Mapping):
            raise CapabilityContractError(
                "capability input must be a mapping",
                capability=capability,
                specialty=binding.specialty,
                boundary="input",
                failure_kind="not_mapping",
            )
        _validate_payload(binding.input_schema, request.get("input", {}), "input",
                          capability=capability, specialty=binding.specialty)
        trusted_request = dict(request)
        trusted_request["project_id"] = auth.project_id
        trusted_request["actor_id"] = auth.actor_id
        payload = {
            "specialty": binding.specialty,
            "action": binding.action,
            "task_data": {"task_data": trusted_request},
        }
        logger.debug("Dispatch capability=%s specialty=%s actor=%s project=%s",
                     capability, binding.specialty, auth.actor_id, auth.project_id)
        result = self.orchestrator.perform_task(payload)
        # Authorization can expire or be revoked by a trusted caller between
        # dispatch and response. Do not release results after such a change.
        auth.require(auth.project_id, binding.required_scope)
        if not isinstance(result, Mapping):
            raise WorkflowDispatchError(
                "specialist did not return a structured result",
                capability=capability,
                specialty=binding.specialty,
                failure_kind="non_mapping_result",
            )
        status = str(result.get("status", "")).lower()
        if status not in {"success", "succeeded", "completed", "ok"}:
            raise WorkflowDispatchError(
                f"specialist result not confirmed successful: {status or 'unknown'}",
                capability=capability,
                specialty=binding.specialty,
                failure_kind="unsuccessful_result",
            )
        _validate_payload(binding.result_schema, result, "result",
                          capability=capability, specialty=binding.specialty)
        refs = result.get("output_refs", {})
        if not isinstance(refs, Mapping) or any(not isinstance(key, str) or not isinstance(value, str)
                                                or not key or not value for key, value in refs.items()):
            raise WorkflowDispatchError(
                "specialist returned invalid artifact revision references",
                capability=capability,
                specialty=binding.specialty,
                failure_kind="invalid_output_refs",
            )
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

        logger.info("Revision workflow runner initialized with max_parallel=%d, max_rounds=%d",
                    max_parallel, max_rounds)

    def _run_task(self, plan_id: str, task_id: str, payload: Mapping[str, Any],
                  auth: AuthorizationContext) -> dict[str, Any]:
        state = self.workflows.get_plan(plan_id)
        specification = state["tasks"][task_id]
        capability = specification.get("capability")
        if not capability:
            logger.debug("Task blocked plan_id=%s task=%s reason=no_capability",
                         plan_id, task_id)
            return {"task_id": task_id, "status": "blocked", "reason": "no capability specified"}
        binding = self.dispatcher.bindings.get(capability)
        if binding is None or not self.dispatcher._available(binding):
            logger.warning("Task blocked plan_id=%s task=%s capability=%s reason=unregistered",
                           plan_id, task_id, capability)
            return {"task_id": task_id, "status": "blocked", "reason": "capability not registered"}
        if (specification.get("capability_version") is not None and
                specification["capability_version"] != binding.version):
            logger.warning("Task blocked plan_id=%s task=%s capability=%s reason=version_mismatch "
                           "(expected=%s, bound=%s)",
                           plan_id, task_id, capability,
                           specification["capability_version"], binding.version)
            return {"task_id": task_id, "status": "blocked", "reason": "capability version mismatch"}
        # Authorize before claiming. An untrusted task payload cannot grant
        # permission to dispatch or change the trusted specialty/action.
        try:
            auth.require(state["project_id"], "workflow:execute", binding.required_scope)
        except WorkflowAuthorizationError as exc:
            logger.warning("Task blocked plan_id=%s task=%s reason=not_authorized: %s",
                           plan_id, task_id, exc)
            return {"task_id": task_id, "status": "blocked", "reason": "not authorized"}
        try:
            attempt = self.workflows.claim_task(plan_id, task_id)
        except RevisionWorkflowError as exc:
            logger.debug("Task claim rejected plan_id=%s task=%s: %s",
                         plan_id, task_id, exc)
            return {"task_id": task_id, "status": "blocked", "reason": "task claim rejected"}
        request = {"project_id": state["project_id"], "plan_id": plan_id,
                   "task_id": task_id, "run_id": attempt["run_id"],
                   "plan_revision": attempt["plan_revision"],
                   "input_revisions": attempt["input_revisions"],
                   "capability_version": binding.version,
                   "input": dict(payload)}
        try:
            refs = self.dispatcher.execute(capability, request, auth)
            updated = self.workflows.complete_task(plan_id, task_id, attempt["run_id"], output_refs=refs)
            logger.info("Task dispatched plan_id=%s task=%s run_id=%s capability=%s status=%s",
                        plan_id, task_id, attempt["run_id"], capability, updated["status"][task_id])
            return {"task_id": task_id, "run_id": attempt["run_id"],
                    "status": updated["status"][task_id], "output_refs": refs}
        except RevisionWorkflowError as exc:
            # Stale results cannot be certified by the execution runner.
            logger.warning("Stale result plan_id=%s task=%s run_id=%s: %s",
                           plan_id, task_id, attempt["run_id"], exc)
            return {"task_id": task_id, "run_id": attempt["run_id"], "status": "stale"}
        except Exception as exc:
            logger.error("Task dispatch failed plan_id=%s task=%s run_id=%s err=%s: %s",
                         plan_id, task_id, attempt["run_id"], type(exc).__name__, exc)
            try:
                updated = self.workflows.complete_task(
                    plan_id, task_id, attempt["run_id"],
                    error=f"{type(exc).__name__}: delegated task failed",
                )
            except RevisionWorkflowError as complete_exc:
                logger.warning("Failed to record task failure plan_id=%s task=%s run_id=%s: %s",
                               plan_id, task_id, attempt["run_id"], complete_exc)
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
            logger.debug("Round %d plan_id=%s group=%s", rounds, plan_id, group)
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
        logger.info("Run complete plan_id=%s rounds=%d remaining=%d complete=%s",
                    plan_id, rounds, len(pending), not pending)
        return {"plan_id": plan_id, "plan_revision": final["plan_revision"],
                "rounds": rounds, "task_results": results, "remaining_tasks": sorted(pending),
                "complete": not pending}


__all__ = ["AuthorizationContext", "CapabilityBinding", "CapabilityContractError",
           "OrchestratorCapabilityAdapter", "RevisionWorkflowRunner",
           "WorkflowAuthorizationError", "WorkflowDispatchError"]