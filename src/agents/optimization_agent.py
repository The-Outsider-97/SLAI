"""Production SLAI OptimizationAgent orchestration façade.

The agent owns SLAI integration concerns around mathematical optimization:
request normalization, BaseAgent lifecycle participation, agent-level policy,
shared/local-memory coordination, result publication, metrics, checkpoint
references, and failure normalization.  Mathematical modelling, classification,
solver selection, numerical algorithms, Pareto analysis, and solver evidence
remain owned by :mod:`src.agents.optimization`.
"""
from __future__ import annotations

__version__ = "2.3.0"

import math
import time
import uuid

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from threading import RLock
from typing import Any, Dict, Optional, Tuple

from .base.utils.config_contract import ConfigContractError, assert_valid_config_contract
from .base.utils.main_config_loader import get_config_section, load_global_config
from .base_agent import BaseAgent
from .runtime_contracts import RuntimeLifecycle
from .optimization.modules.problem_definition import ProblemDefinition
from .optimization.optimization_memory import OptimizationMemory, OptimizationRunRecord
from .optimization.optimization_solver import OptimizationSolver
from .optimization.optimization_types import *
from .optimization.utils.optimization_errors import *
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Optimization Agent")
printer = PrettyPrinter()


class OptimizationAgent(BaseAgent):
    """SLAI agent-level orchestrator for the mathematical Optimization subsystem."""

    AGENT_KEY = "optimization_agent"
    CHECKPOINTING_SUPPORTED = True
    CHECKPOINT_SCHEMA = "slai.optimization_agent.state.v1"

    _ALLOWED_CONFIG_KEYS = {
        "enabled",
        "strict_validation",
        "publish_shared_results",
        "shared_memory_enabled",
        "shared_result_key_prefix",
        "shared_latest_key",
        "shared_event_topic",
        "shared_memory_ttl_seconds",
        "include_decision_variables_in_shared_memory",
        "max_shared_pareto_points",
        "max_shared_warnings",
        "max_batch_size",
        "metrics_enabled",
        "local_reference_limit",
    }

    _TASK_KEYS = {
        "problem",
        "solver",
        "options",
        "allow_fallback",
        "method",
        "workflow",
        "weight_vectors",
        "epsilon_grid",
        "primary",
        "publish",
        "request_id",
        "metadata",
        "context",
        "warm_start",
        "time_limit_seconds",
        "max_evaluations",
        "max_nodes",
        "node_limit",
        "tasks",
    }
    alternative_execute_is_safe_after_failure = True

    def __init__(
        self,
        shared_memory: Any,
        agent_factory: Any,
        config: Optional[Mapping[str, Any]] = None,
        *,
        checkpoint_manager: Any = None,
    ) -> None:
        # Agent-specific orchestration settings must not leak into BaseAgent's
        # base_agent section. BaseAgent still owns SharedMemory, lifecycle,
        # common telemetry, retries/recovery, identity, and checkpoint plumbing.
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            checkpoint_manager=checkpoint_manager,
        )

        self._optimization_lock = RLock()
        self._metrics_lock = RLock()
        self.config = load_global_config()
        self.optimization_config = get_config_section('optimization_agent') or {}
        if config is not None:
            if not isinstance(config, Mapping):
                raise OptimizationConfigurationError(
                    "OptimizationAgent config override must be a mapping",
                    context={"actual_type": type(config).__name__},
                )
            self.optimization_config.update(dict(config))

        try:
            assert_valid_config_contract(
                global_config=self.config,
                agent_key=self.AGENT_KEY,
                agent_config=self.optimization_config,
                agent_allowed_keys=self._ALLOWED_CONFIG_KEYS,
                required_agent_keys={"enabled"},
                require_global_keys=False,
                require_agent_section=True,
                warn_unknown_global_keys=False,
                logger=logger,
            )
        except ConfigContractError as exc:
            raise OptimizationConfigurationError(
                "OptimizationAgent configuration violates the SLAI agent contract",
                context={"agent_key": self.AGENT_KEY},
                cause=exc,
            ) from exc

        self._load_agent_config()
        self._validate_agent_config()

        # Explicit dual-memory ownership:
        # - self.shared_memory remains BaseAgent-owned and is never replaced.
        # - self.local_memory is the authoritative Optimization evidence store.
        self.local_memory = OptimizationMemory()
        self.problem_definition = ProblemDefinition()
        self.optimization_solver = OptimizationSolver(memory=self.local_memory)

        self._requests_total = 0
        self._requests_success = 0
        self._requests_failed = 0
        self._single_objective_runs = 0
        self._multiobjective_runs = 0
        self._warm_start_runs = 0
        self._fallback_runs = 0
        self._infeasible_runs = 0
        self._unbounded_runs = 0
        self._timeout_runs = 0
        self._limit_runs = 0
        self._numerical_failure_runs = 0
        self._shared_publication_failures = 0
        self._local_memory_persistence_failures = 0
        self._runtime_total_ms = 0.0
        self._solver_usage: Counter[str] = Counter()
        self._termination_counts: Counter[str] = Counter()
        self._failure_types: Counter[str] = Counter()
        self._last_run_id: Optional[str] = None
        self._last_problem_fingerprint: Optional[str] = None
        self._last_local_evidence_ids: Tuple[str, ...] = ()

        self._validate_readiness()
        logger.info(
            "OptimizationAgent initialized | shared_publish=%s | checkpointing=%s",
            self.shared_memory_enabled and self.publish_shared_results,
            self.supports_checkpointing,
        )

    # ------------------------------------------------------------------
    # Agent-level configuration: agents_config.yaml only
    # ------------------------------------------------------------------
    @staticmethod
    def _config_bool(cfg: Mapping[str, Any], key: str, default: bool) -> bool:
        value = cfg.get(key, default)
        if not isinstance(value, bool):
            raise OptimizationConfigurationError(f"optimization_agent.{key} must be boolean", context={"value": value})
        return value

    @staticmethod
    def _config_positive_int(cfg: Mapping[str, Any], key: str, default: int) -> int:
        value = cfg.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise OptimizationConfigurationError(f"optimization_agent.{key} must be a positive integer", context={"value": value})
        return value

    def _load_agent_config(self) -> None:
        cfg = self.optimization_config
        self.enabled = self._config_bool(cfg, "enabled", True)
        self.strict_validation = self._config_bool(cfg, "strict_validation", True)
        self.shared_memory_enabled = self._config_bool(cfg, "shared_memory_enabled", True)
        self.publish_shared_results = self._config_bool(cfg, "publish_shared_results", True)
        self.shared_result_key_prefix = str(cfg.get("shared_result_key_prefix", "optimization_agent.result")).strip()
        self.shared_latest_key = str(cfg.get("shared_latest_key", "optimization_agent.latest")).strip()
        self.shared_event_topic = str(cfg.get("shared_event_topic", "optimization.events")).strip()
        raw_ttl = cfg.get("shared_memory_ttl_seconds", 86400)
        if raw_ttl is None:
            self.shared_memory_ttl_seconds = None
        elif isinstance(raw_ttl, bool) or not isinstance(raw_ttl, (int, float)):
            raise OptimizationConfigurationError(
                "optimization_agent.shared_memory_ttl_seconds must be null or numeric",
                context={"value": raw_ttl},
            )
        else:
            self.shared_memory_ttl_seconds = float(raw_ttl)
        self.include_decision_variables_in_shared_memory = self._config_bool(
            cfg, "include_decision_variables_in_shared_memory", True
        )
        self.max_shared_pareto_points = self._config_positive_int(cfg, "max_shared_pareto_points", 32)
        self.max_shared_warnings = self._config_positive_int(cfg, "max_shared_warnings", 16)
        self.max_batch_size = self._config_positive_int(cfg, "max_batch_size", 16)
        self.metrics_enabled = self._config_bool(cfg, "metrics_enabled", True)
        self.local_reference_limit = self._config_positive_int(cfg, "local_reference_limit", 32)

    def _validate_agent_config(self) -> None:
        if self.shared_memory_ttl_seconds is not None:
            if not math.isfinite(self.shared_memory_ttl_seconds) or self.shared_memory_ttl_seconds <= 0:
                raise OptimizationConfigurationError("shared_memory_ttl_seconds must be null or > 0")
        positive_ints = {
            "max_shared_pareto_points": self.max_shared_pareto_points,
            "max_shared_warnings": self.max_shared_warnings,
            "max_batch_size": self.max_batch_size,
            "local_reference_limit": self.local_reference_limit,
        }
        for name, value in positive_ints.items():
            if isinstance(value, bool) or value < 1:
                raise OptimizationConfigurationError(f"{name} must be a positive integer")
        for name, value in {
            "shared_result_key_prefix": self.shared_result_key_prefix,
            "shared_latest_key": self.shared_latest_key,
            "shared_event_topic": self.shared_event_topic,
        }.items():
            if not value:
                raise OptimizationConfigurationError(f"{name} must be non-empty")

    def _validate_readiness(self) -> None:
        if not isinstance(self.local_memory, OptimizationMemory):
            raise OptimizationConfigurationError("OptimizationAgent local_memory must be OptimizationMemory")
        if not isinstance(self.problem_definition, ProblemDefinition):
            raise OptimizationConfigurationError("OptimizationAgent problem_definition must be ProblemDefinition")
        if self.optimization_solver.memory is not self.local_memory:
            raise OptimizationConfigurationError("OptimizationSolver must share OptimizationAgent.local_memory")
        if self.shared_memory is self.local_memory:
            raise OptimizationConfigurationError("shared_memory and local_memory must remain distinct stores")

    # ------------------------------------------------------------------
    # Agent request contract and canonical problem normalization
    # ------------------------------------------------------------------
    def _validate_task_mapping(self, task_data: Any) -> Mapping[str, Any]:
        if not isinstance(task_data, Mapping):
            raise OptimizationValidationError(
                "OptimizationAgent request must be a mapping",
                context={"actual_type": type(task_data).__name__},
            )
        if self.strict_validation:
            unknown = sorted(set(task_data) - self._TASK_KEYS)
            if unknown:
                raise OptimizationValidationError("Unknown OptimizationAgent request fields", context={"unknown": unknown})
        return task_data

    @staticmethod
    def _normalize_request_id(value: Any) -> str:
        if value is None:
            return f"opt-{uuid.uuid4().hex[:20]}"
        if not isinstance(value, str) or not value.strip():
            raise OptimizationValidationError("request_id must be a non-empty string")
        return value.strip()

    @staticmethod
    def _normalize_metadata(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise OptimizationValidationError("metadata must be a mapping")
        return dict(value)

    @staticmethod
    def _normalize_options(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise OptimizationValidationError("options must be a mapping")
        return dict(value)

    @staticmethod
    def _normalize_context(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise OptimizationValidationError("context must be a mapping")
        return dict(value)

    @staticmethod
    def _merge_execution_constraints(request: Mapping[str, Any], options: Mapping[str, Any]) -> Dict[str, Any]:
        """Merge request-level execution budgets into the subsystem options contract."""
        merged = dict(options)
        numeric_budget_keys = ("time_limit_seconds",)
        integer_budget_keys = ("max_evaluations", "max_nodes", "node_limit")

        for key in numeric_budget_keys:
            raw = request.get(key)
            if raw is None:
                continue
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)) or float(raw) <= 0:
                raise OptimizationValidationError(f"{key} must be a finite number > 0")
            value = float(raw)
            if key in merged and merged[key] != value:
                raise OptimizationValidationError(
                    f"Contradictory {key} values were provided",
                    context={"request_value": value, "options_value": merged[key]},
                )
            merged[key] = value

        for key in integer_budget_keys:
            raw = request.get(key)
            if raw is None:
                continue
            if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
                raise OptimizationValidationError(f"{key} must be a positive integer")
            if key in merged and merged[key] != raw:
                raise OptimizationValidationError(
                    f"Contradictory {key} values were provided",
                    context={"request_value": raw, "options_value": merged[key]},
                )
            merged[key] = raw
        return merged

    @staticmethod
    def _apply_explicit_warm_start(problem: OptimizationProblem, warm_start: Any) -> OptimizationProblem:
        """Apply a solver-neutral warm start through canonical Variable.initial values."""
        if warm_start is None:
            return problem
        if not isinstance(warm_start, Mapping):
            raise OptimizationValidationError("warm_start must be a mapping of variable names to values")
        unknown = sorted(set(warm_start) - set(problem.variable_names))
        if unknown:
            raise OptimizationValidationError(
                "warm_start references unknown variables",
                context={"unknown": unknown},
            )
        variables = []
        for variable in problem.variables:
            if variable.name in warm_start:
                validated = variable.validate_value(warm_start[variable.name])
                variables.append(replace(variable, initial=validated))
            else:
                variables.append(variable)
        return replace(problem, variables=tuple(variables))

    @staticmethod
    def _normalize_solver_name(value: Any) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise OptimizationValidationError("solver must be a non-empty string when provided")
        return value.strip()

    @staticmethod
    def _normalize_bool(value: Any, *, name: str, default: bool) -> bool:
        if value is None:
            return default
        if not isinstance(value, bool):
            raise OptimizationValidationError(f"{name} must be boolean")
        return value

    def _normalize_problem(self, spec: Any) -> OptimizationProblem:
        """Normalize the request problem through the canonical subsystem builder."""
        if isinstance(spec, OptimizationProblem):
            return spec
        if not isinstance(spec, Mapping):
            raise OptimizationValidationError(
                "problem must be OptimizationProblem or a problem-definition mapping",
                context={"actual_type": type(spec).__name__},
            )
        return self.problem_definition.build(spec)

    @staticmethod
    def _normalize_mapping_sequence(value: Any, *, name: str) -> Optional[Tuple[Dict[str, float], ...]]:
        if value is None:
            return None
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            raise OptimizationValidationError(f"{name} must be a sequence of mappings")
        normalized = []
        for item in value:
            if not isinstance(item, Mapping):
                raise OptimizationValidationError(f"{name} entries must be mappings")
            normalized.append(dict(item))
        return tuple(normalized)

    # ------------------------------------------------------------------
    # Result shaping and memory ownership
    # ------------------------------------------------------------------
    @staticmethod
    def _attempt_to_dict(attempt: SolverAttempt) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "solver": attempt.solver_name,
            "available": attempt.available,
            "compatible": attempt.compatible,
        }
        if attempt.preference_score is not None:
            payload["preference_score"] = attempt.preference_score
        if attempt.incompatibilities:
            payload["incompatibilities"] = list(attempt.incompatibilities)
        if attempt.status is not None:
            payload["status"] = attempt.status.value
        if attempt.termination_reason is not None:
            payload["termination_reason"] = attempt.termination_reason.value
        if attempt.runtime_seconds is not None:
            payload["runtime_seconds"] = attempt.runtime_seconds
        if attempt.error:
            payload["error"] = attempt.error
        return payload

    @staticmethod
    def _solution_payload(solution: Solution) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "success": solution.success,
            "solver": solution.solver_name,
            "solver_status": solution.status.value,
            "termination_reason": solution.termination_reason.value,
            "feasibility": solution.feasibility.value,
            "optimality": solution.optimality.value,
            "objective_values": dict(solution.objective_values),
            "warnings": list(solution.warnings),
        }
        optional = {
            "decision_variables": None if solution.primal_solution is None else dict(solution.primal_solution),
            "max_constraint_violation": solution.max_constraint_violation,
            "primal_bound": solution.primal_bound,
            "dual_bound": solution.dual_bound,
            "absolute_gap": solution.absolute_gap,
            "relative_gap": solution.relative_gap,
            "iterations": solution.iteration_count,
            "nodes": solution.node_count,
            "evaluations": solution.function_evaluation_count,
            "runtime_seconds": solution.runtime_seconds,
            "solver_version": solution.solver_version,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        if solution.diagnostics:
            payload["diagnostics"] = dict(solution.diagnostics)
        if solution.metadata:
            payload["solver_metadata"] = dict(solution.metadata)
        return payload

    @staticmethod
    def _pareto_payload(front: ParetoFront) -> Dict[str, Any]:
        return {
            "objective_names": list(front.objective_names),
            "approximate": front.approximate,
            "algorithm": front.algorithm,
            "points": [
                {
                    "decision_variables": dict(point.decision),
                    "objective_values": dict(point.objective_values),
                    "feasible": point.feasible,
                    "violation": point.violation,
                    "metadata": dict(point.metadata),
                }
                for point in front.points
            ],
            "metadata": dict(front.metadata),
        }

    @staticmethod
    def _status_label(result: OptimizationResult) -> str:
        solution = result.solution
        if solution is None:
            return "success" if result.pareto_front and len(result.pareto_front) > 0 else "no_solution"
        status_map = {
            SolverStatus.OPTIMAL: "success",
            SolverStatus.FEASIBLE: "success",
            SolverStatus.INFEASIBLE: "infeasible",
            SolverStatus.UNBOUNDED: "unbounded",
            SolverStatus.INFEASIBLE_OR_UNBOUNDED: "infeasible_or_unbounded",
            SolverStatus.TIMEOUT: "timeout",
            SolverStatus.LIMIT_REACHED: "limit_reached",
            SolverStatus.NUMERICAL_FAILURE: "numerical_failure",
            SolverStatus.SOLVER_UNAVAILABLE: "solver_unavailable",
            SolverStatus.UNSUPPORTED: "unsupported",
            SolverStatus.INTERRUPTED: "interrupted",
            SolverStatus.ERROR: "failed",
        }
        return status_map.get(solution.status, "unknown")

    @staticmethod
    def _result_success(result: OptimizationResult) -> bool:
        if result.solution is not None:
            return bool(result.solution.success)
        return bool(result.pareto_front and len(result.pareto_front) > 0)

    def _evidence_since(self, previous_run_id: Optional[str]) -> Tuple[OptimizationRunRecord, ...]:
        recent = list(self.local_memory.history(limit=self.local_reference_limit))
        if previous_run_id is None:
            return tuple(recent)
        indices = [idx for idx, record in enumerate(recent) if record.run_id == previous_run_id]
        if not indices:
            return tuple(recent)
        return tuple(recent[indices[-1] + 1 :])

    @staticmethod
    def _collect_local_memory_failures(result: OptimizationResult) -> Tuple[str, ...]:
        failures: list[str] = []
        if result.solution is not None:
            value = result.solution.metadata.get("optimization_memory_persistence_error")
            if value:
                failures.append(str(value))
            for warning in result.solution.warnings:
                text = str(warning)
                if text.startswith("optimization_memory_persistence_failed:"):
                    normalized = text.split(":", 1)[1].strip()
                    if normalized not in failures:
                        failures.append(normalized)
        multi = result.metadata.get("optimization_memory_persistence_errors")
        if isinstance(multi, Sequence) and not isinstance(multi, (str, bytes, bytearray)):
            for item in multi:
                normalized = str(item).strip()
                if normalized and normalized not in failures:
                    failures.append(normalized)
        return tuple(failures)

    def _build_agent_result(
        self,
        *,
        run_id: str,
        request_id: str,
        problem: OptimizationProblem,
        fingerprint: str,
        result: OptimizationResult,
        evidence: Tuple[OptimizationRunRecord, ...],
        duration_ms: float,
        metadata: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        solution_payload = self._solution_payload(result.solution) if result.solution is not None else None
        pareto_payload = self._pareto_payload(result.pareto_front) if result.pareto_front is not None else None
        selection_payload = None
        if result.selection is not None:
            selection_payload = {
                "selected_solver": result.selection.selected_solver,
                "requested_solver": result.selection.requested_solver,
                "attempts": [self._attempt_to_dict(item) for item in result.selection.attempts],
            }

        local_ids = tuple(record.run_id for record in evidence)
        evidence_classes = tuple(dict.fromkeys(record.problem_class for record in evidence))
        if result.solution is not None:
            resolved_problem_class = result.solution.problem_class.value
        elif problem.problem_class is not ProblemClass.UNKNOWN:
            resolved_problem_class = problem.problem_class.value
        else:
            resolved_problem_class = evidence_classes[-1] if evidence_classes else ProblemClass.UNKNOWN.value
        solver_backends_list = list(dict.fromkeys(record.solver_name for record in evidence))
        if result.solution is not None and result.solution.solver_name not in solver_backends_list:
            solver_backends_list.append(result.solution.solver_name)
        if result.pareto_front is not None:
            for point in result.pareto_front.points:
                backend = point.metadata.get("solver")
                if isinstance(backend, str) and backend and backend not in solver_backends_list:
                    solver_backends_list.append(backend)
        solver_backends = tuple(solver_backends_list)
        selected_solver = None
        if result.solution is not None:
            selected_solver = result.solution.solver_name
        elif result.selection is not None:
            selected_solver = result.selection.selected_solver
        elif len(solver_backends) == 1:
            selected_solver = solver_backends[0]
        payload: Dict[str, Any] = {
            "status": self._status_label(result),
            "success": self._result_success(result),
            "run_id": run_id,
            "request_id": request_id,
            "problem_fingerprint": fingerprint,
            "problem_name": problem.name,
            "problem_class": resolved_problem_class,
            "workflow": str(result.metadata.get("method", "single_objective" if not problem.is_multi_objective else "multiobjective")),
            "duration_ms": duration_ms,
            "solution": solution_payload,
            "pareto_front": pareto_payload,
            "solver_selection": selection_payload,
            "warnings": list(result.solution.warnings) if result.solution is not None else [],
            "provenance": {
                "agent": self.name,
                "agent_version": __version__,
                "local_evidence_ids": list(local_ids),
                "local_evidence_count": len(local_ids),
                "subsystem_metadata": dict(result.metadata),
            },
            "local_memory_reference": {
                "problem_fingerprint": fingerprint,
                "run_ids": list(local_ids),
            },
            "selected_solver": selected_solver,
            "solver_backends": list(solver_backends),
            "shared_memory_reference": None,
            "metadata": dict(metadata),
            "context": dict(context),
        }
        return payload

    def _shared_summary(self, result: Mapping[str, Any]) -> Dict[str, Any]:
        solution = result.get("solution") if isinstance(result.get("solution"), Mapping) else None
        front = result.get("pareto_front") if isinstance(result.get("pareto_front"), Mapping) else None
        summary: Dict[str, Any] = {
            "run_id": result.get("run_id"),
            "request_id": result.get("request_id"),
            "problem_fingerprint": result.get("problem_fingerprint"),
            "problem_name": result.get("problem_name"),
            "problem_class": result.get("problem_class"),
            "workflow": result.get("workflow"),
            "status": result.get("status"),
            "success": result.get("success"),
            "duration_ms": result.get("duration_ms"),
            "local_memory_reference": result.get("local_memory_reference"),
            "selected_solver": result.get("selected_solver"),
            "solver_backends": list(result.get("solver_backends", [])),
        }
        if solution is not None:
            for key in (
                "solver",
                "solver_status",
                "termination_reason",
                "feasibility",
                "optimality",
                "objective_values",
                "max_constraint_violation",
                "primal_bound",
                "dual_bound",
                "relative_gap",
                "runtime_seconds",
                "solver_version",
            ):
                if key in solution:
                    summary[key] = solution[key]
            if self.include_decision_variables_in_shared_memory and "decision_variables" in solution:
                summary["decision_variables"] = solution["decision_variables"]
            warnings = solution.get("warnings")
            if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes)):
                summary["warnings"] = list(warnings[: self.max_shared_warnings])
        if front is not None:
            points = front.get("points")
            compact_points = []
            if isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
                for point in points[: self.max_shared_pareto_points]:
                    if not isinstance(point, Mapping):
                        continue
                    compact_points.append(
                        {
                            "objective_values": dict(point.get("objective_values", {})),
                            **(
                                {"decision_variables": dict(point.get("decision_variables", {}))}
                                if self.include_decision_variables_in_shared_memory
                                else {}
                            ),
                            "feasible": point.get("feasible"),
                            "violation": point.get("violation"),
                        }
                    )
            summary["pareto"] = {
                "objective_names": list(front.get("objective_names", [])),
                "front_size": len(points) if isinstance(points, Sequence) else 0,
                "published_points": compact_points,
                "approximate": front.get("approximate"),
                "algorithm": front.get("algorithm"),
            }
        return summary

    def _record_shared_publication_failure(
        self,
        *,
        operation: str,
        run_id: str,
        error: BaseException,
        result: Dict[str, Any],
    ) -> None:
        self._mark_runtime_degraded("persistence", operation, error)
        with self._metrics_lock:
            self._shared_publication_failures += 1
        warning = f"shared_memory_publication_failed[{operation}]: {type(error).__name__}: {error}"
        warnings = result.setdefault("warnings", [])
        if warning not in warnings:
            warnings.append(warning)
        logger.warning(
            "Optimization result remains valid but SharedMemory operation failed | run=%s | operation=%s | error=%s",
            run_id,
            operation,
            error,
        )

    def _publish_shared_summary(self, result: Dict[str, Any]) -> Optional[str]:
        if not self.shared_memory_enabled:
            return None
        run_id = str(result["run_id"])
        key = f"{self.shared_result_key_prefix}:{run_id}"
        summary = self._shared_summary(result)
        ttl = self.shared_memory_ttl_seconds
        latest_reference = {
            "run_id": run_id,
            "request_id": summary.get("request_id"),
            "problem_fingerprint": summary.get("problem_fingerprint"),
            "status": summary.get("status"),
            "result_key": key,
            "timestamp": time.time(),
        }

        # The per-run summary is the authoritative SharedMemory publication.
        # Latest-pointer and pub/sub event failures are observable but do not
        # erase a successfully published cross-agent result reference.
        try:
            self.shared_memory.set(key, summary, ttl=ttl)
        except Exception as exc:
            self._record_shared_publication_failure(
                operation="optimization.shared_memory.result",
                run_id=run_id,
                error=exc,
                result=result,
            )
            return None
        self._mark_runtime_recovered("persistence", "optimization.shared_memory.result")

        try:
            self.shared_memory.set(self.shared_latest_key, latest_reference, ttl=ttl)
        except Exception as exc:
            self._record_shared_publication_failure(
                operation="optimization.shared_memory.latest",
                run_id=run_id,
                error=exc,
                result=result,
            )
        else:
            self._mark_runtime_recovered("persistence", "optimization.shared_memory.latest")

        publisher = getattr(self.shared_memory, "publish", None)
        if callable(publisher):
            try:
                publisher(self.shared_event_topic, {"event": "optimization.completed", **summary})
            except Exception as exc:
                self._record_shared_publication_failure(
                    operation="optimization.shared_memory.event",
                    run_id=run_id,
                    error=exc,
                    result=result,
                )
            else:
                self._mark_runtime_recovered("persistence", "optimization.shared_memory.event")

        return key

    # ------------------------------------------------------------------
    # Execution orchestration
    # ------------------------------------------------------------------
    def _solve_request(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            raise OptimizationConfigurationError("OptimizationAgent is disabled")

        request_id = self._normalize_request_id(request.get("request_id"))
        run_id = f"opt-run-{uuid.uuid4().hex[:20]}"
        metadata = self._normalize_metadata(request.get("metadata"))
        context = self._normalize_context(request.get("context"))
        publish = self._normalize_bool(
            request.get("publish"),
            name="publish",
            default=self.publish_shared_results,
        )
        problem = self._normalize_problem(request.get("problem"))
        problem = self._apply_explicit_warm_start(problem, request.get("warm_start"))
        solver_name = self._normalize_solver_name(request.get("solver"))
        options = self._merge_execution_constraints(request, self._normalize_options(request.get("options")))
        allow_fallback_raw = request.get("allow_fallback")
        allow_fallback = None if allow_fallback_raw is None else self._normalize_bool(
            allow_fallback_raw,
            name="allow_fallback",
            default=False,
        )

        method_raw = request.get("method", request.get("workflow"))
        method = "weighted" if problem.is_multi_objective and method_raw is None else method_raw
        if method is not None and (not isinstance(method, str) or not method.strip()):
            raise OptimizationValidationError("method/workflow must be a non-empty string when provided")
        method = None if method is None else method.strip().lower()

        weight_vectors = self._normalize_mapping_sequence(request.get("weight_vectors"), name="weight_vectors")
        epsilon_grid = self._normalize_mapping_sequence(request.get("epsilon_grid"), name="epsilon_grid")
        primary = request.get("primary")
        if primary is not None and (not isinstance(primary, str) or not primary.strip()):
            raise OptimizationValidationError("primary must be a non-empty objective name when provided")
        primary = None if primary is None else primary.strip()

        if not problem.is_multi_objective and any(value is not None for value in (weight_vectors, epsilon_grid, primary)):
            raise OptimizationValidationError("multi-objective controls require a multi-objective problem")

        fingerprint = self.local_memory.problem_fingerprint(problem)
        started = time.perf_counter()

        with self._optimization_lock:
            before = self.local_memory.history(limit=1)
            before_latest_id = before[-1].run_id if before else None
            try:
                if problem.is_multi_objective:
                    result = self.optimization_solver.solve_multiobjective(
                        problem,
                        method=method or "weighted",
                        weight_vectors=weight_vectors,
                        epsilon_grid=epsilon_grid,
                        primary=primary,
                        solver=solver_name,
                        options=options,
                        allow_fallback=allow_fallback,
                    )
                else:
                    result = self.optimization_solver.solve_single_detailed(
                        problem,
                        solver=solver_name,
                        options=options,
                        allow_fallback=allow_fallback,
                    )
            except OptimizationError as exc:
                duration_ms = (time.perf_counter() - started) * 1000.0
                self._record_failure_metrics(problem, duration_ms, error=exc)
                raise
            except (ArithmeticError, TypeError, ValueError, RuntimeError) as exc:
                duration_ms = (time.perf_counter() - started) * 1000.0
                self._record_failure_metrics(problem, duration_ms, error=exc)
                raise SolverExecutionError(
                    "Unexpected Optimization subsystem boundary failure",
                    context={"problem": problem.name, "problem_fingerprint": fingerprint},
                    cause=exc,
                ) from exc
            evidence = self._evidence_since(before_latest_id)

        duration_ms = (time.perf_counter() - started) * 1000.0
        payload = self._build_agent_result(
            run_id=run_id,
            request_id=request_id,
            problem=problem,
            fingerprint=fingerprint,
            result=result,
            evidence=evidence,
            duration_ms=duration_ms,
            metadata=metadata,
            context=context,
        )

        memory_errors = self._collect_local_memory_failures(result)
        if memory_errors:
            with self._metrics_lock:
                self._local_memory_persistence_failures += len(memory_errors)
            warnings = payload.setdefault("warnings", [])
            for memory_error in memory_errors:
                warning = f"local_memory_persistence_failed: {memory_error}"
                if warning not in warnings:
                    warnings.append(warning)
            self._mark_runtime_degraded(
                "persistence",
                "optimization.local_memory.record",
                "; ".join(memory_errors),
            )
        else:
            self._mark_runtime_recovered("persistence", "optimization.local_memory.record")

        if publish:
            shared_ref = self._publish_shared_summary(payload)
            if shared_ref is not None:
                payload["shared_memory_reference"] = shared_ref

        self._record_success_metrics(problem, result, duration_ms, options, evidence)
        self._last_run_id = run_id
        self._last_problem_fingerprint = fingerprint
        self._last_local_evidence_ids = tuple(record.run_id for record in evidence)

        logger.info(
            "OptimizationAgent completed | run=%s | problem=%s | class=%s | status=%s | solver=%s | %.2f ms",
            run_id,
            problem.name,
            problem.problem_class.value,
            payload["status"],
            payload.get("solution", {}).get("solver") if isinstance(payload.get("solution"), Mapping) else None,
            duration_ms,
        )
        return payload

    def optimize(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        """Execute one validated optimization request outside BaseAgent's retry envelope."""
        return self._solve_request(self._validate_task_mapping(request))

    def perform_task(self, task_data: Any) -> Dict[str, Any]:
        request = self._validate_task_mapping(task_data)
        batch = request.get("tasks")
        if batch is None:
            return self._solve_request(request)
        conflicting = sorted(set(request) - {"tasks"})
        if conflicting:
            raise OptimizationValidationError(
                "Batch requests must contain only the tasks field; per-run controls belong inside each task",
                context={"conflicting_fields": conflicting},
            )
        if isinstance(batch, (str, bytes, bytearray)) or not isinstance(batch, Sequence):
            raise OptimizationValidationError("tasks must be a sequence of OptimizationAgent requests")
        if len(batch) == 0 or len(batch) > self.max_batch_size:
            raise OptimizationValidationError(
                "OptimizationAgent batch size is outside configured limits",
                context={"size": len(batch), "max_batch_size": self.max_batch_size},
            )
        results = [self._solve_request(self._validate_task_mapping(item)) for item in batch]
        return {
            "status": "success" if all(bool(item.get("success")) for item in results) else "partial_success",
            "batch_size": len(results),
            "results": results,
        }

    def alternative_execute(self, task_data: Any, original_error: Optional[BaseException] = None) -> Dict[str, Any]:
        """Decline generic non-mathematical BaseAgent fallbacks.

        Solver/backend fallback is already handled by ``OptimizationSolver``.
        Returning a structured failure prevents the generic BaseAgent text/rule
        fallback from being mistaken for a valid optimization result.
        """
        return {
            "status": "failed",
            "reason": "OptimizationAgent does not synthesize fallback mathematical solutions.",
            "error_type": type(original_error).__name__ if original_error is not None else None,
            "error": str(original_error) if original_error is not None else None,
        }

    def shutdown(self) -> None:
        """Stop this runtime without clearing BaseAgent SharedMemory or local evidence."""
        lifecycle = self._runtime_status.lifecycle
        if lifecycle in {RuntimeLifecycle.STOPPING, RuntimeLifecycle.STOPPED}:
            return
        self.transition_runtime_lifecycle(RuntimeLifecycle.STOPPING)
        self.operational_state = "stopping"
        self.transition_runtime_lifecycle(RuntimeLifecycle.STOPPED)
        self.operational_state = "stopped"
        logger.info("OptimizationAgent stopped | agent_id=%s", self.agent_id)

    # ------------------------------------------------------------------
    # Metrics and observability
    # ------------------------------------------------------------------
    def _record_failure_metrics(
        self,
        problem: OptimizationProblem,
        duration_ms: float,
        *,
        error: BaseException | None = None,
    ) -> None:
        if not self.metrics_enabled:
            return
        with self._metrics_lock:
            self._requests_total += 1
            self._requests_failed += 1
            self._runtime_total_ms += duration_ms
            if error is not None:
                self._failure_types[type(error).__name__] += 1
            if problem.is_multi_objective:
                self._multiobjective_runs += 1
            else:
                self._single_objective_runs += 1

    def _record_success_metrics(
        self,
        problem: OptimizationProblem,
        result: OptimizationResult,
        duration_ms: float,
        options: Mapping[str, Any],
        evidence: Sequence[OptimizationRunRecord],
    ) -> None:
        if not self.metrics_enabled:
            return
        with self._metrics_lock:
            self._requests_total += 1
            self._runtime_total_ms += duration_ms
            if self._result_success(result):
                self._requests_success += 1
            else:
                self._requests_failed += 1
            if problem.is_multi_objective:
                self._multiobjective_runs += 1
            else:
                self._single_objective_runs += 1
            warm_started = (
                "x0" in options
                or any(variable.initial is not None for variable in problem.variables)
                or any(
                    record.options.get("warm_start_source") == "optimization_memory"
                    for record in evidence
                )
            )
            if warm_started:
                self._warm_start_runs += 1
            fallback_count = int(result.metadata.get("fallback_count", 0) or 0)
            evidence_fallback = any(int(record.metadata.get("orchestrator_attempt", 1) or 1) > 1 for record in evidence)
            if fallback_count > 0 or evidence_fallback:
                self._fallback_runs += 1
            if result.solution is not None:
                solution = result.solution
                self._solver_usage[solution.solver_name] += 1
                self._termination_counts[solution.termination_reason.value] += 1
                if solution.status is SolverStatus.INFEASIBLE:
                    self._infeasible_runs += 1
                elif solution.status is SolverStatus.UNBOUNDED:
                    self._unbounded_runs += 1
                elif solution.status is SolverStatus.TIMEOUT:
                    self._timeout_runs += 1
                elif solution.status is SolverStatus.LIMIT_REACHED:
                    self._limit_runs += 1
                elif solution.status is SolverStatus.NUMERICAL_FAILURE:
                    self._numerical_failure_runs += 1

    def metrics_snapshot(self) -> Dict[str, Any]:
        with self._metrics_lock:
            total = self._requests_total
            return {
                "requests_total": total,
                "requests_success": self._requests_success,
                "requests_failed": self._requests_failed,
                "success_rate": self._requests_success / total if total else 0.0,
                "single_objective_runs": self._single_objective_runs,
                "multiobjective_runs": self._multiobjective_runs,
                "warm_start_runs": self._warm_start_runs,
                "fallback_runs": self._fallback_runs,
                "infeasible_runs": self._infeasible_runs,
                "unbounded_runs": self._unbounded_runs,
                "timeout_runs": self._timeout_runs,
                "limit_runs": self._limit_runs,
                "numerical_failure_runs": self._numerical_failure_runs,
                "shared_memory_publication_failures": self._shared_publication_failures,
                "local_memory_failures": self._local_memory_persistence_failures,
                "runtime_total_ms": self._runtime_total_ms,
                "average_runtime_ms": self._runtime_total_ms / total if total else 0.0,
                "solver_usage": dict(self._solver_usage),
                "termination_counts": dict(self._termination_counts),
                "failure_types": dict(self._failure_types),
                "local_memory_records": len(self.local_memory.history()),
            }

    def extract_performance_metrics(self, result: Any) -> Dict[str, float]:
        metrics = super().extract_performance_metrics(result)
        if isinstance(result, Mapping):
            duration = result.get("duration_ms")
            if isinstance(duration, (int, float)):
                metrics["latency_ms"] = float(duration)
            success = result.get("success")
            if isinstance(success, bool):
                metrics["optimization_success"] = 1.0 if success else 0.0
            solution = result.get("solution")
            if isinstance(solution, Mapping):
                gap = solution.get("relative_gap")
                if isinstance(gap, (int, float)):
                    metrics["relative_gap"] = float(gap)
        return metrics

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "agent": self.name,
            "enabled": self.enabled,
            "shared_memory": {
                "enabled": self.shared_memory_enabled,
                "publish_shared_results": self.publish_shared_results,
                "latest_key": self.shared_latest_key,
            },
            "local_memory": {
                "type": type(self.local_memory).__name__,
                "records": len(self.local_memory.history()),
                "latest_evidence_ids": list(self._last_local_evidence_ids),
            },
            "last_run_id": self._last_run_id,
            "last_problem_fingerprint": self._last_problem_fingerprint,
            "metrics": self.metrics_snapshot(),
            "runtime": self.runtime_status(),
            "checkpointing": {
                "supported": self.supports_checkpointing,
                "enabled": self.checkpointing_enabled,
                "policy": "agent metrics/references only; detailed solver evidence remains OptimizationMemory-owned",
            },
        }

    # ------------------------------------------------------------------
    # Checkpoint integration
    # ------------------------------------------------------------------
    def checkpoint_step(self) -> Optional[int]:
        return self._requests_total

    def checkpoint_metrics(self) -> Mapping[str, Any]:
        snapshot = self.metrics_snapshot()
        return {
            "requests_total": snapshot["requests_total"],
            "requests_success": snapshot["requests_success"],
            "requests_failed": snapshot["requests_failed"],
            "average_runtime_ms": snapshot["average_runtime_ms"],
        }

    def _export_checkpoint_state(self) -> Mapping[str, Any]:
        # Detailed OptimizationRunRecord data is deliberately not duplicated
        # into BaseAgent checkpoints. OptimizationMemory remains authoritative.
        return {
            "schema": self.CHECKPOINT_SCHEMA,
            "version": __version__,
            "metrics": self.metrics_snapshot(),
            "last_run_id": self._last_run_id,
            "last_problem_fingerprint": self._last_problem_fingerprint,
            "last_local_evidence_ids": list(self._last_local_evidence_ids),
        }

    def _import_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise OptimizationValidationError("OptimizationAgent checkpoint state must be a mapping")
        if state.get("schema") != self.CHECKPOINT_SCHEMA:
            raise OptimizationValidationError(
                "Incompatible OptimizationAgent checkpoint schema",
                context={"actual": state.get("schema"), "expected": self.CHECKPOINT_SCHEMA},
            )
        if str(state.get("version", "")) != __version__:
            raise OptimizationValidationError(
                "OptimizationAgent checkpoint version is incompatible",
                context={"actual": state.get("version"), "expected": __version__},
            )
        metrics = state.get("metrics", {})
        if isinstance(metrics, Mapping) and self.metrics_enabled:
            with self._metrics_lock:
                self._requests_total = max(0, int(metrics.get("requests_total", 0)))
                self._requests_success = max(0, int(metrics.get("requests_success", 0)))
                self._requests_failed = max(0, int(metrics.get("requests_failed", 0)))
                self._single_objective_runs = max(0, int(metrics.get("single_objective_runs", 0)))
                self._multiobjective_runs = max(0, int(metrics.get("multiobjective_runs", 0)))
                self._warm_start_runs = max(0, int(metrics.get("warm_start_runs", 0)))
                self._fallback_runs = max(0, int(metrics.get("fallback_runs", 0)))
                self._infeasible_runs = max(0, int(metrics.get("infeasible_runs", 0)))
                self._unbounded_runs = max(0, int(metrics.get("unbounded_runs", 0)))
                self._timeout_runs = max(0, int(metrics.get("timeout_runs", 0)))
                self._limit_runs = max(0, int(metrics.get("limit_runs", 0)))
                self._numerical_failure_runs = max(0, int(metrics.get("numerical_failure_runs", 0)))
                self._shared_publication_failures = max(0, int(metrics.get("shared_memory_publication_failures", 0)))
                self._local_memory_persistence_failures = max(0, int(metrics.get("local_memory_failures", 0)))
                self._runtime_total_ms = max(0.0, float(metrics.get("runtime_total_ms", 0.0)))
                solver_usage = metrics.get("solver_usage", {})
                self._solver_usage = Counter(
                    {str(k): max(0, int(v)) for k, v in solver_usage.items()}
                    if isinstance(solver_usage, Mapping)
                    else {}
                )
                termination_counts = metrics.get("termination_counts", {})
                self._termination_counts = Counter(
                    {str(k): max(0, int(v)) for k, v in termination_counts.items()}
                    if isinstance(termination_counts, Mapping)
                    else {}
                )
                failure_types = metrics.get("failure_types", {})
                self._failure_types = Counter(
                    {str(k): max(0, int(v)) for k, v in failure_types.items()}
                    if isinstance(failure_types, Mapping)
                    else {}
                )
        self._last_run_id = str(state["last_run_id"]) if state.get("last_run_id") else None
        self._last_problem_fingerprint = (
            str(state["last_problem_fingerprint"]) if state.get("last_problem_fingerprint") else None
        )
        raw_ids = state.get("last_local_evidence_ids", ())
        if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes, bytearray)):
            self._last_local_evidence_ids = tuple(str(item) for item in raw_ids[: self.local_reference_limit])
        else:
            self._last_local_evidence_ids = ()


__all__ = ["OptimizationAgent"]


if __name__ == "__main__":
    configure_logging()
    printer.status("TEST", "OptimizationAgent production façade validation", "info")
    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()
    agent = OptimizationAgent(
        shared_memory=shared_memory,
        agent_factory=agent_factory,
        config={"publish_shared_results": False},
    )
    response = agent.perform_task(
        {
            "problem": {
                "name": "agent_smoke_lp",
                "problem_class": "lp",
                "variables": [{"name": "x", "type": "real", "lower": 0, "upper": 3}],
                "objectives": [{"name": "profit", "sense": "maximize", "linear": {"x": 1.0}}],
            }
        }
    )
    assert response["success"] is True, response
    assert response["solution"]["solver_status"] == SolverStatus.OPTIMAL.value, response
    assert response["solution"]["decision_variables"]["x"] == 3.0, response
    printer.status("SUCCESS", "OptimizationAgent LP orchestration smoke passed", "success")
