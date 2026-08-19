"""Single bounded autonomous control loop for SLAI.

The loop is the sole outer autonomy owner. Domain agents may retain bounded
internal algorithms, but they are invoked as capabilities in this fixed order:

``reason -> plan -> authorize -> execute -> evaluate``

Construction and execution are explicit; importing this module creates no
agents, threads, files, network calls, or background work.
"""

from __future__ import annotations

import time
import uuid

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from threading import Event, RLock
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .runtime_contracts import RuntimeLifecycle, RuntimeStatus
from logs.logger import get_logger  # pyright: ignore[reportMissingImports]
from ..utils.configuration import bind_config


logger = get_logger("Autonomous Control Loop")
_CONFIG = bind_config(Path(__file__).parent / "base/configs/agents_config.yaml")
get_config_section = _CONFIG.section

StageCallable = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class AutonomousControlLoopError(RuntimeError):
    """Base error for control-loop configuration, contracts, and execution."""


class ControlLoopConfigurationError(AutonomousControlLoopError):
    """Raised when the autonomous loop configuration is unsafe or malformed."""


class ControlLoopContractError(AutonomousControlLoopError):
    """Raised when a stage violates the shared stage-result contract."""


class ControlLoopBusyError(AutonomousControlLoopError):
    """Raised when a caller attempts concurrent runs on one loop instance."""


def _strict_bool(value: Any, *, field_name: str) -> bool:
    """Parse a control-loop configuration boolean without truthiness coercion."""

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"true", "1", "yes", "on"}:
            return True

        if normalized in {"false", "0", "no", "off"}:
            return False

    if isinstance(value, int) and not isinstance(value, bool):
        if value in {0, 1}:
            return bool(value)

    raise ControlLoopConfigurationError(
        f"{field_name} must be a boolean"
    )


class ControlLoopState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    REVIEW_REQUIRED = "review_required"
    BLOCKED = "blocked"
    EXHAUSTED = "exhausted"
    FAILED = "failed"
    STOPPED = "stopped"
    DISABLED = "disabled"

    @property
    def terminal(self) -> bool:
        return self not in {ControlLoopState.IDLE, ControlLoopState.RUNNING}


@dataclass(frozen=True, slots=True)
class AutonomousLoopConfig:
    """Safety and resource bounds for the single outer control loop."""

    enabled: bool = True
    max_cycles: int = 3
    max_cycle_seconds: float = 45.0
    max_run_seconds: float = 120.0
    min_action_confidence: float = 0.7
    require_safety_approval: bool = True
    require_explicit_execution_task: bool = True
    stop_on_stage_error: bool = True
    publish_events: bool = True
    event_channel: str = "autonomy.events"
    latest_state_key: str = "autonomy:control_loop:latest"

    @classmethod
    def from_mapping(
        cls,
        payload: Optional[Mapping[str, Any]],
    ) -> "AutonomousLoopConfig":
        source = dict(payload or {})
    
        try:
            config = cls(
                enabled=_strict_bool(
                    source.get("enabled", True),
                    field_name="autonomous_control_loop.enabled",
                ),
                max_cycles=int(
                    source.get("max_cycles", 3)
                ),
                max_cycle_seconds=float(
                    source.get("max_cycle_seconds", 45.0)
                ),
                max_run_seconds=float(
                    source.get("max_run_seconds", 120.0)
                ),
                min_action_confidence=float(
                    source.get("min_action_confidence", 0.7)
                ),
                require_safety_approval=_strict_bool(
                    source.get("require_safety_approval", True),
                    field_name=(
                        "autonomous_control_loop."
                        "require_safety_approval"
                    ),
                ),
                require_explicit_execution_task=_strict_bool(
                    source.get("require_explicit_execution_task", True),
                    field_name=(
                        "autonomous_control_loop."
                        "require_explicit_execution_task"
                    ),
                ),
                stop_on_stage_error=_strict_bool(source.get("stop_on_stage_error", True),
                                                field_name=(
                                                    "autonomous_control_loop."
                                                    "stop_on_stage_error"
                                                    ),
                ),
                publish_events=_strict_bool(source.get("publish_events", True),
                                            field_name=(
                                                "autonomous_control_loop."
                                                "publish_events"
                                                ),
                ),
                event_channel=str(source.get("event_channel", "autonomy.events")).strip(),
                latest_state_key=str(source.get(
                    "latest_state_key",
                    "autonomy:control_loop:latest",
                    )).strip())
        except ControlLoopConfigurationError:
            raise
        except (TypeError, ValueError) as exc:
            raise ControlLoopConfigurationError(
                "Invalid autonomous_control_loop "
                f"configuration: {exc}"
            ) from exc
    
        config.validate()
        return config

    def validate(self) -> None:
        if self.max_cycles < 1:
            raise ControlLoopConfigurationError("autonomous_control_loop.max_cycles must be at least 1")
        if self.max_cycle_seconds <= 0:
            raise ControlLoopConfigurationError("autonomous_control_loop.max_cycle_seconds must be positive")
        if self.max_run_seconds < self.max_cycle_seconds:
            raise ControlLoopConfigurationError(
                "autonomous_control_loop.max_run_seconds must be greater than or equal to max_cycle_seconds"
            )
        if not 0.0 <= self.min_action_confidence <= 1.0:
            raise ControlLoopConfigurationError(
                "autonomous_control_loop.min_action_confidence must be between 0 and 1"
            )
        if not self.event_channel:
            raise ControlLoopConfigurationError("autonomous_control_loop.event_channel must be non-empty")
        if not self.latest_state_key:
            raise ControlLoopConfigurationError("autonomous_control_loop.latest_state_key must be non-empty")


@dataclass(frozen=True, slots=True)
class StageResult:
    stage: str
    status: str
    duration_ms: float
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ControlCycleResult:
    cycle: int
    status: str
    duration_ms: float
    stages: tuple[StageResult, ...]
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "reason": self.reason,
            "stages": [stage.to_dict() for stage in self.stages],
        }


@dataclass(frozen=True, slots=True)
class AutonomousRunResult:
    run_id: str
    goal_id: str
    state: ControlLoopState
    started_at: float
    finished_at: float
    cycles: tuple[ControlCycleResult, ...]
    reason: str

    @property
    def succeeded(self) -> bool:
        return self.state == ControlLoopState.SUCCEEDED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal_id": self.goal_id,
            "state": self.state.value,
            "succeeded": self.succeeded,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": max(0.0, (self.finished_at - self.started_at) * 1000.0),
            "reason": self.reason,
            "cycles": [cycle.to_dict() for cycle in self.cycles],
        }


def _safe_value(value: Any) -> Any:
    """Return a bounded, serializable result shape without inventing semantics."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _safe_value(asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _safe_value(to_dict())
        except Exception:
            pass
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _normalized_status(payload: Mapping[str, Any]) -> str:
    raw = payload.get("status")
    if raw is None:
        if payload.get("completed") is True or payload.get("success") is True or payload.get("passed") is True:
            return "success"
        return "unknown"
    return str(raw).strip().lower()


def _is_failed(payload: Mapping[str, Any]) -> bool:
    return _normalized_status(payload) in {"block", "blocked", "critical", "error", "fail", "failed", "failure"} or any(
        payload.get(key) is False for key in ("success", "passed") if key in payload
    )


def _is_success(payload: Mapping[str, Any]) -> bool:
    status = _normalized_status(payload)
    return status in {"allow", "complete", "completed", "normal", "ok", "pass", "passed", "success", "succeeded"} or any(
        payload.get(key) is True for key in ("completed", "success", "passed")
    )


class FactoryAutonomousStages:
    """Lazy adapters from the fixed loop stages to factory-managed agent facades."""

    _STAGE_AGENT = {
        "reason": "reasoning",
        "plan": "planning",
        "execute": "execution",
        "evaluate": "evaluation",
    }

    def __init__(self, factory: Any, shared_memory: Any, config: AutonomousLoopConfig) -> None:
        if factory is None or not callable(getattr(factory, "create", None)):
            raise ControlLoopConfigurationError("from_factory requires an AgentFactory-compatible create() method")
        self.factory = factory
        self.shared_memory = shared_memory
        self.config = config
        self._agents: Dict[str, Any] = {}
        self._last_failed_agent_name: Optional[str] = None

    def _agent(self, name: str) -> Any:
        if name not in self._agents:
            self._agents[name] = self.factory.create(name, shared_memory=self.shared_memory)
        return self._agents[name]

    @staticmethod
    def _goal(payload: Mapping[str, Any]) -> Dict[str, Any]:
        goal = payload.get("goal", {})
        if not isinstance(goal, Mapping):
            raise ControlLoopContractError("Control-loop goal must remain a mapping")
        return dict(goal)

    @staticmethod
    def _call(agent: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(agent, method_name, None)
        if not callable(method):
            raise ControlLoopContractError(
                f"{type(agent).__name__} does not expose required callable {method_name}()"
            )
        return method(*args, **kwargs)

    def _call_agent(self, agent_name: str, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke one named agent and retain its identity on failure."""
    
        try:
            agent = self._agent(agent_name)
            return self._call(agent, method_name, *args, **kwargs)
        except Exception:
            self._last_failed_agent_name = agent_name
            raise

    def reason(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        goal = self._goal(payload)
        objective = goal.get("objective") or goal.get("goal") or goal.get("name")
        if not objective:
            raise ControlLoopContractError("Autonomous goal requires objective, goal, or name")
        result = self._call_agent(
            "reasoning",
            "perform_task",
            {
                "task_type": "reason",
                "problem": str(objective),
                "context": {
                    **dict(payload.get("context", {})),
                    "observation": payload.get("observation", {}),
                    "previous_feedback": payload.get("previous_feedback", {}),
                },
            },
        )
        return {"status": "success", "reasoning": _safe_value(result)}

    @staticmethod
    def _planning_task(goal: Mapping[str, Any]) -> Any:
        supplied = goal.get("planning_task")
        if supplied is not None:
            return supplied

        # Planning imports stay lazy so the control-loop module itself remains
        # free of the historical planning package's initialization cost.
        from .planning.planning_types import Task, TaskType

        objective = str(goal.get("objective") or goal.get("goal") or goal.get("name") or "Autonomous goal")
        raw_steps = goal.get("plan_steps")
        if isinstance(raw_steps, Sequence) and not isinstance(raw_steps, (str, bytes)) and raw_steps:
            steps = []
            for index, raw_step in enumerate(raw_steps, start=1):
                step = dict(raw_step) if isinstance(raw_step, Mapping) else {"name": str(raw_step)}
                steps.append(
                    Task(
                        name=str(step.get("name") or f"{objective}:step:{index}"),
                        task_type=TaskType.PRIMITIVE,
                        description=str(step.get("description") or step.get("name") or objective),
                        context=step,
                        duration=float(step.get("duration", 1.0)),
                        preconditions=[lambda _state: True],
                    )
                )
            return Task(
                name=str(goal.get("name") or objective),
                task_type=TaskType.ABSTRACT,
                goal_state=dict(goal.get("goal_state", {})),
                context=dict(goal.get("context", {})),
                methods=[steps],
            )
        return Task(
            name=str(goal.get("name") or objective),
            task_type=TaskType.PRIMITIVE,
            goal_state=dict(goal.get("goal_state", {})),
            context=dict(goal.get("context", {})),
            description=objective,
            duration=float(goal.get("duration", 1.0)),
            preconditions=[lambda _state: True],
        )

    def plan(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        goal = self._goal(payload)
        planning_task = self._planning_task(goal)
        plan = self._call_agent("planning", "generate_plan", planning_task)
        if not plan:
            return {"status": "failed", "plan": [], "reason": "planning_agent_returned_no_plan"}
        serialized = [_safe_value(item) for item in plan]
        return {"status": "success", "plan": serialized, "planning_task": _safe_value(planning_task)}

    def authorize(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        goal = self._goal(payload)
        plan_output = payload.get("plan", {})

        reason_output = payload.get("reason", {})
        if not isinstance(reason_output, Mapping):
            raise ControlLoopContractError(
                "reason stage output must be a mapping"
            )
        
        confidence = reason_output.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
        ):
            raise ControlLoopContractError(
                "reason.confidence must be an explicit "
                "calibrated number"
            )
        
        if not 0.0 <= float(confidence) <= 1.0:
            raise ControlLoopContractError(
                "reason.confidence must be between 0 and 1"
            )
        
        if float(confidence) < self.config.min_action_confidence:
            return {
                "status": "review_required",
                "approved": False,
                "decision": "review_required",
                "reason": "action_confidence_below_threshold",
                "confidence": float(confidence),
                "required_confidence": (
                    self.config.min_action_confidence
                ),
            }
    
        safety_action = goal.get("safety_action")
        if safety_action is None:
            safety_action = {
                "name": "execute_autonomous_task",
                "goal_id": payload.get("goal_id"),
                "plan": (
                    plan_output.get("plan", [])
                    if isinstance(plan_output, Mapping)
                    else []
                ),
            }
    
        if not isinstance(safety_action, Mapping):
            raise ControlLoopContractError("goal.safety_action must be a mapping")
    
        safety_result = self._call_agent("safety", "validate_action", dict(safety_action),
            {
                "type": "autonomous_control_loop",
                "run_id": payload.get("run_id"),
                "cycle": payload.get("cycle"),
                **dict(payload.get("context", {})),
            },
        )
    
        if not isinstance(safety_result, Mapping):
            raise ControlLoopContractError(
                "SafetyAgent.validate_action() "
                "must return a mapping"
            )
    
        alignment_task = goal.get("alignment_task")
        if not isinstance(alignment_task, Mapping):
            raise ControlLoopContractError(
                "goal.alignment_task must be a mapping "
                "compatible with "
                "AlignmentAgent.verify_alignment()"
            )
    
        alignment_result = self._call_agent("alignment", "verify_alignment", dict(alignment_task))
    
        if not isinstance(alignment_result, Mapping):
            raise ControlLoopContractError(
                "AlignmentAgent.verify_alignment() "
                "must return a mapping"
            )
    
        raw_alignment_decision = alignment_result.get("decision")
        if not isinstance(raw_alignment_decision, Mapping):
            raise ControlLoopContractError(
                "AlignmentAgent.verify_alignment() must "
                "return a mapping-valued decision"
            )
    
        alignment_decision = dict(raw_alignment_decision)
        safety_approved = (safety_result.get("approved") is True)
        alignment_approved = (
            alignment_decision.get("approved") is True
            and alignment_decision.get("requires_review")
            is not True
        )
    
        approved = safety_approved and alignment_approved
        safety_decision = str(safety_result.get("decision", safety_result.get("overall_recommendation", ""))).strip().lower()
    
        if approved:
            decision = "approved"
        elif safety_decision in {"block", "blocked", "deny", "denied"}:
            decision = "blocked"
        else:
            decision = "review_required"
    
        return {
            "status": decision,
            "approved": approved,
            "decision": decision,
            "safety": dict(safety_result),
            "alignment": dict(alignment_result),
        }

    def execute(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        goal = self._goal(payload)
        execution_task = goal.get("execution_task")
        if execution_task is None and self.config.require_explicit_execution_task:
            return {
                "status": "blocked",
                "success": False,
                "reason": "explicit_execution_task_required",
            }
        if execution_task is None:
            execution_task = {
                "name": str(goal.get("name") or goal.get("objective") or "autonomous_task"),
                "goal_type": "autonomous",
            }
        if not isinstance(execution_task, Mapping):
            raise ControlLoopContractError("goal.execution_task must be a mapping")
        task = dict(execution_task)
        task.setdefault("id", f"{payload.get('goal_id')}:cycle:{payload.get('cycle')}")
        task.setdefault("name", str(goal.get("name") or goal.get("objective") or "autonomous_task"))
        task.setdefault("goal_type", "autonomous")
        task.setdefault("metadata", {}) # type: ignore
        if isinstance(task["metadata"], Mapping):
            task["metadata"] = { # type: ignore
                **dict(task["metadata"]),
                "autonomous_run_id": payload.get("run_id"),
                "autonomous_cycle": payload.get("cycle"),
            }
        result = self._call_agent("execution", "perform_task", task)
        if not isinstance(result, Mapping):
            raise ControlLoopContractError("ExecutionAgent.perform_task() must return a mapping")
        return dict(result)

    def evaluate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        goal = self._goal(payload)
        evaluation_params = goal.get("evaluation_params")
    
        if evaluation_params is None:
            return {
                "status": "incomplete",
                "completed": False,
                "passed": False,
                "reason": "evaluation_params_required",
            }
    
        if not isinstance(evaluation_params, Mapping):
            raise ControlLoopContractError("goal.evaluation_params must be a mapping")
    
        execution = payload.get("execute", {})
        execution_succeeded = (
            isinstance(execution, Mapping)
            and _is_success(execution)
            and not _is_failed(execution)
        )
    
        raw_stage_metrics = payload.get("stage_metrics", {})
        if not isinstance(raw_stage_metrics, Mapping):
            raise ControlLoopContractError("stage_metrics must remain a mapping")
    
        params = dict(evaluation_params)
        params.setdefault("control_loop_execution", _safe_value(execution))
        params.setdefault("agent_performance_metrics", dict(raw_stage_metrics))
    
        result = self._call(
            self._agent("evaluation"),
            "execute_validation_cycle",
            params,
        )
    
        if not isinstance(result, Mapping):
            raise ControlLoopContractError(
                "EvaluationAgent.execute_validation_cycle() "
                "must return a mapping"
            )
    
        normalized = dict(result)
        normalized.setdefault("completed", execution_succeeded
                              and not _is_failed(normalized))
    
        return normalized

    def handle(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        failed_stage = str(payload.get("failed_stage", "")).strip().lower()
    
        target_name = self._last_failed_agent_name
        if target_name is None:
            target_name = self._STAGE_AGENT.get(failed_stage)
    
        # Prevent stale failure identity from leaking into another cycle.
        self._last_failed_agent_name = None
    
        target_agent = (
            self._agent(target_name)
            if target_name is not None
            else None
        )
    
        result = self._call(self._agent("handler"), "perform_task",
                            {
                                "error": payload.get("error"),
                                "target_agent": target_agent,
                                "task_data": self._goal(payload),
                                "context": {
                                    "source": "autonomous_control_loop",
                                    "run_id": payload.get("run_id"),
                                    "cycle": payload.get("cycle"),
                                    "stage": failed_stage,
                                    "agent": target_name,
                                },
                            },
                        )
    
        if not isinstance(result, Mapping):
            raise ControlLoopContractError("HandlerAgent.perform_task() must return a mapping")
    
        return dict(result)

    def stage_mapping(self) -> Dict[str, StageCallable]:
        return {
            "reason": self.reason,
            "plan": self.plan,
            "authorize": self.authorize,
            "execute": self.execute,
            "evaluate": self.evaluate,
        }


class AutonomousControlLoop:
    """One synchronous, bounded, safety-gated owner of autonomous progression."""

    STAGE_ORDER = ("reason", "plan", "authorize", "execute", "evaluate")

    def __init__(
        self,
        stages: Mapping[str, StageCallable],
        *,
        shared_memory: Any = None,
        handler: Optional[StageCallable] = None,
        observation_provider: Optional[StageCallable] = None,
        config: Optional[Mapping[str, Any] | AutonomousLoopConfig] = None,
    ) -> None:
        configured = get_config_section("autonomous_control_loop", default={})
        if isinstance(config, AutonomousLoopConfig):
            self.config = config
        else:
            if config:
                configured.update(dict(config))
            self.config = AutonomousLoopConfig.from_mapping(configured)

        self.stages = dict(stages)
        missing = [stage for stage in self.STAGE_ORDER if not callable(self.stages.get(stage))]
        if missing:
            raise ControlLoopConfigurationError(f"Missing callable autonomous stages: {missing}")
        if handler is not None and not callable(handler):
            raise ControlLoopConfigurationError("handler must be callable when provided")
        if observation_provider is not None and not callable(observation_provider):
            raise ControlLoopConfigurationError("observation_provider must be callable when provided")

        self.shared_memory = shared_memory
        self.handler = handler
        self.observation_provider = observation_provider
        self._lock = RLock()
        self._stop_event = Event()
        self._running = False
        self._state = ControlLoopState.IDLE
        self._runtime_status = RuntimeStatus()
        self._runtime_status.transition(RuntimeLifecycle.ACTIVE)
        self._last_result: Optional[AutonomousRunResult] = None

    @classmethod
    def from_factory(
        cls,
        factory: Any,
        *,
        shared_memory: Any,
        config: Optional[Mapping[str, Any] | AutonomousLoopConfig] = None,
        observation_provider: Optional[StageCallable] = None,
    ) -> "AutonomousControlLoop":
        configured = get_config_section("autonomous_control_loop", default={})
        if isinstance(config, AutonomousLoopConfig):
            resolved_config = config
        else:
            if config:
                configured.update(dict(config))
            resolved_config = AutonomousLoopConfig.from_mapping(configured)
        adapters = FactoryAutonomousStages(factory, shared_memory, resolved_config)
        return cls(
            adapters.stage_mapping(),
            shared_memory=shared_memory,
            handler=adapters.handle,
            observation_provider=observation_provider,
            config=resolved_config,
        )

    @property
    def state(self) -> ControlLoopState:
        with self._lock:
            return self._state

    @property
    def last_result(self) -> Optional[AutonomousRunResult]:
        with self._lock:
            return self._last_result

    def request_stop(self) -> None:
        """Cooperatively stop before the next stage boundary."""

        self._stop_event.set()

    def close(self) -> None:
        self.request_stop()
        with self._lock:
            if self._runtime_status.lifecycle in {RuntimeLifecycle.ACTIVE, RuntimeLifecycle.DEGRADED}:
                self._runtime_status.transition(RuntimeLifecycle.STOPPING)
                self._runtime_status.transition(RuntimeLifecycle.STOPPED)

    stop = close

    def health(self) -> Dict[str, Any]:
        payload = self._runtime_status.snapshot()
        payload.update(
            {
                "control_loop_state": self.state.value,
                "running": self._running,
                "last_run_id": self._last_result.run_id if self._last_result else None,
            }
        )
        return payload

    @staticmethod
    def _normalize_goal(goal: Mapping[str, Any] | str) -> Dict[str, Any]:
        if isinstance(goal, str):
            normalized: Dict[str, Any] = {"objective": goal}
        elif isinstance(goal, Mapping):
            normalized = dict(goal)
        else:
            raise ControlLoopContractError("Autonomous goal must be a mapping or non-empty string")
        objective = normalized.get("objective") or normalized.get("goal") or normalized.get("name")
        if not str(objective or "").strip():
            raise ControlLoopContractError("Autonomous goal requires objective, goal, or name")
        normalized.setdefault("objective", str(objective))
        normalized.setdefault("id", f"goal:{uuid.uuid4().hex[:12]}")
        return normalized

    def _publish(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if not self.config.publish_events or self.shared_memory is None:
            return
        event = {
            "event_type": event_type,
            "timestamp": time.time(),
            **{str(key): _safe_value(value) for key, value in payload.items()},
        }
        try:
            setter = getattr(self.shared_memory, "set", None)
            if callable(setter):
                setter(self.config.latest_state_key, event)
            publisher = getattr(self.shared_memory, "publish", None)
            if callable(publisher):
                publisher(self.config.event_channel, event)
            self._runtime_status.mark_recovered("telemetry", "autonomous_event.publish")
        except Exception as exc:
            self._runtime_status.mark_degraded("telemetry", "autonomous_event.publish", exc)
            logger.warning("Autonomous control-loop telemetry is degraded: %s", exc)

    @staticmethod
    def _validate_stage_output(stage: str, output: Mapping[str, Any]) -> None:
        status = _normalized_status(output)
        if stage == "reason" and status == "unknown":
            raise ControlLoopContractError("reason stage must declare status, success, passed, or completed")
        if stage == "plan":
            plan = output.get("plan")
            if status == "unknown":
                raise ControlLoopContractError("plan stage must declare an explicit status")
            if not _is_failed(output) and (
                not isinstance(plan, Sequence) or isinstance(plan, (str, bytes)) or not plan
            ):
                raise ControlLoopContractError("successful plan stage must return a non-empty plan sequence")
        if stage == "authorize":
            approved = output.get("approved")
            decision = str(output.get("decision", output.get("status", ""))).strip().lower()
            if "approved" in output and not isinstance(approved, bool):
                raise ControlLoopContractError("authorize.approved must be a boolean")
            if approved is None and decision not in {
                "allow",
                "approved",
                "block",
                "blocked",
                "human_review",
                "pass",
                "passed",
                "review",
                "review_required",
            }:
                raise ControlLoopContractError("authorize stage must declare approved or a recognized decision")
        if stage == "execute" and not (_is_success(output) or _is_failed(output)):
            raise ControlLoopContractError("execute stage must declare an explicit success or failure status")
        if stage == "evaluate":
            if "completed" in output and not isinstance(output.get("completed"), bool):
                raise ControlLoopContractError("evaluate.completed must be a boolean")
            if "passed" in output and not isinstance(output.get("passed"), bool):
                raise ControlLoopContractError("evaluate.passed must be a boolean")
            if (
                "completed" not in output
                and "passed" not in output
                and status
                not in {
                    "complete",
                    "completed",
                    "critical",
                    "error",
                    "fail",
                    "failed",
                    "incomplete",
                    "normal",
                    "ok",
                    "pass",
                    "passed",
                    "pending",
                    "retry",
                    "success",
                    "succeeded",
                }
            ):
                raise ControlLoopContractError("evaluate stage must declare completion, pass/fail, or retry status")

    def _invoke_stage(self, stage: str, payload: Mapping[str, Any]) -> StageResult:
        started = time.monotonic()
        try:
            output = self.stages[stage](payload)
            if not isinstance(output, Mapping):
                raise ControlLoopContractError(
                    f"Autonomous stage {stage!r} must return a mapping, got {type(output).__name__}"
                )
            normalized = {str(key): _safe_value(value) for key, value in output.items()}
            self._validate_stage_output(stage, normalized)
            # A safety denial and a negative evaluation are valid stage results,
            # not transport/runtime failures. The loop interprets those decisions
            # at their explicit gates below.
            decision_stage = stage in {"authorize", "evaluate"}
            return StageResult(
                stage=stage,
                status="failed" if _is_failed(normalized) and not decision_stage else "completed",
                duration_ms=max(0.0, (time.monotonic() - started) * 1000.0),
                output=normalized,
            )
        except Exception as exc:
            return StageResult(
                stage=stage,
                status="error",
                duration_ms=max(0.0, (time.monotonic() - started) * 1000.0),
                error=f"{type(exc).__name__}: {exc}",
            )

    def _handle_failure(self, payload: Mapping[str, Any], failed_stage: StageResult) -> Dict[str, Any]:
        if self.handler is None:
            return {"status": "skipped", "reason": "no_failure_handler"}
        handler_payload = {
            **dict(payload),
            "failed_stage": failed_stage.stage,
            "error": failed_stage.error or failed_stage.output.get("reason") or "stage_failed",
        }
        try:
            result = self.handler(handler_payload)
            if not isinstance(result, Mapping):
                raise ControlLoopContractError("Autonomous failure handler must return a mapping")
            return {str(key): _safe_value(value) for key, value in result.items()}
        except Exception as exc:
            self._runtime_status.mark_degraded("recovery", "autonomous_failure.handle", exc)
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    def _authorization_state(self, output: Mapping[str, Any]) -> Optional[ControlLoopState]:
        approved = output.get("approved")
        decision = str(output.get("decision", output.get("status", ""))).strip().lower()
        if approved is True or decision in {"allow", "approved", "pass", "passed"}:
            return None
        if not self.config.require_safety_approval and decision not in {"block", "blocked"}:
            return None
        if decision in {"review", "review_required", "human_review"}:
            return ControlLoopState.REVIEW_REQUIRED
        return ControlLoopState.BLOCKED

    @staticmethod
    def _evaluation_completed(output: Mapping[str, Any]) -> bool:
        if output.get("completed") is not None:
            return output.get("completed") is True
        if output.get("passed") is not None:
            return output.get("passed") is True
        return _is_success(output) and not _is_failed(output)

    def _result(
        self,
        *,
        run_id: str,
        goal_id: str,
        state: ControlLoopState,
        started_at: float,
        cycles: Sequence[ControlCycleResult],
        reason: str,
    ) -> AutonomousRunResult:
        result = AutonomousRunResult(
            run_id=run_id,
            goal_id=goal_id,
            state=state,
            started_at=started_at,
            finished_at=time.time(),
            cycles=tuple(cycles),
            reason=reason,
        )
        with self._lock:
            self._state = state
            self._last_result = result
        self._publish(
            "autonomy.run_finished",
            {
                "run_id": run_id,
                "goal_id": goal_id,
                "state": state.value,
                "reason": reason,
                "cycles": len(cycles),
            },
        )
        return result

    def run(
        self,
        goal: Mapping[str, Any] | str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> AutonomousRunResult:
        """Run one goal synchronously within configured safety and resource bounds."""

        normalized_goal = self._normalize_goal(goal)
        run_id = f"autonomy:{uuid.uuid4().hex}"
        goal_id = str(normalized_goal["id"])
        started_at = time.time()

        with self._lock:
            if self._running:
                raise ControlLoopBusyError("AutonomousControlLoop already has an active run")
            self._running = True
            self._state = ControlLoopState.RUNNING
            self._stop_event.clear()

        cycles: list[ControlCycleResult] = []
        run_started_monotonic = time.monotonic()
        previous_feedback: Dict[str, Any] = {}
        try:
            if not self.config.enabled:
                return self._result(
                    run_id=run_id,
                    goal_id=goal_id,
                    state=ControlLoopState.DISABLED,
                    started_at=started_at,
                    cycles=cycles,
                    reason="autonomous_control_loop_disabled",
                )

            self._publish("autonomy.run_started", {"run_id": run_id, "goal_id": goal_id})
            for cycle_number in range(1, self.config.max_cycles + 1):
                cycle_started = time.monotonic()
                if self._stop_event.is_set():
                    return self._result(
                        run_id=run_id,
                        goal_id=goal_id,
                        state=ControlLoopState.STOPPED,
                        started_at=started_at,
                        cycles=cycles,
                        reason="stop_requested",
                    )
                if cycle_started - run_started_monotonic > self.config.max_run_seconds:
                    return self._result(
                        run_id=run_id,
                        goal_id=goal_id,
                        state=ControlLoopState.EXHAUSTED,
                        started_at=started_at,
                        cycles=cycles,
                        reason="max_run_seconds_exceeded",
                    )

                payload: Dict[str, Any] = {
                    "run_id": run_id,
                    "goal_id": goal_id,
                    "cycle": cycle_number,
                    "goal": normalized_goal,
                    "context": dict(context or {}),
                    "previous_feedback": previous_feedback,
                    "stage_metrics": {},
                }
                if self.observation_provider is not None:
                    observation = self.observation_provider(payload)
                    if not isinstance(observation, Mapping):
                        raise ControlLoopContractError("observation_provider must return a mapping")
                    payload["observation"] = {str(key): _safe_value(value) for key, value in observation.items()}
                else:
                    payload["observation"] = dict((context or {}).get("observation", {})) if isinstance((context or {}).get("observation", {}), Mapping) else {}

                stage_results: list[StageResult] = []
                terminal_state: Optional[ControlLoopState] = None
                terminal_reason: Optional[str] = None

                for stage in self.STAGE_ORDER:
                    if self._stop_event.is_set():
                        terminal_state = ControlLoopState.STOPPED
                        terminal_reason = "stop_requested"
                        break
                    if time.monotonic() - cycle_started > self.config.max_cycle_seconds:
                        terminal_state = ControlLoopState.EXHAUSTED
                        terminal_reason = "max_cycle_seconds_exceeded"
                        break
                    if time.monotonic() - run_started_monotonic > self.config.max_run_seconds:
                        terminal_state = ControlLoopState.EXHAUSTED
                        terminal_reason = "max_run_seconds_exceeded"
                        break

                    stage_result = self._invoke_stage(stage, payload)
                    stage_results.append(stage_result)
                    payload["stage_metrics"][stage] = {
                        "latency_ms": stage_result.duration_ms,
                        "memory_mb": 0.0,
                    }
                    if time.monotonic() - cycle_started > self.config.max_cycle_seconds:
                        terminal_state = ControlLoopState.EXHAUSTED
                        terminal_reason = "max_cycle_seconds_exceeded"
                        break
                    if time.monotonic() - run_started_monotonic > self.config.max_run_seconds:
                        terminal_state = ControlLoopState.EXHAUSTED
                        terminal_reason = "max_run_seconds_exceeded"
                        break
                    if stage_result.status in {"error", "failed"}:
                        recovery = self._handle_failure(payload, stage_result)
                        previous_feedback = {
                            "failed_stage": stage,
                            "stage_output": stage_result.output,
                            "error": stage_result.error,
                            "recovery": recovery,
                        }
                        terminal_reason = f"stage_{stage}_{stage_result.status}"
                        if stage == "execute" and str(stage_result.output.get("status", "")).lower() in {
                            "block",
                            "blocked",
                        }:
                            terminal_state = ControlLoopState.BLOCKED
                        if self.config.stop_on_stage_error:
                            terminal_state = terminal_state or ControlLoopState.FAILED
                        break

                    payload[stage] = stage_result.output
                    if stage == "authorize":
                        authorization_state = self._authorization_state(stage_result.output)
                        if authorization_state is not None:
                            terminal_state = authorization_state
                            terminal_reason = "safety_review_required" if authorization_state == ControlLoopState.REVIEW_REQUIRED else "safety_blocked"
                            break
                    if stage == "evaluate":
                        previous_feedback = dict(stage_result.output)
                        if self._evaluation_completed(stage_result.output):
                            terminal_state = ControlLoopState.SUCCEEDED
                            terminal_reason = "evaluation_completed"

                cycle_status = terminal_state.value if terminal_state is not None else "incomplete"
                cycles.append(
                    ControlCycleResult(
                        cycle=cycle_number,
                        status=cycle_status,
                        duration_ms=max(0.0, (time.monotonic() - cycle_started) * 1000.0),
                        stages=tuple(stage_results),
                        reason=terminal_reason,
                    )
                )
                self._publish(
                    "autonomy.cycle_finished",
                    {
                        "run_id": run_id,
                        "goal_id": goal_id,
                        "cycle": cycle_number,
                        "status": cycle_status,
                        "reason": terminal_reason,
                        "stages": [
                            {"stage": item.stage, "status": item.status, "duration_ms": item.duration_ms}
                            for item in stage_results
                        ],
                    },
                )

                if terminal_state is not None:
                    return self._result(
                        run_id=run_id,
                        goal_id=goal_id,
                        state=terminal_state,
                        started_at=started_at,
                        cycles=cycles,
                        reason=terminal_reason or terminal_state.value,
                    )

            return self._result(
                run_id=run_id,
                goal_id=goal_id,
                state=ControlLoopState.EXHAUSTED,
                started_at=started_at,
                cycles=cycles,
                reason="max_cycles_exhausted",
            )
        except Exception as exc:
            logger.error("Autonomous control loop failed: %s", exc)
            return self._result(
                run_id=run_id,
                goal_id=goal_id,
                state=ControlLoopState.FAILED,
                started_at=started_at,
                cycles=cycles,
                reason=f"{type(exc).__name__}: {exc}",
            )
        finally:
            with self._lock:
                self._running = False


__all__ = [
    "AutonomousControlLoop",
    "AutonomousControlLoopError",
    "AutonomousLoopConfig",
    "AutonomousRunResult",
    "ControlCycleResult",
    "ControlLoopBusyError",
    "ControlLoopConfigurationError",
    "ControlLoopContractError",
    "ControlLoopState",
    "FactoryAutonomousStages",
    "StageResult",
]
