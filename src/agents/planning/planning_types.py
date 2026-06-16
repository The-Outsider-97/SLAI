"""
Planning Types – Core data structures for planning, scheduling, memory, safety,
and execution orchestration.

This module defines the canonical type contract used by the planning subsystem:
resource profiles, temporal constraints, safety reports, plan snapshots, runtime
adjustments, and the Task work unit.  It intentionally keeps the existing public
API shape while adding stricter validation, safer serialisation, lifecycle
helpers, and config-driven defaults.
"""

from __future__ import annotations

import copy
import time

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any as AnyType, Callable, Dict, Iterable, List, Optional, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.planning_errors import * # type: ignore
from .utils.planning_helpers import * # type: ignore
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Planning Types")
printer = PrettyPrinter()


# -----------------------------------------------------------------------------
# Module configuration
# -----------------------------------------------------------------------------
def _planning_types_config() -> Dict[str, AnyType]:
    """Load the planning_types config section without changing config handling."""
    config = load_global_config()
    section = get_config_section("planning_types", config=config, default={})
    return deep_update(
        {
            "validation_level": "strict",
            "type_checks": "enabled",
            "default_task_name": "Planning Task",
            "default_task_version": "1.0",
            "default_owner": "system",
            "default_duration": 300.0,
            "default_estimated_duration": 0.0,
            "default_cost": 1.0,
            "default_priority": 1,
            "default_max_retries": 3,
            "default_success_threshold": 0.9,
            "relative_time_threshold_seconds": 1_000_000_000,
            "max_progress": 1.0,
            "allowed_criticalities": ["low", "medium", "high", "critical"],
            "allowed_adjustments": ["modify_task", "add_task", "remove_task"],
            "default_safety_margins": {
                "gpu_buffer": 0.15,
                "ram_buffer": 0.20,
                "min_task_duration": 30,
                "max_concurrent": 5,
                "time_buffer": 120,
            },
        },
        section,
    )


def _is_strict_validation() -> bool:
    level = str(_planning_types_config().get("validation_level", "strict")).lower()
    return level in {"strict", "enabled", "true", "1"}


def _now() -> float:
    return time.time()


# -----------------------------------------------------------------------------
# Any – typed container with constraints
# -----------------------------------------------------------------------------
class Any:
    """
    Universal value container with runtime constraints.

    Constraint values may be Python types, callable predicates, or string domain
    tags. Callable constraints are validated at runtime but are represented by
    name during serialisation because arbitrary functions cannot be safely
    reconstructed from JSON.
    """

    __slots__ = ("_value", "_type", "_constraints")

    def __init__(self, value: object, constraints: Tuple[AnyType, ...] = ()) -> None:
        require_type(constraints, tuple, "constraints")
        self._value = value
        self._type = type(value)
        self._constraints = constraints
        self._validate(constraints)

    @property
    def value(self) -> object:
        return self._value

    @property
    def type(self) -> type:
        return self._type

    @property
    def constraints(self) -> Tuple[AnyType, ...]:
        return self._constraints

    def _validate(self, constraints: Tuple[AnyType, ...]) -> None:
        for constraint in constraints:
            if isinstance(constraint, type):
                if not isinstance(self._value, constraint):
                    raise AcademicPlanningError(
                        f"Value {self._value!r} violates type constraint {constraint.__name__}"
                    )
            elif isinstance(constraint, str):
                continue
            elif callable(constraint):
                try:
                    valid = bool(constraint(self._value))
                except Exception as exc:
                    raise AcademicPlanningError(
                        f"Constraint {getattr(constraint, '__name__', 'anonymous')} raised: {exc}"
                    ) from exc
                if not valid:
                    raise AcademicPlanningError(
                        f"Value {self._value!r} violates predicate constraint "
                        f"{getattr(constraint, '__name__', 'anonymous')}"
                    )
            else:
                raise AcademicPlanningError(f"Invalid constraint type: {type(constraint).__name__}")

    def is_compatible(self, other: "Any") -> bool:
        require_type(other, Any, "other")
        if not issubclass(self._type, other.type):
            return False
        for constraint in other.constraints:
            if callable(constraint):
                try:
                    if not constraint(self.value):
                        return False
                except Exception:
                    return False
        return True

    def constrain(self, new_constraints: Tuple[AnyType, ...]) -> "Any":
        require_type(new_constraints, tuple, "new_constraints")
        return Any(self._value, self._constraints + new_constraints)

    def to_json(self) -> Dict[str, AnyType]:
        return {
            "value": self._value,
            "type": self._type.__name__,
            "constraints": [
                c.__name__ if isinstance(c, type) else getattr(c, "__name__", c)
                for c in self._constraints
            ],
        }

    @classmethod
    def from_json(cls, data: Dict[str, AnyType]) -> "Any":
        require_type(data, dict, "data")
        type_map = {t.__name__: t for t in (int, float, str, bool, list, dict, tuple)}
        constraints: List[AnyType] = []
        for item in data.get("constraints", []):
            if item in type_map:
                constraints.append(type_map[item])
            elif isinstance(item, str):
                constraints.append(item)
            else:
                raise AcademicPlanningError(f"Unreconstructible constraint: {item!r}")
        return cls(data.get("value"), tuple(constraints))

    def __add__(self, other: object) -> "Any":
        if not isinstance(other, Any):
            other = Any(other)
        if self.type is not other.type:
            raise AcademicPlanningError(
                f"Additive type mismatch: {self.type.__name__} != {other.type.__name__}"
            )
        return Any(self.value + other.value, self._constraints)  # type: ignore[operator]

    def __radd__(self, other: object) -> "Any":
        return self.__add__(other)

    def __eq__(self, other: object) -> bool:
        return self.value == (other.value if isinstance(other, Any) else other)

    def __repr__(self) -> str:
        return f"Any<{self._type.__name__}>({self._value!r})"


# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
WorldState = Tuple[Tuple[str, AnyType], ...]
MethodSignature = Tuple[str, int]
TemporalRelation = Tuple["Task", "Task", str]
MemoKey = Tuple[MethodSignature, WorldState]
PlanStep = Tuple[int, "Task", MethodSignature]


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------
class TaskStatus(Enum):
    PENDING = 0
    EXECUTING = 1
    SUCCESS = 2
    FAILED = 3
    CANCELLED = 4
    BLOCKED = 5

    @property
    def is_terminal(self) -> bool:
        return self in {TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.CANCELLED}


class TaskType(Enum):
    PRIMITIVE = 0
    ABSTRACT = 1
    COMPOSITE = 2


def _coerce_enum(value: AnyType, enum_cls: type[Enum], field_name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        key = value.upper()
        if key in enum_cls.__members__:
            return enum_cls[key]
    if isinstance(value, int):
        try:
            return enum_cls(value)
        except ValueError:
            pass
    raise PlanningConfigError(
        f"Invalid {field_name}: {value!r}",
        config_key=field_name,
        config_section="planning_types",
        expected_type=f"{enum_cls.__name__} member",
    )


# -----------------------------------------------------------------------------
# Resource and safety dataclasses
# -----------------------------------------------------------------------------
@dataclass
class ResourceProfile:
    gpu: float = 0.0
    ram: float = 0.0
    specialized_hardware: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.gpu = float(self.gpu or 0.0)
        self.ram = float(self.ram or 0.0)
        require_non_negative(self.gpu, "resource_profile.gpu")
        require_non_negative(self.ram, "resource_profile.ram")
        if self.specialized_hardware is None:
            self.specialized_hardware = []
        require_type(self.specialized_hardware, list, "resource_profile.specialized_hardware")
        self.specialized_hardware = list(dict.fromkeys(str(hw) for hw in self.specialized_hardware if hw))

    def count_requirements(self) -> int:
        return int(self.gpu > 0) + int(self.ram > 0) + len(self.specialized_hardware)

    def to_dict(self) -> Dict[str, AnyType]:
        return {
            "gpu": self.gpu,
            "ram": self.ram,
            "specialized_hardware": list(self.specialized_hardware),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, AnyType]]) -> "ResourceProfile":
        if data is None:
            return cls()
        require_type(data, dict, "resource_profile")
        return cls(
            gpu=float(data.get("gpu", 0.0) or 0.0),
            ram=float(data.get("ram", 0.0) or 0.0),
            specialized_hardware=list(data.get("specialized_hardware", []) or []),
        )

    def fits_within(self, available: "ClusterResources") -> bool:
        require_type(available, ClusterResources, "available")
        if self.gpu > available.gpu_total or self.ram > available.ram_total:
            return False
        return set(self.specialized_hardware).issubset(set(available.specialized_hardware_available))

    def __add__(self, other: "ResourceProfile") -> "ResourceProfile":
        require_type(other, ResourceProfile, "other")
        return ResourceProfile(
            gpu=self.gpu + other.gpu,
            ram=self.ram + other.ram,
            specialized_hardware=list(dict.fromkeys(self.specialized_hardware + other.specialized_hardware)),
        )

    def __sub__(self, other: "ResourceProfile") -> "ResourceProfile":
        require_type(other, ResourceProfile, "other")
        return ResourceProfile(
            gpu=max(0.0, self.gpu - other.gpu),
            ram=max(0.0, self.ram - other.ram),
            specialized_hardware=[hw for hw in self.specialized_hardware if hw not in set(other.specialized_hardware)],
        )


@dataclass
class ClusterResources:
    gpu_total: float = 0.0
    ram_total: float = 0.0
    specialized_hardware_available: List[str] = field(default_factory=list)
    current_allocations: Dict[str, ResourceProfile] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.gpu_total = float(self.gpu_total or 0.0)
        self.ram_total = float(self.ram_total or 0.0)
        require_non_negative(self.gpu_total, "cluster_resources.gpu_total")
        require_non_negative(self.ram_total, "cluster_resources.ram_total")
        self.specialized_hardware_available = list(
            dict.fromkeys(str(hw) for hw in (self.specialized_hardware_available or []) if hw)
        )
        self.current_allocations = {
            str(k): (v if isinstance(v, ResourceProfile) else ResourceProfile.from_dict(v))
            for k, v in dict(self.current_allocations or {}).items()
        }

    def allocated_totals(self) -> ResourceProfile:
        total = ResourceProfile()
        for profile in self.current_allocations.values():
            total = total + profile
        return total

    def available_profile(self) -> ResourceProfile:
        allocated = self.allocated_totals()
        return ResourceProfile(
            gpu=max(0.0, self.gpu_total - allocated.gpu),
            ram=max(0.0, self.ram_total - allocated.ram),
            specialized_hardware=[
                hw for hw in self.specialized_hardware_available
                if hw not in set(allocated.specialized_hardware)
            ],
        )

    def can_allocate(self, requirements: ResourceProfile) -> bool:
        return requirements.fits_within(
            ClusterResources(
                gpu_total=self.available_profile().gpu,
                ram_total=self.available_profile().ram,
                specialized_hardware_available=self.available_profile().specialized_hardware,
            )
        )

    def allocate(self, task_id: str, requirements: ResourceProfile) -> None:
        validate_task_id(task_id, "cluster_allocation")
        if not self.can_allocate(requirements):
            available = self.available_profile()
            raise ResourceAcquisitionError(
                f"Insufficient resources for task {task_id}",
                resource_type="cluster",
                requested=requirements.to_dict(),
                available=available.to_dict(),
                task_id=task_id,
            )
        self.current_allocations[task_id] = copy.deepcopy(requirements)

    def release(self, task_id: str) -> ResourceProfile:
        validate_task_id(task_id, "cluster_release")
        return self.current_allocations.pop(task_id, ResourceProfile())

    def to_dict(self) -> Dict[str, AnyType]:
        return {
            "gpu_total": self.gpu_total,
            "ram_total": self.ram_total,
            "specialized_hardware_available": list(self.specialized_hardware_available),
            "current_allocations": {
                task_id: profile.to_dict() for task_id, profile in self.current_allocations.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, AnyType]]) -> "ClusterResources":
        if data is None:
            return cls()
        require_type(data, dict, "cluster_resources")
        return cls(
            gpu_total=float(data.get("gpu_total", 0.0) or 0.0),
            ram_total=float(data.get("ram_total", 0.0) or 0.0),
            specialized_hardware_available=list(data.get("specialized_hardware_available", []) or []),
            current_allocations={
                str(k): ResourceProfile.from_dict(v)
                for k, v in dict(data.get("current_allocations", {}) or {}).items()
            },
        )


@dataclass
class RepairCandidate:
    strategy: str
    repaired_plan: List["Task"]
    estimated_cost: float
    risk_assessment: Dict[str, AnyType]

    def __post_init__(self) -> None:
        require_non_empty(self.strategy, "repair_candidate.strategy")
        require_non_negative(float(self.estimated_cost), "repair_candidate.estimated_cost")
        require_type(self.risk_assessment, dict, "repair_candidate.risk_assessment")

    def to_dict(self) -> Dict[str, AnyType]:
        return {
            "strategy": self.strategy,
            "repaired_plan": [t.to_dict() if hasattr(t, "to_dict") else str(t) for t in self.repaired_plan],
            "estimated_cost": self.estimated_cost,
            "risk_assessment": copy.deepcopy(self.risk_assessment),
        }


@dataclass
class Adjustment:
    type: str
    task_id: Optional[str] = None
    task: Optional["Task"] = None
    updates: Optional[Dict[str, AnyType]] = None
    priority: int = 3
    cascade: bool = False
    origin: str = "api"
    timestamp: float = field(default_factory=_now)
    _retry_count: int = 0

    def __post_init__(self) -> None:
        cfg = _planning_types_config()
        allowed = set(cfg.get("allowed_adjustments", ["modify_task", "add_task", "remove_task"]))
        if self.type not in allowed:
            raise AdjustmentError(
                f"Unsupported adjustment type: {self.type}",
                adjustment=self.to_dict(include_task=False),
                conflict_details={"allowed": sorted(allowed)},
            )
        if self.task_id:
            validate_task_id(self.task_id, "adjustment")
        if self.updates is not None:
            require_type(self.updates, dict, "adjustment.updates")
        self.priority = int(self.priority)
        self._retry_count = int(max(0, self._retry_count))

    def to_dict(self, *, include_task: bool = True) -> Dict[str, AnyType]:
        return {
            "type": self.type,
            "task_id": self.task_id,
            "task": self.task.to_dict() if include_task and self.task is not None else None,
            "updates": copy.deepcopy(self.updates or {}),
            "priority": self.priority,
            "cascade": self.cascade,
            "origin": self.origin,
            "timestamp": self.timestamp,
            "_retry_count": self._retry_count,
        }


@dataclass
class PerformanceMetrics:
    timestamp: float = field(default_factory=_now)
    system_load: float = 0.0
    network_latency: float = -1.0
    service_health: Dict[str, str] = field(default_factory=dict)
    plan_execution_rate: float = 0.0

    def __post_init__(self) -> None:
        self.timestamp = float(self.timestamp or _now())
        self.system_load = clamp(float(self.system_load or 0.0), 0.0, 1.0)
        self.network_latency = float(self.network_latency)
        require_non_negative(float(self.plan_execution_rate), "performance.plan_execution_rate")
        self.service_health = {str(k): str(v) for k, v in dict(self.service_health or {}).items()}

    def to_dict(self) -> Dict[str, AnyType]:
        return asdict(self)


@dataclass
class PlanSnapshot:
    timestamp: float = field(default_factory=_now)
    task_ids: List[str] = field(default_factory=list)
    resource_utilization: Dict[str, Union[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamp = float(self.timestamp or _now())
        self.task_ids = [str(tid) for tid in self.task_ids]
        self.resource_utilization = dict(self.resource_utilization or {})

    def to_dict(self) -> Dict[str, AnyType]:
        return asdict(self)


@dataclass
class TemporalConstraints:
    start_time: float = 0.0
    end_time: float = 0.0
    min_duration: float = 0.0
    max_duration: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    max_wait: float = 0.0
    time_buffer: float = 0.0
    constraints: List[Callable[[float], bool]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in ("start_time", "end_time", "min_duration", "max_duration", "max_wait", "time_buffer"):
            setattr(self, name, float(getattr(self, name) or 0.0))
            require_non_negative(getattr(self, name), f"temporal_constraints.{name}")
        self.dependencies = [str(dep) for dep in (self.dependencies or [])]
        if self.min_duration and self.max_duration and self.min_duration > self.max_duration:
            raise TemporalViolation(
                "min_duration cannot exceed max_duration",
                violation_type="duration",
                constraint_details={"min_duration": self.min_duration, "max_duration": self.max_duration},
            )
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise TemporalViolation(
                "start_time cannot be after end_time",
                violation_type="window",
                constraint_details={"start_time": self.start_time, "end_time": self.end_time},
            )

    def validate(self, current_time: float) -> bool:
        if self.start_time > 0.0 and current_time < self.start_time:
            return False
        if self.end_time > 0.0 and current_time > self.end_time:
            return False
        for constraint in self.constraints:
            try:
                if not constraint(current_time):
                    return False
            except Exception as exc:
                raise TemporalViolation(
                    f"Temporal constraint raised: {exc}",
                    violation_type="custom_constraint",
                    time_delta=0.0,
                ) from exc
        return True

    def to_dict(self) -> Dict[str, AnyType]:
        data = asdict(self)
        data["constraints"] = [getattr(c, "__name__", "anonymous") for c in self.constraints]
        return data


@dataclass
class SafetyViolation:
    violation_type: str
    resource: str
    measured_value: float
    threshold: float
    task_id: str
    timestamp: float = field(default_factory=_now)
    severity: str = "medium"
    corrective_action: str = ""
    impact_analysis: Dict[str, AnyType] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.violation_type, "safety_violation.violation_type")
        require_non_empty(self.resource, "safety_violation.resource")
        if self.task_id:
            validate_task_id(self.task_id, "safety_violation")
        self.measured_value = float(self.measured_value)
        self.threshold = float(self.threshold)
        allowed = {"low", "medium", "high", "critical"}
        if self.severity not in allowed:
            raise SafetyMarginError(
                f"Invalid safety severity: {self.severity}",
                resource_type=self.resource,
                requested=self.measured_value,
                available=self.threshold,
            )

    def to_dict(self) -> Dict[str, AnyType]:
        return asdict(self)


@dataclass
class SafetyMargins:
    gpu_buffer: float = 0.15
    ram_buffer: float = 0.20
    min_task_duration: float = 30.0
    max_concurrent: int = 5
    time_buffer: float = 120.0

    def __post_init__(self) -> None:
        validate_probability(float(self.gpu_buffer), "safety_margins.gpu_buffer")
        validate_probability(float(self.ram_buffer), "safety_margins.ram_buffer")
        require_non_negative(float(self.min_task_duration), "safety_margins.min_task_duration")
        require_positive(int(self.max_concurrent), "safety_margins.max_concurrent")
        require_non_negative(float(self.time_buffer), "safety_margins.time_buffer")
        self.gpu_buffer = float(self.gpu_buffer)
        self.ram_buffer = float(self.ram_buffer)
        self.min_task_duration = float(self.min_task_duration)
        self.max_concurrent = int(self.max_concurrent)
        self.time_buffer = float(self.time_buffer)

    @classmethod
    def from_config(cls, config: Optional[Dict[str, AnyType]] = None) -> "SafetyMargins":
        cfg = config if config is not None else load_global_config()
        safety_cfg = get_config_section("safety_margins", config=cfg, default={})
        defaults = _planning_types_config().get("default_safety_margins", {})
        resource_buffers = safety_cfg.get("resource_buffers", {}) if isinstance(safety_cfg, dict) else {}
        temporal = safety_cfg.get("temporal", {}) if isinstance(safety_cfg, dict) else {}
        return cls(
            gpu_buffer=float(resource_buffers.get("gpu", defaults.get("gpu_buffer", 0.15))),
            ram_buffer=float(resource_buffers.get("ram", defaults.get("ram_buffer", 0.20))),
            min_task_duration=float(temporal.get("min_task_duration", defaults.get("min_task_duration", 30.0))),
            max_concurrent=int(temporal.get("max_concurrent", defaults.get("max_concurrent", 5))),
            time_buffer=float(temporal.get("time_buffer", defaults.get("time_buffer", 120.0))),
        )

    def to_dict(self) -> Dict[str, AnyType]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Utility conversion helpers
# -----------------------------------------------------------------------------
def to_world_state_tuple(state: Dict[str, AnyType]) -> WorldState:
    """Return a stable immutable representation of a world-state dictionary."""
    require_type(state, dict, "world_state")
    items: List[Tuple[str, AnyType]] = []
    for key, value in state.items():
        if isinstance(value, Any):
            value = value.value
        try:
            hash(value)
            safe_value = value
        except Exception:
            safe_value = safe_json_dumps(value, fallback_str=str(value))
        items.append((str(key), safe_value))
    return tuple(sorted(items, key=lambda item: item[0]))


def from_world_state_tuple(state: WorldState) -> Dict[str, AnyType]:
    require_type(state, tuple, "world_state_tuple")
    return dict(state)


def _serialise_value(value: AnyType) -> AnyType:
    if isinstance(value, Any):
        return {"__planning_any__": value.to_json()}
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, ResourceProfile):
        return value.to_dict()
    if isinstance(value, ClusterResources):
        return value.to_dict()
    if isinstance(value, Task):
        return value.to_dict(include_methods=False, include_children=False)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _serialise_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialise_value(v) for v in value]
    if callable(value):
        return getattr(value, "__name__", "anonymous")
    return value


# -----------------------------------------------------------------------------
# Task – the core work unit
# -----------------------------------------------------------------------------
class Task:
    """Represents a unit of work in the planning system."""

    _id_counter = 0

    def __init__(
        self,
        name: Optional[str] = None,
        task_type: Union[TaskType, str, int] = TaskType.ABSTRACT,
        **kwargs: AnyType,
    ) -> None:
        cfg = _planning_types_config()
        Task._id_counter += 1

        self.id: str = str(kwargs.pop("id", f"task_{int(time.time() * 1000)}_{Task._id_counter}"))
        if not is_valid_task_id(self.id):
            raise PlanningConfigError(
                f"Invalid task id: {self.id!r}",
                config_key="task.id",
                config_section="planning_types",
                expected_type="valid task id",
            )

        self.name: str = str(name if name is not None else cfg.get("default_task_name", "Planning Task"))
        require_non_empty(self.name, "task.name")
        self.task_type: TaskType = _coerce_enum(task_type, TaskType, "task_type")  # type: ignore[assignment]
        self.type: TaskType = self.task_type
        self.status: TaskStatus = TaskStatus.PENDING

        self.parent: Optional["Task"] = None
        self.parent_task: Optional["Task"] = None
        self.children: List["Task"] = []
        self.methods: List[List["Task"]] = []
        self.selected_method: int = 0

        self.goal_state: Optional[Dict[str, AnyType]] = None
        self.duration: float = float(cfg.get("default_duration", 300.0))
        self.estimated_duration: float = float(cfg.get("default_estimated_duration", 0.0))
        self.actual_duration: float = 0.0
        self.cost: float = float(cfg.get("default_cost", 1.0))
        self.is_probabilistic: bool = False
        self.probabilistic_actions: List[AnyType] = []
        self.success_threshold: float = float(cfg.get("default_success_threshold", 0.9))
        self.risk_score: float = 0.0
        self.dependencies: List[str] = []
        self.execution_modes: List[str] = ["full"]
        self.description: str = "No description provided"
        self.created_at: float = _now()
        self.creation_time: float = self.created_at
        self.last_updated: float = self.created_at
        self.owner: str = str(cfg.get("default_owner", "system"))
        self.required_skills: List[str] = []
        self.progress: float = 0.0
        self.required_tools: List[str] = []
        self.location: str = "unspecified"
        self.retry_count: int = 0
        self.max_retries: int = int(cfg.get("default_max_retries", 3))
        self.timeout: float = 0.0
        self.criticality: str = "medium"
        self.category: str = "general"
        self.parameters: Dict[str, AnyType] = {}
        self.preconditions: List[Callable[[Dict[str, AnyType]], bool]] = []
        self.effects: List[Callable[[Dict[str, AnyType]], None]] = []
        self.precondition_errors: List[str] = []
        self.effect_errors: List[str] = []
        self.history: List[Dict[str, AnyType]] = []
        self.context: Dict[str, AnyType] = {}
        self.energy_consumption: float = 0.0
        self.data_requirements: Dict[str, AnyType] = {}
        self.safety_constraints: List[str] = []
        self.quality_metrics: Dict[str, float] = {}
        self.failure_reason: str = ""
        self.recovery_strategy: str = ""
        self.parallelizable: bool = False
        self.human_interaction_required: bool = False
        self.verification_method: str = "automatic"
        self.documentation: str = ""
        self.tags: List[str] = []
        self.version: str = str(cfg.get("default_task_version", "1.0"))
        self.source: str = "internal"
        self.expected_outcome: str = ""
        self.actual_outcome: str = ""
        self.sensor_requirements: List[str] = []
        self.communication_requirements: Dict[str, AnyType] = {}
        self.environmental_constraints: Dict[str, AnyType] = {}
        self.compliance_requirements: List[str] = []
        self.optimization_metrics: List[str] = []
        self.learning_curve: float = 0.0
        self.example_goal: Optional[Dict[str, AnyType]] = None
        self.resource_requirements: ResourceProfile = ResourceProfile()
        self.temporal_constraints: Optional[TemporalConstraints] = None
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.deadline: float = 0.0
        self.priority: int = int(cfg.get("default_priority", 1))
        self.workload: float = 0.0                     # Abstract effort (e.g., story points)
        self.urgency: int = 0                         # 0-10, separate from priority
        self.estimated_effort: float = 0.0            # Person‑hours or effort units
        self.actual_effort: float = 0.0
        self.assigned_agent: Optional[str] = None
        self.execution_node: str = "unspecified"
        self.retry_delay: float = 1.0                 # Base delay between retries (seconds)
        self.timeout_strategy: str = "fail"           # fail | retry | escalate
        self.blocked_by: List[str] = []               # Resource / condition IDs blocking this task
        self.outputs: Dict[str, AnyType] = {}         # Produced artifacts
        self.inputs: Dict[str, AnyType] = {}          # Required inputs
        self.validation_criteria: List[Callable[[Dict[str, AnyType]], bool]] = []
        self.rollback_action: Optional[Callable[[Dict[str, AnyType]], None]] = None
        self.monitoring_metrics: Dict[str, float] = {}  # Runtime measurements (e.g., peak memory)
        self.cost_model: str = "fixed"                # fixed | linear | per_unit
        self.resource_limits: Dict[str, float] = {}   # Per‑resource max (overrides cluster limits)
        self.security_level: int = 0                  # 0 (public) to 10 (top secret)
        self.environment: Dict[str, str] = {}         # Environment variable overrides
        self.on_success: Optional[Callable[[], None]] = None
        self.on_failure: Optional[Callable[[Exception], None]] = None
        self.on_timeout: Optional[Callable[[], None]] = None

        for key, value in kwargs.items():
            if key == "task_type":
                self.task_type = _coerce_enum(value, TaskType, "task_type")  # type: ignore[assignment]
                self.type = self.task_type
            elif key == "status":
                self.status = _coerce_enum(value, TaskStatus, "status")  # type: ignore[assignment]
            elif key == "resource_requirements":
                self.resource_requirements = value if isinstance(value, ResourceProfile) else ResourceProfile.from_dict(value)
            elif key == "requirements":
                self.resource_requirements = value if isinstance(value, ResourceProfile) else ResourceProfile.from_dict(value)
            elif key == "temporal_constraints":
                if value is None:
                    self.temporal_constraints = None
                else:
                    self.temporal_constraints = value if isinstance(value, TemporalConstraints) else TemporalConstraints(**value)
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning("Ignoring unknown Task field %s for task %s", key, self.name)

        self._post_init()

    # ------------------------------------------------------------------
    # Validation and lifecycle
    # ------------------------------------------------------------------
    def _post_init(self) -> None:
        cfg = _planning_types_config()
        self.task_type = _coerce_enum(self.task_type, TaskType, "task_type")  # type: ignore[assignment]
        self.type = self.task_type
        self.status = _coerce_enum(self.status, TaskStatus, "status")  # type: ignore[assignment]
        self.duration = max(0.0, float(self.duration or 0.0))
        self.estimated_duration = float(self.estimated_duration or 0.0)
        if self.estimated_duration <= 0.0 and self.duration > 0.0:
            self.estimated_duration = self.duration
        self.actual_duration = max(0.0, float(self.actual_duration or 0.0))
        self.cost = max(0.0, float(self.cost or 0.0))
        self.priority = int(self.priority or 0)
        self.retry_count = int(max(0, self.retry_count))
        self.max_retries = int(max(0, self.max_retries))
        self.progress = clamp(float(self.progress or 0.0), 0.0, float(cfg.get("max_progress", 1.0)))
        self.success_threshold = clamp(float(self.success_threshold), 0.0, 1.0)
        self.risk_score = clamp(float(self.risk_score or 0.0), 0.0, 1.0)
        self.dependencies = [str(dep) for dep in (self.dependencies or [])]
        self.execution_modes = [str(mode) for mode in (self.execution_modes or ["full"])]
        self.tags = [str(tag) for tag in (self.tags or [])]
        self.required_skills = [str(skill) for skill in (self.required_skills or [])]
        self.required_tools = [str(tool) for tool in (self.required_tools or [])]
        self.resource_requirements = (
            self.resource_requirements
            if isinstance(self.resource_requirements, ResourceProfile)
            else ResourceProfile.from_dict(self.resource_requirements)
        )
        allowed_criticalities = set(cfg.get("allowed_criticalities", ["low", "medium", "high", "critical"]))
        if self.criticality not in allowed_criticalities:
            if _is_strict_validation():
                raise PlanningConfigError(
                    f"Invalid task criticality: {self.criticality}",
                    config_key="task.criticality",
                    config_section="planning_types",
                    expected_type=f"one of {sorted(allowed_criticalities)}",
                )
            self.criticality = "medium"

        self.urgency = max(0, min(10, int(self.urgency or 0)))
        self.security_level = max(0, min(10, int(self.security_level or 0)))
        allowed_timeout_strategies = {"fail", "retry", "escalate"}
        if self.timeout_strategy not in allowed_timeout_strategies:
            if _is_strict_validation():
                raise PlanningConfigError(
                    f"Invalid timeout_strategy: {self.timeout_strategy}",
                    config_key="task.timeout_strategy",
                    config_section="planning_types",
                    expected_type=f"one of {sorted(allowed_timeout_strategies)}",
                )
            self.timeout_strategy = "fail"
        allowed_cost_models = {"fixed", "linear", "per_unit"}
        if self.cost_model not in allowed_cost_models:
            if _is_strict_validation():
                raise PlanningConfigError(
                    f"Invalid cost_model: {self.cost_model}",
                    config_key="task.cost_model",
                    config_section="planning_types",
                    expected_type=f"one of {sorted(allowed_cost_models)}",
                )
            self.cost_model = "fixed"

        threshold = float(cfg.get("relative_time_threshold_seconds", 1_000_000_000))
        now = _now()
        for attr in ("start_time", "deadline", "end_time"):
            value = float(getattr(self, attr, 0.0) or 0.0)
            if 0.0 < value < threshold:
                value = now + value
            setattr(self, attr, value)
        if self.end_time == 0.0 and self.start_time > 0.0 and self.duration > 0.0:
            self.end_time = self.start_time + self.duration
        if self.deadline and self.start_time and self.deadline < self.start_time and _is_strict_validation():
            raise TemporalViolation(
                "Task deadline cannot be before start_time",
                violation_type="window",
                task_name=self.name,
                task_id=self.id,
                constraint_details={"start_time": self.start_time, "deadline": self.deadline},
            )
        self.last_updated = _now()

    def validate(self) -> None:
        require_non_empty(self.id, "task.id")
        validate_task_id(self.id, "task.validate")
        require_non_empty(self.name, "task.name")
        require_type(self.resource_requirements, ResourceProfile, "task.resource_requirements")
        validate_probability(self.success_threshold, "task.success_threshold")
        validate_probability(self.risk_score, "task.risk_score")
        for dep in self.dependencies:
            validate_task_id(dep, "task.dependencies")

    def update_status(self, status: Union[TaskStatus, str, int], *, reason: str = "") -> None:
        self.status = _coerce_enum(status, TaskStatus, "status")  # type: ignore[assignment]
        if self.status == TaskStatus.EXECUTING and self.start_time <= 0.0:
            self.start_time = _now()
        if self.status.is_terminal:
            self.end_time = _now()
            if self.start_time > 0.0:
                self.actual_duration = max(0.0, self.end_time - self.start_time)
        if self.status == TaskStatus.FAILED and reason:
            self.failure_reason = reason
        self.history.append({"timestamp": _now(), "status": self.status.name, "reason": reason})
        self.last_updated = _now()

    def mark_success(self, outcome: str = "") -> None:
        self.actual_outcome = outcome or self.expected_outcome
        self.progress = 1.0
        self.update_status(TaskStatus.SUCCESS)

    def mark_failed(self, reason: str = "") -> None:
        self.update_status(TaskStatus.FAILED, reason=reason)

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def remaining_retries(self) -> int:
        return max(0, self.max_retries - self.retry_count)

    # ------------------------------------------------------------------
    # Composition and execution
    # ------------------------------------------------------------------
    def add_child(self, task: "Task") -> None:
        require_type(task, Task, "child task")
        task.parent = self
        task.parent_task = self
        self.children.append(task)
        self.last_updated = _now()

    def add_method(self, subtasks: List["Task"]) -> None:
        require_type(subtasks, list, "subtasks")
        for task in subtasks:
            require_type(task, Task, "method subtask")
        self.methods.append([task.copy() for task in subtasks])
        self.last_updated = _now()

    def get_subtasks(self, method_index: Optional[int] = None) -> List["Task"]:
        if self.task_type == TaskType.PRIMITIVE or not self.methods:
            return []
        idx = self.selected_method if method_index is None else int(method_index)
        if 0 <= idx < len(self.methods):
            return [subtask.copy() for subtask in self.methods[idx]]
        logger.warning("Invalid method index %s for task %s", idx, self.name)
        return []

    def check_preconditions(self, world_state: Dict[str, AnyType], *, raise_on_failure: bool = False) -> bool:
        require_type(world_state, dict, "world_state")
        try:
            check_preconditions(world_state, self.preconditions, task_name=self.name, task_id=self.id)
            self.precondition_errors.clear()
            return True
        except PreconditionViolation as exc:
            self.precondition_errors = list(getattr(exc, "failed_conditions", []))
            if raise_on_failure:
                raise
            logger.debug("Precondition failure for %s: %s", self.name, truncate_for_logging(exc))
            return False

    def apply_effects(self, world_state: Dict[str, AnyType], *, raise_on_failure: bool = False) -> Dict[str, AnyType]:
        require_type(world_state, dict, "world_state")
        before = copy.deepcopy(world_state)
        try:
            result = apply_state_effects(world_state, self.effects)
            world_state.clear()
            world_state.update(result)
            self.effect_errors.clear()
            self.actual_outcome = safe_json_dumps(diff_states(before, world_state), fallback_str="effects_applied")
            return world_state
        except Exception as exc:
            self.effect_errors.append(str(exc))
            if raise_on_failure:
                raise PostconditionViolation(
                    f"Effects failed for task {self.name}: {exc}",
                    task_name=self.name,
                    task_id=self.id,
                    expected_state=self.goal_state or {},
                    actual_state=world_state,
                ) from exc
            logger.error("Error applying effects for task %s: %s", self.name, exc)
            return world_state

    # ------------------------------------------------------------------
    # Copying and serialisation
    # ------------------------------------------------------------------
    def copy(self) -> "Task":
        """Return a deep copy of this task, preserving all attributes."""
        return copy.deepcopy(self)

    clone = copy

    def to_dict(
        self,
        *,
        include_methods: bool = True,
        include_children: bool = True,
        include_callables: bool = False,
    ) -> Dict[str, AnyType]:
        data: Dict[str, AnyType] = {}
        for key, value in self.__dict__.items():
            if key in {"parent", "parent_task"}:
                data[key] = getattr(value, "id", None) if value is not None else None
            elif key == "children":
                data[key] = [child.to_dict(include_methods=False, include_children=True) for child in value] if include_children else []
            elif key == "methods":
                data[key] = [
                    [subtask.to_dict(include_methods=False, include_children=True) for subtask in method]
                    for method in value
                ] if include_methods else []
            elif callable(value):
                data[key] = getattr(value, "__name__", "anonymous") if include_callables else None
            elif isinstance(value, list) and value and all(callable(v) for v in value):
                data[key] = [getattr(v, "__name__", "anonymous") for v in value] if include_callables else []
            else:
                data[key] = _serialise_value(value)
        data["task_type"] = self.task_type.name
        data["status"] = self.status.name
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, AnyType]) -> "Task":
        require_type(data, dict, "task data")
        payload = copy.deepcopy(data)
        children_payload = payload.pop("children", []) or []
        methods_payload = payload.pop("methods", []) or []
        payload.pop("parent", None)
        payload.pop("parent_task", None)
        preconditions = payload.pop("preconditions", []) or []
        effects = payload.pop("effects", []) or []
        
        if preconditions or effects:
            logger.debug(
                "Task.from_dict: preconditions/effects cannot be restored from serialized data "
                "(they were stored as %s / %s)", preconditions, effects
            )
        payload["task_type"] = payload.get("task_type", payload.get("type", TaskType.ABSTRACT))
        task = cls(name=payload.pop("name", None), **payload)
        task.preconditions = []
        task.effects = []
        task.children = [cls.from_dict(child) if isinstance(child, dict) else child for child in children_payload]
        for child in task.children:
            if isinstance(child, Task):
                child.parent = task
                child.parent_task = task
        task.methods = [
            [cls.from_dict(sub) if isinstance(sub, dict) else sub for sub in method]
            for method in methods_payload
            if isinstance(method, list)
        ]
        task._post_init()
        return task

    @property
    def requirements(self) -> ResourceProfile:
        return self.resource_requirements

    @property
    def task(self) -> TaskType:
        return self.task_type

    @property
    def age_seconds(self) -> float:
        return max(0.0, _now() - float(self.created_at or self.creation_time or _now()))

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Task) and self.id == other.id

    def __repr__(self) -> str:
        return f"Task(id='{self.id}', name='{self.name}', type={self.task_type.name}, status={self.status.name})"

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Task":
        """Create a deep copy while breaking parent/child cycles."""
        new_task = Task.__new__(Task)
        # Copy all attributes except parent/children (to avoid cycles)
        for k, v in self.__dict__.items():
            if k in ("parent", "parent_task", "children"):
                continue
            new_task.__dict__[k] = copy.deepcopy(v, memo)
        # Reset parent/children for the copy
        new_task.parent = None
        new_task.parent_task = None
        new_task.children = []
        return new_task

__all__ = [
    "Any",
    "WorldState",
    "MethodSignature",
    "TemporalRelation",
    "MemoKey",
    "PlanStep",
    "TaskStatus",
    "TaskType",
    "ResourceProfile",
    "ClusterResources",
    "RepairCandidate",
    "Adjustment",
    "PerformanceMetrics",
    "PlanSnapshot",
    "TemporalConstraints",
    "SafetyViolation",
    "SafetyMargins",
    "Task",
    "to_world_state_tuple",
    "from_world_state_tuple",
]


if __name__ == "__main__":
    print("\n=== Running Planning Types ===\n")
    printer.status("TEST", "Planning Types initialized", "info")

    margins = SafetyMargins.from_config()
    assert margins.max_concurrent > 0

    req = ResourceProfile(gpu=1, ram=2, specialized_hardware=["tensor_core"])
    cluster = ClusterResources(gpu_total=2, ram_total=8, specialized_hardware_available=["tensor_core"])
    assert req.fits_within(cluster)
    cluster.allocate("task_alloc", req)
    assert cluster.available_profile().gpu == 1
    cluster.release("task_alloc")

    state = {"ready": True, "done": False}
    task = Task(
        name="SmokeTask",
        task_type=TaskType.PRIMITIVE,
        resource_requirements=req,
        preconditions=[lambda s: s.get("ready") is True],
        effects=[lambda s: s.update({"done": True})],
        deadline=3600,
    )
    assert task.check_preconditions(state)
    task.apply_effects(state)
    task.mark_success("done")
    assert state["done"] is True and task.status == TaskStatus.SUCCESS

    restored = Task.from_dict(task.to_dict(include_callables=False))
    assert restored.id == task.id and restored.requirements.gpu == 1
    ws = to_world_state_tuple(state)
    assert from_world_state_tuple(ws)["done"] is True

    wrapped = Any(5, (int, lambda x: x > 0))
    assert wrapped.is_compatible(Any(1, (int,)))

    print("\n=== Test ran successfully ===\n")
