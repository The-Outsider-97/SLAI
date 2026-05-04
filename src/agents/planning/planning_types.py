"""
Planning Types – Core data structures for planning, scheduling, and safety.

Includes:
- `Any`: a typed container with constraints (academic style).
- Enums: TaskStatus, TaskType.
- Dataclasses for resources, constraints, snapshots, etc.
- The main `Task` class with full lifecycle support.
"""

import copy
import os
import time
import yaml, json

from enum import Enum
from dataclasses import dataclass, field
from typing import Any as AnyType, Callable, Dict, List, Optional, Tuple, Union

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from src.agents.planning.utils.planning_errors import (AdjustmentError, ReplanningError, TemporalViolation,
                                                SafetyMarginError, ResourceViolation, AcademicPlanningError)
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Planning Types")
printer = PrettyPrinter


# -------------------------------------------------------------------------
# Any – typed container with constraints
# -------------------------------------------------------------------------
class Any:
    """
    Universal value container with academic rigor, implementing concepts from:
    - Pierce's 'Types and Programming Languages' (type system foundations)
    - Reynolds' 'Polymorphism is Not Set-Theoretic' (parametricity)
    - Wadler's 'Theorems for Free!' (generic operations)

    Provides strict typing while allowing controlled flexibility.
    """
    __slots__ = ('_value', '_type', '_constraints')
    
    def __init__(self, value: object, constraints: tuple = ()):
        """
        Initialize with value and optional academic constraints:
        - Type constraints: (int, float) - Value constraints: (lambda x: x > 0,) - Domain constraints: ('physical', 'temporal')
        """
        self._value = value
        self._type = type(value)
        self._constraints = constraints
        
        # Validate constraints at initialization
        self._validate(constraints)

    @property
    def value(self):
        return self._value

    @property
    def type(self):
        return self._type

    def _validate(self, constraints: tuple):
        """Apply constraint checking using Hoare logic principles."""
        for constraint in constraints:
            if isinstance(constraint, type):
                if not isinstance(self._value, constraint):
                    raise TypeError(
                        f"Value {self._value} violates type constraint {constraint}"
                    )
            elif isinstance(constraint, str):
                # Domain tags are checked externally
                continue
            elif callable(constraint):
                if not constraint(self._value):
                    raise ValueError(
                        f"Value {self._value} violates predicate constraint"
                    )
            else:
                raise AcademicPlanningError(
                    f"Invalid constraint type: {type(constraint)}"
                )

    def is_compatible(self, other: "Any") -> bool:
        """
        Structural compatibility check using Mitchell's subtype theory.
        Returns True if:
        1. Types are compatible (via inheritance)
        2. All constraints of 'other' are satisfied by self
        """
        return issubclass(self._type, other.type) and all(
            c(self.value) for c in other.constraints if callable(c)
        )

    def constrain(self, new_constraints: tuple) -> "Any":
        """Create new Any instance with additional constraints."""
        return Any(self._value, self._constraints + new_constraints)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Any):
            return self.value == other.value
        return self.value == other

    def __repr__(self) -> str:
        return f"Any<{self._type.__name__}>({self._value})"

    def to_json(self) -> dict:
        """Serialization using Pierce's recursive type encoding."""
        return {
            "value": self._value,
            "type": self._type.__name__,
            "constraints": [
                c.__name__ if callable(c) else c for c in self._constraints
            ],
        }

    @classmethod
    def from_json(cls, data: dict) -> "Any":
        """Deserialization with runtime type reconstruction."""
        type_map = {
            t.__name__: t for t in (int, float, str, bool, list, dict)
        }
        value = data["value"]
        constraints = []

        for c in data["constraints"]:
            if c in type_map:
                constraints.append(type_map[c])
            elif c in ("physical", "temporal"):
                constraints.append(c)
            else:
                raise AcademicPlanningError(f"Unreconstructible constraint: {c}")

        return cls(value, tuple(constraints))

    def __add__(self, other: "Any") -> "Any":
        """Additive operation with type conservation rules."""
        if not isinstance(other, Any):
            other = Any(other)

        if self.type != other.type:
            raise AcademicPlanningError("Additive type mismatch")

        return Any(self.value + other.value, self._constraints)

    def __radd__(self, other: object) -> "Any":
        return self.__add__(other)


# -------------------------------------------------------------------------
# Type aliases
# -------------------------------------------------------------------------
WorldState = Tuple[Tuple[str, Any], ...]  # Immutable state representation
MethodSignature = Tuple[str, int]  # (task_name, method_index)
TemporalRelation = Tuple["Task", "Task", str]  # (task_a, task_b, relation type)
MemoKey = Tuple[MethodSignature, WorldState]
PlanStep = Tuple[int, "Task", MethodSignature]  # (step_id, task, method_used)


# -------------------------------------------------------------------------
# Enums
# -------------------------------------------------------------------------
class TaskStatus(Enum):
    """Represents the execution status of a task."""

    PENDING = 0
    EXECUTING = 1
    SUCCESS = 2
    FAILED = 3


class TaskType(Enum):
    """Differentiates between primitive actions and abstract goals."""

    PRIMITIVE = 0
    ABSTRACT = 1


# -------------------------------------------------------------------------
# Dataclasses for resources, constraints, etc.
# -------------------------------------------------------------------------
@dataclass
class ResourceProfile:
    gpu: int = 0
    ram: int = 0  # In GB
    specialized_hardware: List[str] = field(default_factory=list)

    def count_requirements(self) -> int:
        return len(self.__dict__)


@dataclass
class ClusterResources:
    gpu_total: int = 0
    ram_total: int = 0
    specialized_hardware_available: List[str] = field(default_factory=list)
    current_allocations: Dict[str, ResourceProfile] = field(default_factory=dict)


@dataclass
class RepairCandidate:
    """Represents a candidate solution for repairing a failed plan."""

    strategy: str
    repaired_plan: List["Task"]
    estimated_cost: float
    risk_assessment: Dict[str, AnyType]


@dataclass
class Adjustment:
    """Data structure for plan adjustment requests."""

    type: str  # 'modify_task', 'add_task', 'remove_task'
    task_id: Optional[str] = None
    task: Optional["Task"] = None
    updates: Optional[Dict[str, AnyType]] = None
    priority: int = 3
    cascade: bool = False
    origin: str = "api"
    timestamp: float = field(default_factory=time.time)
    _retry_count: int = 0


@dataclass
class PerformanceMetrics:
    """Captures system performance metrics at a point in time."""

    timestamp: float = field(default_factory=time.time)
    system_load: float = 0.0
    network_latency: float = -1.0  # ms, -1 indicates error
    service_health: Dict[str, str] = field(default_factory=dict)
    plan_execution_rate: float = 0.0  # tasks/min


@dataclass
class PlanSnapshot:
    """Snapshot of current plan state."""

    timestamp: float = field(default_factory=time.time)
    task_ids: List[str] = field(default_factory=list)
    resource_utilization: Dict[str, str] = field(default_factory=dict)


@dataclass
class TemporalConstraints:
    """Comprehensive temporal constraint system."""

    start_time: float = 0.0
    end_time: float = 0.0
    min_duration: float = 0.0
    max_duration: float = 0.0
    dependencies: List[str] = field(default_factory=list)  # Task IDs this task depends on
    max_wait: float = 0.0  # Max time to wait for dependencies
    time_buffer: float = 0.0  # Buffer time after completion
    constraints: List[Callable] = field(default_factory=list)  # Custom constraint functions

    def validate(self, current_time: float) -> bool:
        """Check if temporal constraints are satisfied."""
        if self.start_time > 0 and current_time < self.start_time:
            return False
        if self.end_time > 0 and current_time > self.end_time:
            return False
        return all(constraint(current_time) for constraint in self.constraints)


@dataclass
class SafetyViolation:
    """Detailed safety violation report."""

    violation_type: str
    resource: str
    measured_value: float
    threshold: float
    task_id: str
    timestamp: float = field(default_factory=time.time)
    severity: str = "medium"  # low, medium, high, critical
    corrective_action: str = ""
    impact_analysis: Dict[str, AnyType] = field(default_factory=dict)


@dataclass
class SafetyMargins:
    """Configuration for safety buffers."""

    gpu_buffer: float = 0.15  # 15% buffer
    ram_buffer: float = 0.20  # 20% buffer
    min_task_duration: int = 30  # seconds
    max_concurrent: int = 5
    time_buffer: int = 120  # seconds


# -------------------------------------------------------------------------
# Task – the core work unit
# -------------------------------------------------------------------------
class Task:
    """Represents a unit of work in the planning system."""

    _id_counter = 0

    def __init__(
        self,
        name: str = "Planning Task",
        task_type: TaskType = TaskType.ABSTRACT,
        **kwargs,
    ):
        """
        Create a new Task.

        Args:
            name: Human-readable name.
            task_type: PRIMITIVE or ABSTRACT.
            **kwargs: Additional fields (see the class docstring for full list).
        """
        # Generate a unique ID
        Task._id_counter += 1
        self.id = f"task_{int(time.time()*1000)}_{Task._id_counter}"

        # Basic fields
        self.name = name
        self.task_type = task_type
        self.type = task_type  # alias
        self.status = TaskStatus.PENDING
        self.parent = None
        self.parent_task = None
        self.children: List["Task"] = []
        self.methods: List[List["Task"]] = []
        self.selected_method = 0
        self.goal_state: Optional[Dict[str, AnyType]] = None
        self.duration = 300.0
        self.estimated_duration = 0.0
        self.actual_duration = 0.0
        self.cost = 1.0
        self.is_probabilistic = False
        self.probabilistic_actions: List[AnyType] = []
        self.success_threshold = 0.9
        self.risk_score = 0.0
        self.dependencies: List[str] = []
        self.execution_modes: List[str] = ["full"]
        self.description = "No description provided"
        self.created_at = time.time()
        self.last_updated = time.time()
        self.owner = "system"
        self.required_skills: List[str] = []
        self.progress = 0.0
        self.required_tools: List[str] = []
        self.location = "unspecified"
        self.retry_count = 0
        self.max_retries = 3
        self.timeout = 0.0
        self.criticality = "medium"
        self.category = "general"
        self.parameters: Dict[str, AnyType] = {}
        self.preconditions: List[Callable] = []
        self.effects: List[Callable] = []
        self.precondition_errors: List[str] = []
        self.effect_errors: List[str] = []
        self.history: List[Dict[str, AnyType]] = []
        self.context: Dict[str, AnyType] = {}
        self.energy_consumption = 0.0
        self.data_requirements: Dict[str, AnyType] = {}
        self.safety_constraints: List[str] = []
        self.quality_metrics: Dict[str, float] = {}
        self.failure_reason = ""
        self.recovery_strategy = ""
        self.parallelizable = False
        self.human_interaction_required = False
        self.verification_method = "automatic"
        self.documentation = ""
        self.tags: List[str] = []
        self.version = "1.0"
        self.source = "internal"
        self.expected_outcome = ""
        self.actual_outcome = ""
        self.sensor_requirements: List[str] = []
        self.communication_requirements: Dict[str, AnyType] = {}
        self.environmental_constraints: Dict[str, AnyType] = {}
        self.compliance_requirements: List[str] = []
        self.optimization_metrics: List[str] = []
        self.learning_curve = 0.0
        self.example_goal: Optional[Dict[str, AnyType]] = None

        # Resource requirements
        self.resource_requirements = ResourceProfile()

        # Timing fields
        self.start_time = 0.0
        self.end_time = 0.0
        self.deadline = 0.0
        self.priority = 1

        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Ignoring unknown field '{key}' for task '{name}'")

        # Post‑initialization adjustments
        self._post_init()

    def _post_init(self):
        """Perform post‑init adjustments (e.g., time conversions)."""
        current_time = time.time()

        # Convert relative times to absolute times
        if isinstance(self.start_time, (int, float)) and self.start_time < current_time:
            self.start_time = current_time + self.start_time

        if isinstance(self.deadline, (int, float)) and self.deadline < current_time:
            self.deadline = current_time + self.deadline

        # Set derived fields
        if self.estimated_duration == 0.0 and self.duration > 0:
            self.estimated_duration = self.duration
        if self.end_time == 0.0 and self.start_time and self.duration:
            self.end_time = self.start_time + self.duration

    def copy(self) -> "Task":
        """Create a deep copy of this task."""
        # We use copy.deepcopy but also manually recreate children to avoid recursion.
        new_task = Task(
            name=self.name,
            task_type=self.task_type,
        )
        # Copy all fields
        for key, value in self.__dict__.items():
            if key == "children":
                continue  # we'll handle children separately
            setattr(new_task, key, copy.deepcopy(value))

        # Copy children recursively
        new_task.children = [child.copy() for child in self.children]
        return new_task

    def get_subtasks(self, method_index: Optional[int] = None) -> List["Task"]:
        """Return a copy of the subtasks for the given method index."""
        if self.task_type == TaskType.PRIMITIVE or not self.methods:
            return []
        idx = method_index if method_index is not None else self.selected_method
        if 0 <= idx < len(self.methods):
            return [subtask.copy() for subtask in self.methods[idx]]
        logger.warning(f"Invalid method index {idx} for task '{self.name}'")
        return []

    def check_preconditions(self, world_state: Dict[str, AnyType]) -> bool:
        """Check all preconditions against the given world state."""
        try:
            return all(precond(world_state) for precond in self.preconditions)
        except Exception as e:
            logger.error(f"Error checking preconditions for task '{self.name}': {e}")
            return False

    def apply_effects(self, world_state: Dict[str, AnyType]) -> None:
        """Apply all effects to the world state."""
        try:
            for effect in self.effects:
                effect(world_state)
        except Exception as e:
            logger.error(f"Error applying effects for task '{self.name}': {e}")

    @property
    def requirements(self) -> ResourceProfile:
        """Alias for resource_requirements."""
        return self.resource_requirements

    @property
    def task(self) -> TaskType:
        """Alias for task_type."""
        return self.task_type

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Task) and self.id == other.id

    def __repr__(self) -> str:
        return (
            f"Task(id='{self.id}', name='{self.name}', "
            f"type={self.task_type.name}, status={self.status.name})"
        )


# -------------------------------------------------------------------------
# Main test (kept for demonstration)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Planning Types Test ===\n")
    printer.status("Init", "Planning Types initialized", "success")

    planning = Task()
    print(planning)

    print("\n=== Kitchen Planning Simulation ===")

    # Create world state with initial conditions
    world_state = {
        "vegetables_chopped": Any(False, (bool,)),
        "stove_on": Any(False, (bool, lambda x: x in [True, False])),
        "ingredients_available": Any(["tomato", "onion", "garlic"], (list,)),
        "meal_cooked": Any("pending", (str,)),
    }

    # ----- Create Primitive Tasks -----
    def precondition_chop_vegetables(state):
        return state["ingredients_available"].value

    def effect_chop_vegetables(state):
        state["vegetables_chopped"]._value = True

    chop_veggies = Task(
        name="ChopVegetables",
        task_type=TaskType.PRIMITIVE,
        preconditions=[precondition_chop_vegetables],
        effects=[effect_chop_vegetables],
        cost=2.0,
    )

    # ----- Create Abstract Task with Multiple Methods -----
    def precondition_cook_meal(state):
        return state["vegetables_chopped"].value

    def effect_cook_meal(state):
        state["meal_cooked"]._value = "ready"

    # Method 1: Standard cooking flow
    cook_stove = Task(
        name="CookOnStove",
        task_type=TaskType.PRIMITIVE,
        preconditions=[lambda s: s["stove_on"].value],
        effects=[lambda s: s.update({"stove_on": Any(False)})],
        cost=3.0,
    )

    # Method 2: Alternative cooking method
    microwave = Task(
        name="UseMicrowave",
        task_type=TaskType.PRIMITIVE,
        preconditions=[lambda s: len(s["ingredients_available"].value) >= 2],
        cost=2.5,
    )

    cook_meal = Task(
        name="CookMeal",
        task_type=TaskType.ABSTRACT,
        methods=[[chop_veggies, cook_stove], [chop_veggies, microwave]],
        preconditions=[precondition_cook_meal],
        effects=[effect_cook_meal],
        cost=5.0,
    )

    # ----- Test Execution Flow -----
    def print_state():
        print("\nCurrent World State:")
        for k, v in world_state.items():
            print(f"- {k}: {v.value} (Type: {v.type.__name__})")

    print("\n=== Initial State ===")
    print_state()

    # Test constraint validation
    try:
        invalid_task = Task(
            name="InvalidTask", task_type="invalid_type"  # should be TaskType enum
        )
    except Exception as e:
        print(f"\nConstraint Validation Error: {e}")

    # Test task decomposition
    print("\nTesting Meal Preparation:")
    try:
        print(f"Main task: {cook_meal}")
        print("Attempting method 0:")
        subtasks = cook_meal.get_subtasks(method_index=0)
        for i, task in enumerate(subtasks, 1):
            print(f"Step {i}: {task.name}")
            if task.check_preconditions(world_state):
                print("  Preconditions met - executing...")
                task.apply_effects(world_state)
                task.status = TaskStatus.SUCCESS
            else:
                print(f"  Preconditions failed for {task.name}")

        print_state()
    except AcademicPlanningError as e:
        print(f"Planning Error: {e}")
    except Exception as e:
        print(f"General Error: {e}")

    # Test JSON serialization/deserialization
    print("\nTesting Any Serialization:")
    try:
        original = Any(42, (int, lambda x: x > 0))
        json_data = original.to_json()
        reconstructed = Any.from_json(json_data)
        print(f"Original: {original} | Reconstructed: {reconstructed}")
    except AcademicPlanningError as e:
        print(f"Serialization Error: {e}")

    # Test type mismatch error
    print("\nTesting Type Safety:")
    try:
        num = Any(5, (int,))
        text = Any("hello", (str,))
        result = num + text  # Should raise AcademicPlanningError
    except AcademicPlanningError as e:
        print(f"Caught expected type error: {e}")

    print("\n=== Simulation Complete ===")