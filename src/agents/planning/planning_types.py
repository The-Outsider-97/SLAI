
import os
import time
import yaml, json

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Any

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from src.agents.planning.utils.planning_errors import (AdjustmentError, ReplanningError, TemporalViolation,
                                                SafetyMarginError, ResourceViolation, AcademicPlanningError)
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Planning Types")
printer = PrettyPrinter

class Any:
    """
    Universal value container with academic rigor, implementing concepts from:
    - Pierce's 'Types and Programming Languages' (type system foundations)
    - Reynolds' 'Polymorphism is Not Set-Theoretic' (parametricity)
    - Wadler's 'Theorems for Free!' (generic operations)
    
    Provides strict typing while allowing controlled flexibility
    """
    __slots__ = ('_value', '_type', '_constraints')
    
    def __init__(self, value: object, constraints: tuple = ()):
        """
        Initialize with value and optional academic constraints:
        - Type constraints: (int, float)
        - Value constraints: (lambda x: x > 0,)
        - Domain constraints: ('physical', 'temporal')
        """
        self._value = value
        self._type = type(value)
        self._constraints = constraints
        
        # Validate constraints at initialization
        self._validate(constraints)

    @property
    def value(self):
        """Get the stored value."""
        return self._value

    @property
    def type(self):
        """Get the type of the stored value."""
        return self._type

    def _validate(self, constraints: tuple):
        """Apply constraint checking using Hoare logic principles"""
        for constraint in constraints:
            if isinstance(constraint, type):
                if not isinstance(self._value, constraint):
                    raise TypeError(f"Value {self._value} violates type constraint {constraint}")
            elif isinstance(constraint, str):  # Domain tag
                continue  # Domain checks handled externally
            elif callable(constraint):
                if not constraint(self._value):
                    raise ValueError(f"Value {self._value} violates predicate constraint")
            else:
                raise AcademicPlanningError(f"Invalid constraint type: {type(constraint)}")

    def is_compatible(self, other: 'Any') -> bool:
        """
        Structural compatibility check using Mitchell's subtype theory
        Returns True if:
        1. Types are compatible (via inheritance)
        2. All constraints of 'other' are satisfied by self
        """
        return issubclass(self._type, other.type) and all(
            c(self.value) for c in other.constraints if callable(c))

    def constrain(self, new_constraints: tuple) -> 'Any':
        """
        Create new Any instance with additional constraints
        Follows linear type system principles (Wadler 1990)
        """
        return Any(self._value, self._constraints + new_constraints)

    def __eq__(self, other: object) -> bool:
        """Value equality comparison."""
        if isinstance(other, Any):
            return self.value == other.value
        return self.value == other

    def __repr__(self) -> str:
        return f"Any<{self._type.__name__}>({self._value})"

    def to_json(self) -> dict:
        """Serialization using Pierce's recursive type encoding"""
        return {
            'value': self._value,
            'type': self._type.__name__,
            'constraints': [
                c.__name__ if callable(c) else c 
                for c in self._constraints
            ]
        }

    @classmethod
    def from_json(cls, data: dict) -> 'Any':
        """Deserialization with runtime type reconstruction"""
        type_map = {t.__name__: t for t in (int, float, str, bool, list, dict)}
        value = data['value']
        constraints = []
        
        for c in data['constraints']:
            if c in type_map:
                constraints.append(type_map[c])
            elif c in ('physical', 'temporal'):
                constraints.append(c)
            else:
                raise AcademicPlanningError(f"Unreconstructible constraint: {c}")
        
        return cls(value, tuple(constraints))

    # Academic operator overloading
    def __add__(self, other: 'Any') -> 'Any':
        """Additive operation with type conservation rules"""
        if not isinstance(other, Any):
            other = Any(other)
            
        if self.type != other.type:
            raise AcademicPlanningError("Additive type mismatch")
            
        return Any(self.value + other.value, self._constraints)

    def __radd__(self, other: object) -> 'Any':
        return self.__add__(other)


# --- Type Hint Definitions ---
WorldState = Tuple[Tuple[str, Any], ...]  # Immutable state representation
MethodSignature = Tuple[str, int]  # (task_name, method_index)
TemporalRelation = Tuple['Task', 'Task', str]  # (task_a, task_b, relation type)
MemoKey = Tuple[MethodSignature, WorldState]

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

@dataclass
class ResourceProfile:
    gpu: int = 0
    ram: int = 0  # In GB
    specialized_hardware: List[str] = field(default_factory=list)

    def count_requirements(self):
        return len(self.__dict__)

@dataclass
class ClusterResources:
    gpu_total: int = 0
    ram_total: int = 0
    specialized_hardware_available: List[str] = field(default_factory=list)
    current_allocations: Dict[str, ResourceProfile] = field(default_factory=dict)

@dataclass
class RepairCandidate:
    """Represents a candidate solution for repairing a failed plan"""
    strategy: str
    repaired_plan: List['Task']  # Forward reference
    estimated_cost: float
    risk_assessment: Dict[str, Any]

@dataclass
class Adjustment:
    """Data structure for plan adjustment requests"""
    type: str  # 'modify_task', 'add_task', 'remove_task'
    task_id: Optional[str] = None
    task: Optional['Task'] = None  # Forward reference
    updates: Optional[Dict[str, Any]] = None
    priority: int = 3
    cascade: bool = False
    origin: str = 'api'
    timestamp: float = field(default_factory=time.time)
    _retry_count: int = 0

@dataclass
class PerformanceMetrics:
    """Captures system performance metrics at a point in time"""
    timestamp: float = field(default_factory=time.time)
    system_load: float = 0.0
    network_latency: float = -1.0  # ms, -1 indicates error
    service_health: Dict[str, str] = field(default_factory=dict)
    plan_execution_rate: float = 0.0  # tasks/min

@dataclass
class PlanSnapshot:
    """Snapshot of current plan state"""
    timestamp: float = field(default_factory=time.time)
    task_ids: List[str] = field(default_factory=list)
    resource_utilization: Dict[str, str] = field(default_factory=dict)

@dataclass
class TemporalConstraints:
    """Comprehensive temporal constraint system"""
    start_time: float = 0.0
    end_time: float = 0.0
    min_duration: float = 0.0
    max_duration: float = 0.0
    dependencies: List[str] = field(default_factory=list)  # Task IDs this task depends on
    max_wait: float = 0.0  # Max time to wait for dependencies
    time_buffer: float = 0.0  # Buffer time after completion
    constraints: List[Callable] = field(default_factory=list)  # Custom constraint functions
    
    def validate(self, current_time: float) -> bool:
        """Check if temporal constraints are satisfied"""
        if self.start_time > 0 and current_time < self.start_time:
            return False
        if self.end_time > 0 and current_time > self.end_time:
            return False
        return all(constraint(current_time) for constraint in self.constraints)

@dataclass
class SafetyViolation:
    """Detailed safety violation report"""
    violation_type: str
    resource: str
    measured_value: float
    threshold: float
    task_id: str
    timestamp: float = field(default_factory=time.time)
    severity: str = "medium"  # low, medium, high, critical
    corrective_action: str = ""
    impact_analysis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SafetyMargins:
    """Configuration for safety buffers"""
    gpu_buffer: float = 0.15  # 15% buffer
    ram_buffer: float = 0.20  # 20% buffer
    min_task_duration: int = 30  # seconds
    max_concurrent: int = 5
    time_buffer: int = 120  # seconds

@dataclass
class Task:
    id: str = field(default_factory=lambda: f"task_{int(time.time()*1000)}")
    name: str = "Planning Typers"
    task_description: str = ""
    task_type: TaskType = TaskType.ABSTRACT
    type: TaskType = field(init=False)  # Alias for task_type
    status: TaskStatus = TaskStatus.PENDING
    deadline: float = 0
    priority: int = 1
    resource_requirements: ResourceProfile = field(default_factory=lambda: ResourceProfile())
    execution_modes: List = field(default_factory=list)
    input_data: Any = None
    output_target: str = None
    start_time: float = 0
    end_time: float = 0
    parent_task: Optional[str] = None
    processing_stage: Optional[str] = None
    methods: Optional[List[List['Task']]] = field(default_factory=list)
    preconditions: List[Callable] = field(default_factory=list)
    effects: List[Callable] = field(default_factory=list)
    parent: Optional['Task'] = None
    selected_method: int = 0
    goal_state: Optional[Dict] = field(default=None)
    duration: float = 300.0
    cost: float = 1.0
    id_counter = 0
    is_probabilistic: bool = False
    probabilistic_actions: List[Any] = field(default_factory=list)  # List of ProbabilisticAction objects
    success_threshold: float = 0.9  # Default success probability threshold
    risk_score: float = 0.0
    dependencies: List["Task"] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # String-based dependencies
    execution_modes: List[str] = field(default_factory=lambda: ['full'])  # Modes
    description: str = "No description provided"
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    owner: str = "system"
    required_skills: List[str] = field(default_factory=list)
    progress: float = 0.0  # 0.0 to 1.0
    estimated_duration: float = 0.0
    actual_duration: float = 0.0
    required_tools: List[str] = field(default_factory=list)
    location: str = "unspecified"
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 0.0  # Time after which task is considered failed
    criticality: str = "medium"  # low, medium, high, critical
    category: str = "general"
    parameters: Dict[str, Any] = field(default_factory=dict)
    precondition_errors: List[str] = field(default_factory=list)
    effect_errors: List[str] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)
    children: List['Task'] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    energy_consumption: float = 0.0
    data_requirements: Dict[str, Any] = field(default_factory=dict)
    safety_constraints: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    failure_reason: str = ""
    recovery_strategy: str = ""
    parallelizable: bool = False
    human_interaction_required: bool = False
    verification_method: str = "automatic"
    documentation: str = ""
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    source: str = "internal"
    expected_outcome: str = ""
    actual_outcome: str = ""
    sensor_requirements: List[str] = field(default_factory=list)
    communication_requirements: Dict[str, Any] = field(default_factory=dict)
    environmental_constraints: Dict[str, Any] = field(default_factory=dict)
    compliance_requirements: List[str] = field(default_factory=list)
    optimization_metrics: List[str] = field(default_factory=list)
    learning_curve: float = 0.0  # How much the task improves with repetition
    example_goal: Optional[Dict] = field(default=None)

    def __post_init__(self):
        self.type = self.task_type
        current_time = time.time()
        
        # Convert relative times to absolute times
        if isinstance(self.start_time, (int, float)) and self.start_time < current_time:
            self.start_time = current_time + self.start_time
            
        if isinstance(self.deadline, (int, float)) and self.deadline < current_time:
            self.deadline = current_time + self.deadline
            
        # Set derived fields
        if self.estimated_duration == 0.0 and self.duration > 0:
            self.estimated_duration = self.duration
        if not hasattr(self, 'end_time') and self.start_time and self.duration:
            self.end_time = self.start_time + self.duration

        if not hasattr(self, 'methods'):
            self.methods = []
        # Set estimated_duration to duration if not specified
        if self.estimated_duration == 0.0 and self.duration > 0:
            self.estimated_duration = self.duration

    def copy(self) -> 'Task':
        new_task = Task(
            id=self.id,
            name=self.name,
            task_type=self.task_type,
            methods=self.methods,
            preconditions=self.preconditions,
            effects=self.effects,
            cost=self.cost,
            goal_state=self.goal_state,
            is_probabilistic=self.is_probabilistic,
            probabilistic_actions=self.probabilistic_actions.copy(),
            success_threshold=self.success_threshold,
            description=self.description,
            owner=self.owner,
            required_skills=self.required_skills.copy(),
            required_tools=self.required_tools.copy(),
            location=self.location,
            max_retries=self.max_retries,
            criticality=self.criticality,
            category=self.category,
            parameters=self.parameters.copy(),
            context=self.context.copy(),
            energy_consumption=self.energy_consumption,
            data_requirements=self.data_requirements.copy(),
            safety_constraints=self.safety_constraints.copy(),
            parallelizable=self.parallelizable,
            human_interaction_required=self.human_interaction_required,
            verification_method=self.verification_method,
            tags=self.tags.copy(),
            version=self.version,
            source=self.source,
            expected_outcome=self.expected_outcome,
            sensor_requirements=self.sensor_requirements.copy(),
            communication_requirements=self.communication_requirements.copy(),
            environmental_constraints=self.environmental_constraints.copy(),
            compliance_requirements=self.compliance_requirements.copy(),
            optimization_metrics=self.optimization_metrics.copy(),
            learning_curve=self.learning_curve,
        )
        return new_task
    
    @property
    def requirements(self) -> ResourceProfile:
        return self.resource_requirements

    @property
    def task(self) -> ResourceProfile:
        return self.task_type

    def get_subtasks(self, method_index: Optional[int] = None) -> List['Task']:
        if self.task_type == TaskType.PRIMITIVE or not self.methods:
            return []
        idx_to_use = method_index if method_index is not None else self.selected_method
        if 0 <= idx_to_use < len(self.methods):
            return [subtask.copy() for subtask in self.methods[idx_to_use]]
        else:
            print(f"Warning: Invalid method index {idx_to_use} for task '{self.name}'")
            return []

    def check_preconditions(self, world_state: Dict[str, Any]) -> bool:
        try:
            return all(precond(world_state) for precond in self.preconditions)
        except Exception as e:
            print(f"Error checking preconditions for task '{self.name}': {e}")
            return False

    def apply_effects(self, world_state: Dict[str, Any]) -> None:
        try:
            for effect in self.effects:
                effect(world_state)
        except Exception as e:
            print(f"Error applying effects for task '{self.name}': {e}")

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Task) and self.id == other.id

    def __repr__(self):
        #method_info = f", exec_mode={self.execution_modes}" if self.execution_modes else ""
        #task_type = self.task_type.name if self.task_type else "None"
        #status = self.status.name if self.status else "None"
        return (f"Task(id='{self.id}', name='{self.name}', "
                f"type={self.task_type.name}, status={self.status.name})")

PlanStep = Tuple[int, Task, MethodSignature] # (step_id, task, method_used)

if __name__ == "__main__":
    print("\n=== Running Planning Types Test ===\n")
    printer.status("Init", "Planning Types initialized", "success")

    planning = Task()
    print(planning)

    print("\n=== Kitchen Planning Simulation ===")
    
    # Create world state with initial conditions
    world_state = {
        'vegetables_chopped': Any(False, (bool,)),
        'stove_on': Any(False, (bool, lambda x: x in [True, False])),
        'ingredients_available': Any(['tomato', 'onion', 'garlic'], (list,)),
        'meal_cooked': Any('pending', (str,))
    }

    # ----- Create Primitive Tasks -----
    def precondition_chop_vegetables(state):
        return state['ingredients_available'].value  # At least 1 ingredient available
    
    def effect_chop_vegetables(state):
        state['vegetables_chopped']._value = True
        
    chop_veggies = Task(
        name="ChopVegetables",
        task_type=TaskType.PRIMITIVE,
        preconditions=[precondition_chop_vegetables],
        effects=[effect_chop_vegetables],
        cost=2.0
    )

    # ----- Create Abstract Task with Multiple Methods -----
    def precondition_cook_meal(state):
        return state['vegetables_chopped'].value
    
    def effect_cook_meal(state):
        state['meal_cooked']._value = 'ready'
        
    # Method 1: Standard cooking flow
    cook_stove = Task(
        name="CookOnStove",
        task_type=TaskType.PRIMITIVE,
        preconditions=[lambda s: s['stove_on'].value],
        effects=[lambda s: s.update({'stove_on': Any(False)})],
        cost=3.0
    )
    
    # Method 2: Alternative cooking method
    microwave = Task(
        name="UseMicrowave",
        task_type=TaskType.PRIMITIVE,
        preconditions=[lambda s: len(s['ingredients_available'].value) >= 2],
        cost=2.5
    )

    cook_meal = Task(
        name="CookMeal",
        task_type=TaskType.ABSTRACT,
        methods=[
            [chop_veggies, cook_stove],  # Method 0
            [chop_veggies, microwave]      # Method 1 (intentional typo to test error)
        ],
        preconditions=[precondition_cook_meal],
        effects=[effect_cook_meal],
        cost=5.0
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
            name="InvalidTask",
            task_type="invalid_type",  # Should be TaskType enum
            agent=None
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
                print("Preconditions met - executing...")
                task.apply_effects(world_state)
                task.status = TaskStatus.SUCCESS
            else:
                print(f"Preconditions failed for {task.name}")
                
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
