
import os
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
class Task:
    name: str = "Planning Typers"
    task_type: TaskType = TaskType.ABSTRACT
    status: TaskStatus = TaskStatus.PENDING
    deadline: float = 0
    priority: int = 1
    resource_requirements: List = field(default_factory=list)
    dependencies: List = field(default_factory=list)
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
    cost: float = 1.0

    def copy(self) -> 'Task':
        return Task(
            name=self.name,
            task_type=self.task_type,
            methods=self.methods,
            preconditions=self.preconditions,
            effects=self.effects,
            cost=self.cost
        )

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

    def __repr__(self):
        method_info = f", exec_mode={self.execution_modes}" if self.execution_modes else ""
        task_type = self.task_type.name if self.task_type else "None"
        status = self.status.name if self.status else "None"
        return f"Task(name='{self.name}', type={task_type}, status={status}{method_info})"

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
