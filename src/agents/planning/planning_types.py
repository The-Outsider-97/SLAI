import math
from enum import Enum
from typing import List, Dict, Optional, Callable, Tuple, Any, Set

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

class AcademicPlanningError(Exception):
    """Custom exception for type violations and planning semantics."""
    pass

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


class Task:
    """
    Represents a task in the planning hierarchy, either primitive or abstract.
    Includes decomposition methods, preconditions, and effects.
    """
    def __init__(self, name: str, task_type: TaskType,
                 methods: Optional[List[List['Task']]] = None,
                 preconditions: Optional[List[Callable[[Dict[str, Any]], bool]]] = None,
                 effects: Optional[List[Callable[[Dict[str, Any]], None]]] = None,
                 cost: float = 1.0,
                 start_time: Optional[float] = None,
                 end_time: Optional[float] = None):
        """
        Initializes a Task instance.

        Args:
            name (str): The unique name identifier for the task.
            task_type (TaskType): Whether the task is PRIMITIVE or ABSTRACT.
            methods (Optional[List[List['Task']]]): For ABSTRACT tasks, a list of
                possible decomposition methods. Each method is a list of subtasks.
            preconditions (Optional[List[Callable]]): Functions that check if the
                task can be executed given the current world state. Each callable
                takes the world state dictionary and returns True/False.
            effects (Optional[List[Callable]]): Functions that modify the world state
                upon successful execution of the task. Each callable takes the
                world state dictionary and modifies it in place.
            cost (float): The cost associated with executing this task. Default is 1.0.
            start_time (Optional[float]): The execution start time (for metrics).
            end_time (Optional[float]): The execution end time (for metrics).
        """
        self.name = name
        self.type = task_type
        # Ensure methods is a list of lists, even if empty or None
        self.methods = methods if methods is not None else ([[]] if task_type == TaskType.ABSTRACT else [])
        self.preconditions = preconditions or []
        self.effects = effects or []
        self.status = TaskStatus.PENDING
        self.parent: Optional[Task] = None # Track parent task in hierarchy if needed
        self.selected_method: int = 0 # Index of the method used for decomposition
        self.cost = cost
        self.start_time = start_time
        self.end_time = end_time


    def copy(self) -> 'Task':
        """
        Creates a shallow copy of the task, primarily for use in plan generation
        where the core definition (methods, preconditions, effects) is shared,
        but runtime state (status, parent, selected_method) might differ.
        """
        new_task = Task(
            name=self.name,
            task_type=self.type,
            methods=self.methods, # Shallow copy of methods list is usually sufficient
            preconditions=self.preconditions, # Shallow copy of functions
            effects=self.effects, # Shallow copy of functions
            cost=self.cost
            # Runtime state (status, parent, selected_method, times) are NOT copied,
            # they get set during planning/execution.
        )
        return new_task

    def get_subtasks(self, method_index: Optional[int] = None) -> List['Task']:
        """
        Gets the list of subtasks for a specific decomposition method.
        If method_index is None, uses the task's `selected_method`.

        Args:
            method_index (Optional[int]): The index of the decomposition method to use.

        Returns:
            List[Task]: A list of copied subtasks for the specified method.
                        Returns an empty list if the index is invalid, the task
                        is primitive, or has no methods.
        """
        if self.type == TaskType.PRIMITIVE or not self.methods:
            return []

        idx_to_use = method_index if method_index is not None else self.selected_method

        if 0 <= idx_to_use < len(self.methods):
            # Return copies of the subtasks from the template
            return [subtask.copy() for subtask in self.methods[idx_to_use]]
        else:
            print(f"Warning: Invalid method index {idx_to_use} for task '{self.name}'")
            return []

    def check_preconditions(self, world_state: Dict[str, Any]) -> bool:
        """Checks if all preconditions are met in the given world state."""
        try:
            return all(precond(world_state) for precond in self.preconditions)
        except Exception as e:
            print(f"Error checking preconditions for task '{self.name}': {e}")
            return False

    def apply_effects(self, world_state: Dict[str, Any]) -> None:
        """Applies all effects to the given world state."""
        try:
            for effect in self.effects:
                effect(world_state)
        except Exception as e:
            print(f"Error applying effects for task '{self.name}': {e}")


    def __repr__(self):
        method_info = f", method:{self.selected_method}" if self.type == TaskType.ABSTRACT else ""
        return f"Task(name='{self.name}', type={self.type.name}, status={self.status.name}{method_info})"

PlanStep = Tuple[int, Task, MethodSignature] # (step_id, task, method_used)
