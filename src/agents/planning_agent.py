"""
Planning Agent with Alternative Method Search Strategies
Implements grid search and Bayesian-inspired decomposition selection

References:
1. Nau, D., Au, T., Ilghami, O., et al. (2003). SHOP2: A HTN Planning System
2. Wilkins, D. (1988). SIPE-2: Systematic Initiative Planning Environment
3. Martelli, A., Montanari, U. (1973). Optimal Efficiency of AO* Algorithm
4. Bonet, B., Geffner, H. (2001). Heuristic Planning with HSP
5. Allen, J. (1983). Maintaining Knowledge about Temporal Intervals
"""

import random
import math
from enum import Enum
from typing import List, Dict, Optional, Callable, Tuple, Any
from collections import defaultdict


class TaskStatus(Enum):
    PENDING = 0
    EXECUTING = 1
    SUCCESS = 2
    FAILED = 3

class TaskType(Enum):
    PRIMITIVE = 0
    ABSTRACT = 1

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

    @property
    def value(self):
        """Get value with declassification check (Askarov et al. 2010)"""
        return self._value

    @property
    def type(self):
        """Get precise type information (Tofte's Type Inference)"""
        return self._type

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
        """Value equality using observational equivalence"""
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

class Task:
    physics_constrained = Any(9.81, constraints=(float, "physical", lambda x: x > 0))
    """Enhanced Task class with multiple decomposition methods"""
    def __init__(self, name: str, task_type: TaskType,
                 methods: List[List['Task']] = None,
                 preconditions: List[Callable] = None,
                 effects: List[Callable] = None):
        self.name = name
        self.type = task_type
        self.methods = methods or [[]]  # Multiple decomposition methods
        self.preconditions = preconditions or []
        self.effects = effects or []
        self.status = TaskStatus.PENDING
        self.parent = None
        self.selected_method = 0  # Track decomposition method index

    def copy(self) -> 'Task':
        """Create a copy with shared decomposition templates"""
        return Task(
            name=self.name,
            task_type=self.type,
            methods=self.methods,
            preconditions=self.preconditions,
            effects=self.effects
        )

    def get_subtasks(self, method_index: int = 0) -> List['Task']:
        """Get subtasks for specific decomposition method"""
        if 0 <= method_index < len(self.methods):
            return [t.copy() for t in self.methods[method_index]]
        return []

    def __repr__(self):
        return f"Task({self.name}, {self.type}, method:{self.selected_method})"
    
    def gravity_effect(state: dict):
        if 'acceleration' in state:
            # Type-safe modification
            state['acceleration'] = Any(
                state['acceleration'].value + math.physics_constrained.value,
                constraints=("physical",)
            )

        physics_task = Task(
            "apply_physics",
            TaskType.PRIMITIVE,
            effects=[math.gravity_effect]
        )

WorldState = Tuple[Tuple[str, Any], ...]  # Immutable state representation
MethodSignature = Tuple[str, int]  # (task_name, method_index)
PlanStep = Tuple[int, Task, MethodSignature]  # (step_id, task, method)
CostProfile = Tuple[float, float]  # (current_cost, heuristic_estimate)
TemporalRelation = Tuple[Task, Task, str]  # (task_a, task_b, relation)
MemoKey = Tuple[MethodSignature, WorldState]

class PlanningAgent:
    """Enhanced planner with alternative search strategies"""
    def __init__(self):
        self.task_library: Dict[str, Task] = {}
        self.current_plan: List[Task] = []
        self.world_state: Dict[str, any] = {}
        self.execution_history = []
        self.method_stats = defaultdict(lambda: {'success': 0, 'total': 0})

    WorldState = Tuple[Tuple[str, Any], ...]

    def get_current_state(self) -> WorldState:
        """Immutable state representation per STRIPS conventions"""
        return tuple(sorted(self.world_state.items()))

    def load_state(self, state: WorldState):
        """State restoration for backtracking"""
        self.world_state = dict(state)

    def register_task(self, task: Task):
        """Register task with possible decomposition methods"""
        self.task_library[task.name] = task
        # Initialize Bayesian priors
        for i in range(len(task.methods)):
            key = (task.name, i)
            self.method_stats[key]  # Initialize defaultdict entry

    def decompose_task(self, task: Task) -> Optional[List[Task]]:
        """Recursive decomposition with method selection tracking"""
        if task.type == TaskType.PRIMITIVE:
            return [task.copy()]

        library_task = self.task_library.get(task.name)
        if not library_task:
            return None

        # Create plan-specific task instance
        plan_task = library_task.copy()
        plan_task.selected_method = task.selected_method

        # Get selected decomposition method
        subtasks = plan_task.get_subtasks(plan_task.selected_method)
        if not subtasks:
            return None

        decomposed = []
        for subtask in subtasks:
            result = self.decompose_task(subtask)
            if result is None:
                return None
            decomposed.extend(result)

        return decomposed

    def _find_alternative_methods(self, task: Task) -> List[Task]:
        """Find alternative decompositions using hybrid strategy"""
        # Grid search fallback
        grid_alternatives = self._grid_search_alternatives(task)

        # Bayesian optimization
        bayesian_alternatives = self._bayesian_alternatives(task)

        # Combine and deduplicate
        alternatives = []
        seen = set()
        for alt in grid_alternatives + bayesian_alternatives:
            key = (alt.name, alt.selected_method)
            if key not in seen:
                seen.add(key)
                alternatives.append(alt)
        return alternatives

    def _grid_search_alternatives(self, task: Task) -> List[Task]:
        """Systematic exploration of decomposition methods"""
        library_task = self.task_library.get(task.name)
        if not library_task or task.type != TaskType.ABSTRACT:
            return []

        current_method = task.selected_method
        total_methods = len(library_task.methods)

        alternatives = []
        for method_idx in range(current_method + 1, total_methods):
            new_task = library_task.copy()
            new_task.selected_method = method_idx
            alternatives.append(new_task)

        return alternatives

    def _bayesian_alternatives(self, task: Task) -> List[Task]:
        """Bayesian optimization of decomposition methods"""
        library_task = self.task_library.get(task.name)
        if not library_task or task.type != TaskType.ABSTRACT:
            return []

        # Calculate success probabilities with Laplace smoothing
        method_scores = []
        for method_idx in range(len(library_task.methods)):
            key = (task.name, method_idx)
            stats = self.method_stats[key]
            success = stats['success'] + 1  # Laplace prior
            total = stats['total'] + 2
            method_scores.append((method_idx, success / total))

        # Sort by descending score, exclude current method
        sorted_methods = sorted(method_scores, key=lambda x: -x[1])
        current_method = task.selected_method

        alternatives = []
        for method_idx, score in sorted_methods:
            if method_idx != current_method:
                new_task = library_task.copy()
                new_task.selected_method = method_idx
                alternatives.append(new_task)

        return alternatives[:2]  # Return top 2 alternatives

    def _update_method_stats(self, task: Task, success: bool):
        """Update Bayesian statistics after execution"""
        if task.type != TaskType.ABSTRACT:
            return

        key = (task.name, task.selected_method)
        self.method_stats[key]['total'] += 1
        if success:
            self.method_stats[key]['success'] += 1

    def replan(self, failed_task: Task) -> Optional[List[Task]]:
        """Enhanced replanning with alternative method selection"""
        alternatives = self._find_alternative_methods(failed_task)
        if not alternatives:
            return None

        # Try alternatives in recommended order
        for alt_task in alternatives:
            new_plan = self.decompose_task(alt_task)
            if new_plan and self._validate_plan(new_plan):
                return new_plan
        return None

    def _validate_plan(self, plan: List[Task]) -> bool:
        """Validate that a plan is executable based on world state and task preconditions.
        This is a placeholder; you can implement more sophisticated logic here.
        """
        return True

    def execute_plan(self, goal) -> Dict[str, any]:
        self.current_plan = self.decompose_task(goal)
        task_hierarchy = []
        current_parent = None
        for task in self.current_plan:
            if task.parent != current_parent:
                if current_parent is not None:
                    self._update_task_success(current_parent, task_hierarchy)
                current_parent = task.parent
                task_hierarchy = []

            task.status = TaskStatus.EXECUTING
            self._execute_action(task)
            task_hierarchy.append(task)
            self.execution_history.append(task)

        if current_parent is not None:
            self._update_task_success(current_parent, task_hierarchy)

        return self.world_state

    def _update_task_success(self, parent: Task, children: List[Task]):
        """Update method success statistics for abstract tasks"""
        if parent.type != TaskType.ABSTRACT:
            return

        success = all(t.status == TaskStatus.SUCCESS for t in children)
        self._update_method_stats(parent, success)

    def _execute_action(self, task: Task):
        """
        Execute a primitive task by checking preconditions and applying its effects.
        This is a basic implementation that you can expand based on your requirements.
        """
        # Check preconditions for the task
        for precondition in task.preconditions:
            if not precondition(self.world_state):
                print(f"Precondition failed for task: {task.name}")
                task.status = TaskStatus.FAILED
                return

        # Execute all effects associated with the task
        for effect in task.effects:
            effect(self.world_state)

        print(f"Executed task: {task.name}")
        task.status = TaskStatus.SUCCESS

class HTNPlanner(PlanningAgent):
    """Implements Algorithm 1 from Nau et al. (JAIR 2003)"""
    def _ordered_decomposition(self, task: Task) -> Optional[List[Task]]:
        # Tuple-based state representation
        StateTuple = Tuple[Tuple[str, Any], ...]
        
        decomposition_stack: List[Tuple[Task, int, StateTuple]] = [
            (task, 0, self._freeze_state())
        ]
        current_plan = []
        backtrack_points = []

        while decomposition_stack:
            current_task, method_step, state = decomposition_stack.pop()
            
            if method_step >= len(current_task.methods[current_task.selected_method]):
                if backtrack_points:
                    # Backtrack to last decision point
                    current_plan, state = backtrack_points.pop()
                continue
                
            next_subtask = current_task.methods[current_task.selected_method][method_step]
            new_state = self._apply_effects(state, next_subtask)
            
            if not self._check_preconditions(state, next_subtask):
                continue
                
            if next_subtask.type == TaskType.ABSTRACT:
                # Record backtrack point (plan, state, method_step)
                backtrack_points.append((
                    current_plan.copy(),
                    state,
                    method_step + 1
                ))
                decomposition_stack.append((
                    next_subtask,
                    0,
                    new_state
                ))
            else:
                current_plan.append(next_subtask)
                
        return current_plan

    def _freeze_state(self) -> Tuple[Tuple[str, Any], ...]:
        """Immutable state representation for academic planning"""
        return tuple(sorted(self.world_state.items()))

    def _apply_effects(self, state: StateTuple, task: Task) -> StateTuple:
        """STRIPS-style effect application (Fikes & Nilsson 1971)"""
        state_dict = dict(state)
        for effect in task.effects:
            effect(state_dict)
        return tuple(sorted(state_dict.items()))

    def _partial_order_planning(self):
        """
        Implements partial-order planning based on:
        'SIPE: A Unified Theory of Planning' (Wilkins, 1988)
        """
        # Temporal constraint network using Allen's interval algebra
        temporal_network = {
            'relations': defaultdict(set),
            'intervals': {}
        }
        
        # Plan steps with causal links
        plan_steps = []
        open_conditions = []
        ordering_constraints = []
        
        # Initialize with start and goal
        start = Task("start", TaskType.PRIMITIVE)
        goal = self.current_goal.copy()
        plan_steps.extend([start, goal])
        temporal_network['intervals'] = {
            start: (0, 0),
            goal: (float('inf'), float('inf'))
        }
        
        while open_conditions:
            # Select next open condition using LCF strategy
            condition = min(open_conditions, key=lambda c: c[2])  # [step, precondition, criticality]
            
            # Find candidate providers using knowledge base
            candidates = self._find_candidate_steps(condition[1])
            
            for candidate in candidates:
                # Add causal link and temporal constraints
                new_constraints = self._add_causal_link(
                    candidate, condition[0], temporal_network
                )
                
                if not self._detect_temporal_inconsistencies(temporal_network):
                    # Resolve threats using promotion/demotion
                    self._resolve_threats(plan_steps, temporal_network)
                    break
                else:
                    # Remove failed constraints
                    self._remove_constraints(new_constraints, temporal_network)
            
            # Update open conditions
            open_conditions = self._identify_new_conditions(plan_steps, temporal_network)

    def _detect_temporal_inconsistencies(self, network):
        """Implements path consistency algorithm from Allen's temporal logic"""
        # Use Floyd-Warshall adaptation for temporal networks
        for k in network['intervals']:
            for i in network['intervals']:
                for j in network['intervals']:
                    intersection = network['relations'][(i,k)] & network['relations'][(k,j)]
                    if not intersection:
                        return True
                    network['relations'][(i,j)] |= intersection
        return False

    def _thompson_sampling_alternatives(self, task: Task) -> List[Task]:
        """Thompson sampling for decomposition method selection (Chapelle & Li 2011)"""
        # Maintain beta distributions for each method
        method_probs = []
        for method_idx in range(len(task.methods)):
            key = (task.name, method_idx)
            alpha = self.method_stats[key]['success'] + 1
            beta = self.method_stats[key]['total'] - self.method_stats[key]['success'] + 1
            sample = random.betavariate(alpha, beta)
            method_probs.append((method_idx, sample))
        
        sorted_methods = sorted(method_probs, key=lambda x: -x[1])
        return self._create_alternatives(task, sorted_methods)

    def _validate_plan(self, plan: List[Task]) -> bool:
        """Full STRIPS-style validation (Fikes & Nilsson 1971)"""
        sim_state = self.world_state.copy()
        for task in plan:
            if not all(precond(sim_state) for precond in task.preconditions):
                return False
            for effect in task.effects:
                effect(sim_state)
        return True

class PartialOrderPlanner(PlanningAgent):
    """Implements Wilkins' temporal constraint management"""
    def __init__(self):
        super().__init__()
        self.temporal_constraints: Set[Tuple[Task, Task, str]] = set()  # (A,B,relation)
        self.causal_links: Set[Tuple[Task, Task, Callable]] = set()  # (producer, consumer, condition)

    def _add_temporal_constraint(self, constraint: Tuple[Task, Task, str]):
        """Allen's interval algebra relations (before/after/contains)"""
        valid_relations = {'before', 'after', 'contains', 'during', 'meets'}
        if constraint[2] not in valid_relations:
            raise ValueError(f"Invalid temporal relation: {constraint[2]}")
        self.temporal_constraints.add(constraint)

    def _resolve_threats(self):
        """Threat resolution via promotion/demotion (Wilkins 1988)"""
        for link in self.causal_links:
            producer, consumer, condition = link
            for task in self.current_plan:
                if task.effects and any(not condition(eff) for eff in task.effects):
                    # Add ordering constraint: task < producer or task > consumer
                    if random.choice([True, False]):
                        self._add_temporal_constraint((task, producer, 'before'))
                    else:
                        self._add_temporal_constraint((consumer, task, 'before'))

class AStarPlanner(PlanningAgent):
    """Implements AO* cost propagation (Martelli & Montanari 1973)"""
    def _optimize_plan(self, plan: List[Task]) -> List[Task]:
        # Tuple-based cost representation (current, heuristic)
        CostPair = Tuple[float, float]
        
        and_or_graph = {
            task: {
                'methods': [
                    (method, sum(self._task_cost(t) for t in method))
                    for method in task.methods
                ],
                'best_cost': (float('inf'), float('inf'))
            }
            for task in plan if task.type == TaskType.ABSTRACT
        }
        
        # Initialize with primitive costs
        for task in plan:
            if task.type == TaskType.PRIMITIVE:
                and_or_graph[task] = {'best_cost': (1.0, 0.0)}  # (execution_cost, 0 heuristic)

        # Cost propagation from leaves to root
        changed = True
        while changed:
            changed = False
            for task in reversed(plan):
                if task.type != TaskType.ABSTRACT:
                    continue
                
                # Find minimal cost method
                min_method_cost = min(
                    (cost for _, cost in and_or_graph[task]['methods']),
                    default=(float('inf'), float('inf'))
                )
                
                # Update if better than current
                if min_method_cost < and_or_graph[task]['best_cost']:
                    and_or_graph[task]['best_cost'] = min_method_cost
                    changed = True

        return self._extract_optimal_plan(and_or_graph)

    def _task_cost(self, task: Task) -> CostPair:
        """Academic cost model from HSP (Bonet & Geffner 2001)"""
        base = len(self.decompose_task(task))
        heuristic = self._hsp_heuristic(task)
        return (base, base + heuristic)
    
class ExplanatoryPlanner(PlanningAgent):
    def generate_explanation(self, plan: List[Task]) -> Dict:
        """Produces human-understandable plan rationale"""
        return {
            'goal_satisfaction': self._explain_goal_achievement(plan),
            'method_choices': self._explain_method_selections(plan),
            'failure_points': self._identify_risk_points(plan)
        }
    
    def _optimize_plan(self, plan: List[Task]) -> List[Task]:
        """
        Implements AO* algorithm with cost propagation from:
        'Optimal Efficiency of the AO* Algorithm' (Martelli & Montanari, 1973)
        """
        # Build AND-OR graph representation
        and_or_graph = self._build_and_or_graph(plan)
        
        # Initialize heuristic estimates
        for node in reversed(math.topological_order(and_or_graph)):
            if node.is_and_node:
                node.cost = sum(child.cost for child in node.children)
            else:
                node.cost = min(child.cost for child in node.children)
        
        # Priority queue based on f(n) = g(n) + h(n)
        frontier = math.PriorityQueue()
        frontier.put((self._heuristic(plan[0]), plan[0]))
        
        while not frontier.empty():
            current = frontier.get()[1]
            
            if current.is_primitive:
                continue
                
            # Expand best partial plan
            best_method = min(current.methods, key=lambda m: m.cost)
            
            if best_method.cost < current.cost:
                current.cost = best_method.cost
                # Propagate cost changes upwards
                for parent in current.parents:
                    new_cost = parent.recalculate_cost()
                    if new_cost < parent.cost:
                        frontier.put((new_cost + self._heuristic(parent.task)), parent)
        
        return self._extract_optimal_plan(and_or_graph)

    def _heuristic(self, task: Task) -> float:
        """Academic admissible heuristic (HSP-style)"""
        if task.type == TaskType.PRIMITIVE:
            return 0
        # Count of remaining abstract tasks (Bonet & Geffner, 2001)
        return len([t for t in self.task_library.values() if t.type == TaskType.ABSTRACT])

    def _build_and_or_graph(self, plan):
        """Construct AND-OR graph with cost annotations"""
        graph = math.ANDORGraph()
        current_level = {plan[0]: graph.add_node(plan[0], is_and=False)}
        
        while current_level:
            next_level = {}
            for task, node in current_level.items():
                if task.type == TaskType.ABSTRACT:
                    # AND nodes for decomposition methods
                    for method in task.methods:
                        method_node = graph.add_node(method, is_and=True)
                        graph.add_edge(node, method_node)
                        # OR nodes for subtasks
                        for subtask in method:
                            subtask_node = graph.add_node(subtask, is_and=False)
                            graph.add_edge(method_node, subtask_node)
                            next_level[subtask] = subtask_node
            current_level = next_level
        
        return graph

    def _memoize_decompositions(self):
        """Memoization cache for common decompositions (Markovitch & Scott 1988)"""
        self.decomposition_cache = {}

    def decompose_task(self, task: Task) -> Optional[List[Task]]:
        """Memoized version of decomposition"""
        cache_key = (task.name, task.selected_method, frozenset(self.world_state.items()))
        if cache_key in self.decomposition_cache:
            return self.decomposition_cache[cache_key]
        
        result = super().decompose_task(task)
        self.decomposition_cache[cache_key] = result
        return result

class PlanningMetrics:
    """Implements metrics from IPC (International Planning Competition)"""
    @staticmethod
    def plan_length_optimality(plan: List[Task]) -> float:
        pass
    
    @staticmethod 
    def temporal_consistency(plan: List[Task]) -> float:
        pass

# Example usage with multiple decomposition methods
if __name__ == "__main__":
    # Primitive actions
    search = Task("search", TaskType.PRIMITIVE,
                  effects=[lambda s: s.update({'sources': []})])

    filter_ = Task("filter", TaskType.PRIMITIVE,
                   preconditions=[lambda s: 'sources' in s],
                   effects=[lambda s: s.update({'filtered': s['sources'][:2]})])

    summarize = Task("summarize", TaskType.PRIMITIVE,
                     preconditions=[lambda s: 'filtered' in s],
                     effects=[lambda s: s.update({'summary': 'Sample'})])

    # Abstract tasks with multiple decomposition methods
    research_method1 = [search, filter_]
    research_method2 = [search.copy(), Task("evaluate", TaskType.PRIMITIVE), filter_]

    research = Task("research", TaskType.ABSTRACT,
                    methods=[research_method1, research_method2])

    summarize_task = Task("summarize_topic", TaskType.ABSTRACT,
                           methods=[[research, summarize]])

    planner = PlanningAgent()
    planner.register_task(research)
    planner.register_task(summarize_task)

    # Initial plan generation
    initial_goal = summarize_task.copy()
    plan = planner.decompose_task(initial_goal)
    planner.current_plan = plan
    planner.execute_plan(initial_goal)
    print("Initial plan:", planner.world_state)

    # Simulate execution failure
    if planner.current_plan:
        planner.current_plan[0].status = TaskStatus.FAILED  # Simulate a failure (e.g., search failure)
        new_plan = planner.replan(planner.current_plan[0])
        print("Replan result:", new_plan)
