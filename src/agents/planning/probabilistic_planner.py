
import time
import copy

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import defaultdict

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from src.agents.planning.planning_memory import PlanningMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("ProbabilisticPlanner")
printer = PrettyPrinter

# Type definitions consistent with your architecture
StateTuple = Tuple[Tuple[str, Any], ...]
Policy = Dict[StateTuple, 'ProbabilisticAction']

@dataclass
class ProbabilisticAction:
    """
    Defines an action with probabilistic outcomes, central to PPDDL-style planning.
    """
    name: str
    preconditions: Callable[[Dict[str, Any]], bool]
    # Each outcome is a tuple: (probability, effect_function)
    outcomes: List[Tuple[float, Callable[[Dict[str, Any]], None]]]

    def __post_init__(self):
        # Validate that outcome probabilities sum to 1.0
        total_prob = sum(prob for prob, _ in self.outcomes)
        if not abs(total_prob - 1.0) < 1e-6:
            raise ValueError(f"Probabilities for action '{self.name}' do not sum to 1.0 (got {total_prob})")

class ProbabilisticPlanner():
    """
    A planner that handles uncertainty by computing a policy that maximizes the
    probability of reaching a goal state. It uses Value Iteration to solve
    the underlying stochastic shortest-path problem.
    """
    def __init__(self):
        self.config = load_global_config()
        self.pp_config = get_config_section('probabilistic_planner')
        
        # Algorithm parameters
        self.gamma = self.pp_config.get('gamma')  # Discount factor
        self.convergence_threshold = self.pp_config.get('convergence_threshold')
        self.max_iterations = self.pp_config.get('max_iterations')
        
        # Planner state
        self.probabilistic_actions: Dict[str, ProbabilisticAction] = {}
        self.value_function: Dict[StateTuple, float] = defaultdict(float)
        self.policy: Policy = {}
        self.memory = PlanningMemory()

    def register_action(self, action: ProbabilisticAction):
        """Registers a probabilistic action available to the planner."""
        self.probabilistic_actions[action.name] = action

    def perform_task(self, task_data: Dict[str, Any]) -> Optional[Policy]:
        """
        Computes the optimal policy for a given probabilistic planning problem.

        Args:
            task_data: A dictionary containing:
                - 'initial_state': The starting world state dictionary.
                - 'goal_state': A dictionary defining the goal conditions.
                - 'success_threshold': The minimum acceptable probability of success.
        
        Returns:
            An optimal policy mapping states to actions, or None if no policy meets the threshold.
        """
        initial_state = task_data.get('initial_state')
        goal_state = task_data.get('goal_state')

        if not all([initial_state, goal_state]):
            logger.error("Probabilistic planning requires 'initial_state' and 'goal_state'.")
            return None

        self._compute_optimal_policy(initial_state, goal_state)
        
        initial_state_tuple = self._state_to_tuple(initial_state)
        success_probability = self.value_function.get(initial_state_tuple, 0.0)
        
        printer.status(f"Policy computed.", f"Estimated success probability from initial state: {success_probability:.4f}", "info")

        if success_probability >= task_data.get('success_threshold', 0.9):
            return self.policy
        else:
            logger.warning(f"Could not find a policy with success probability >= {task_data.get('success_threshold', 0.9)}")
            return None

    def _compute_optimal_policy(self, initial_state: Dict[str, Any], goal_state: Dict[str, Any]):
        """
        Executes the Value Iteration algorithm to find the optimal value function
        and then extracts the corresponding policy.
        """
        printer.status("PPDDL", "Starting Value Iteration to compute optimal policy.")
        reachable_states = self._get_reachable_states(initial_state, goal_state)

        for i in range(self.max_iterations):
            max_delta = 0.0
            
            for state_tuple in reachable_states:
                if self._is_goal_state(state_tuple, goal_state):
                    self.value_function[state_tuple] = 1.0
                    continue

                old_value = self.value_function[state_tuple]
                
                q_values = {
                    action.name: self._calculate_q_value(state_tuple, action, goal_state)
                    for action in self._get_applicable_actions(state_tuple)
                }
                
                new_value = max(q_values.values()) if q_values else 0.0
                self.value_function[state_tuple] = new_value
                max_delta = max(max_delta, abs(new_value - old_value))
            
            if max_delta < self.convergence_threshold:
                logger.info(f"Value function converged after {i+1} iterations.")
                break
        else:
            logger.warning(f"Value Iteration did not converge after {self.max_iterations} iterations.")

        self._extract_policy(reachable_states, goal_state)

    def _extract_policy(self, states: set, goal_state: Dict[str, Any]):
        """
        Extracts the best policy from the converged value function.
        For each state, it finds the action that leads to the highest expected value.
        """
        printer.status("PPDDL", "Extracting policy from value function.", "info")
        self.policy.clear()
        for state_tuple in states:
            if self._is_goal_state(state_tuple, goal_state):
                continue

            applicable_actions = self._get_applicable_actions(state_tuple)
            if not applicable_actions:
                continue

            best_action = max(
                applicable_actions,
                key=lambda action: self._calculate_q_value(state_tuple, action, goal_state)
            )
            self.policy[state_tuple] = best_action

    def _calculate_q_value(self, state_tuple: StateTuple, action: ProbabilisticAction, goal_state: Dict[str, Any]) -> float:
        """
        Calculates the expected value (Q-value) of taking a specific action in a given state.
        Q(s, a) = Σ [p(s'|s, a) * (R(s, a, s') + γ * V(s'))]
        """
        memo_key = ('q_value', state_tuple, action.name)
        cached = self.memory.base_state.get(memo_key)
        if cached:
            return cached

        expected_value = 0.0
        current_state_dict = dict(state_tuple)

        for prob, effect_func in action.outcomes:
            if prob == 0:
                continue
            
            next_state_dict = copy.deepcopy(current_state_dict)
            effect_func(next_state_dict)
            next_state_tuple = self._state_to_tuple(next_state_dict)
            
            reward = 1.0 if self._is_goal_state(next_state_tuple, goal_state) else 0.0
            
            expected_value += prob * (reward + self.gamma * self.value_function[next_state_tuple])

        self.memory.base_state[memo_key] = expected_value
        return expected_value

    def _get_reachable_states(self, initial_state: Dict[str, Any], goal_state: Dict[str, Any]) -> set:
        """
        Performs a forward search (BFS) to find all states reachable from the initial state.
        """
        printer.status("PPDDL", "Discovering reachable state space...", "info")
        initial_tuple = self._state_to_tuple(initial_state)
        queue = [initial_tuple]
        visited = {initial_tuple}

        while queue:
            current_state_tuple = queue.pop(0)

            if self._is_goal_state(current_state_tuple, goal_state):
                continue

            for action in self._get_applicable_actions(current_state_tuple):
                current_state_dict = dict(current_state_tuple)
                for prob, effect_func in action.outcomes:
                    if prob > 0:
                        next_state_dict = copy.deepcopy(current_state_dict)
                        effect_func(next_state_dict)
                        next_state_tuple = self._state_to_tuple(next_state_dict)
                        if next_state_tuple not in visited:
                            visited.add(next_state_tuple)
                            queue.append(next_state_tuple)
        
        logger.info(f"Discovered {len(visited)} reachable states.")
        return visited

    def _get_applicable_actions(self, state_tuple: StateTuple) -> List[ProbabilisticAction]:
        """Returns a list of actions whose preconditions are met in the given state."""
        state_dict = dict(state_tuple)
        return [
            action for action in self.probabilistic_actions.values()
            if action.preconditions(state_dict)
        ]

    def _is_goal_state(self, state_tuple: StateTuple, goal_state: Dict[str, Any]) -> bool:
        """Checks if a state satisfies the goal conditions."""
        state_dict = dict(state_tuple)
        return all(state_dict.get(key) == value for key, value in goal_state.items())

    def _state_to_tuple(self, state_dict: Dict[str, Any]) -> StateTuple:
        """Converts a mutable state dictionary to an immutable, hashable tuple."""
        return tuple(sorted(state_dict.items()))
