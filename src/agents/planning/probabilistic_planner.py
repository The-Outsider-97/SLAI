"""
Probabilistic Planner – Value iteration for stochastic shortest‑path problems.

This module defines a probabilistic action and a planner that uses value iteration
to find a policy that maximises the probability of reaching a goal state.
"""

import copy
import threading

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from src.agents.planning.planning_memory import PlanningMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Probabilistic Planner")
printer = PrettyPrinter

# Type definitions
StateTuple = Tuple[Tuple[str, Any], ...]
Policy = Dict[StateTuple, "ProbabilisticAction"]


@dataclass
class ProbabilisticAction:
    """
    Defines an action with probabilistic outcomes, central to PPDDL‑style planning.

    Attributes:
        name: Unique identifier.
        preconditions: Callable that returns True if the action can be executed.
        outcomes: List of (probability, effect_function) pairs.
        cost: Execution cost (not used in value iteration, but may be used later).
        failure_modes: Optional mapping from failure type to probability.
    """

    name: str
    preconditions: Callable[[Dict[str, Any]], bool]
    outcomes: List[Tuple[float, Callable[[Dict[str, Any]], None]]]
    cost: float = 1.0
    failure_modes: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate outcome probabilities sum to 1.0
        total = sum(p for p, _ in self.outcomes)
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Action '{self.name}' outcome probabilities sum to {total} (not 1.0)")

    def to_policy_format(self) -> Dict:
        """Convert to a policy execution format (for external use)."""
        return {
            "action": self.name,
            "outcomes": [
                {"probability": p, "effect": effect.__name__ if hasattr(effect, "__name__") else str(effect)}
                for p, effect in self.outcomes
            ],
        }


class ProbabilisticPlanner:
    """
    Planner that computes a policy maximising the probability of reaching a goal
    using value iteration on a stochastic shortest‑path problem.

    The planner is thread‑safe and caches Q‑values internally.
    """

    def __init__(self) -> None:
        self.config = load_global_config()
        self.pp_config = get_config_section("probabilistic_planner")

        # Algorithm parameters
        self.gamma = self.pp_config.get("gamma", 0.99)  # Discount factor
        self.convergence_threshold = self.pp_config.get("convergence_threshold", 0.0001)
        self.max_iterations = self.pp_config.get("max_iterations", 1000)

        # Planner state
        self._actions: Dict[str, ProbabilisticAction] = {}
        self._value_function: Dict[StateTuple, float] = defaultdict(float)
        self._policy: Policy = {}
        self._q_cache: Dict[Tuple[str, StateTuple], float] = {}  # (action_name, state_tuple) -> Q-value

        self._lock = threading.RLock()
        logger.info("Probabilistic Planner successfully initialized")

    # -------------------------------------------------------------------------
    # Action registration
    # -------------------------------------------------------------------------
    def register_action(self, action: Union[ProbabilisticAction, Dict]) -> None:
        """
        Register a probabilistic action. If a dict is provided, it must contain
        'task_name', 'probability', 'preconditions', and 'effect' keys.
        """
        if isinstance(action, dict):
            success_prob = float(action.get("probability", 1.0))
            success_prob = min(1.0, max(0.0, success_prob))
            action = ProbabilisticAction(
                name=action.get("task_name", "unnamed"),
                preconditions=action.get("preconditions", lambda s: True),
                outcomes=[
                    (success_prob, action.get("effect", lambda s: s)),
                    (1.0 - success_prob, lambda s: s),
                ],
            )
        with self._lock:
            self._actions[action.name] = action
        logger.debug(f"Registered action: {action.name}")

    # -------------------------------------------------------------------------
    # Public planning entry point
    # -------------------------------------------------------------------------
    def perform_task(self, task_data: Dict[str, Any]) -> Optional[Policy]:
        """
        Compute the optimal policy for a given problem.

        Args:
            task_data: Dict containing:
                - "initial_state": The starting world state (dict).
                - "goal_state": Goal condition (dict).
                - "success_threshold": Minimum acceptable success probability (optional, default 0.9).

        Returns:
            Policy (dict mapping StateTuple to ProbabilisticAction) if a policy
            meeting the threshold is found, otherwise None.
        """
        initial_state = task_data.get("initial_state")
        goal_state = task_data.get("goal_state")
        success_threshold = task_data.get("success_threshold", 0.9)

        if initial_state is None or goal_state is None:
            logger.error("Both 'initial_state' and 'goal_state' must be provided")
            return None

        with self._lock:
            self._compute_optimal_policy(initial_state, goal_state)

            initial_tuple = self._state_to_tuple(initial_state)
            success_prob = self._value_function.get(initial_tuple, 0.0)

        printer.status(
            "PPDDL",
            f"Estimated success probability from initial state: {success_prob:.4f}",
            "info",
        )

        if success_prob >= success_threshold:
            return self._policy
        else:
            logger.warning(f"Policy success probability {success_prob:.4f} < threshold {success_threshold}")
            return None

    # -------------------------------------------------------------------------
    # Value iteration
    # -------------------------------------------------------------------------
    def _compute_optimal_policy(self, initial_state: Dict[str, Any], goal_state: Dict[str, Any]) -> None:
        """
        Execute value iteration to compute optimal value function and policy.
        """
        printer.status("PPDDL", "Starting Value Iteration", "info")
        self._q_cache.clear()
        reachable_states = self._get_reachable_states(initial_state, goal_state)

        for iteration in range(self.max_iterations):
            max_delta = 0.0

            for state_tuple in reachable_states:
                if self._is_goal_state(state_tuple, goal_state):
                    self._value_function[state_tuple] = 1.0
                    continue

                old_value = self._value_function[state_tuple]
                q_values = {}
                for action in self._get_applicable_actions(state_tuple):
                    q_val = self._calculate_q_value(state_tuple, action, goal_state)
                    q_values[action.name] = q_val

                new_value = max(q_values.values()) if q_values else 0.0
                self._value_function[state_tuple] = new_value
                max_delta = max(max_delta, abs(new_value - old_value))

            if max_delta < self.convergence_threshold:
                logger.info(f"Value function converged after {iteration + 1} iterations")
                break
            # Q values depend on V(s) and must be recomputed every iteration.
            self._q_cache.clear()
        else:
            logger.warning(f"Value iteration did not converge after {self.max_iterations} iterations")

        self._extract_policy(reachable_states, goal_state)

    def _extract_policy(self, states: set, goal_state: Dict[str, Any]) -> None:
        """
        Extract the optimal policy from the converged value function.
        """
        printer.status("PPDDL", "Extracting policy", "info")
        self._policy.clear()
        for state_tuple in states:
            if self._is_goal_state(state_tuple, goal_state):
                continue

            applicable = self._get_applicable_actions(state_tuple)
            if not applicable:
                continue

            best_action = max(
                applicable,
                key=lambda a: self._calculate_q_value(state_tuple, a, goal_state),
            )
            self._policy[state_tuple] = best_action

    # -------------------------------------------------------------------------
    # Q‑value calculation (with caching)
    # -------------------------------------------------------------------------
    def _calculate_q_value(
        self, state_tuple: StateTuple, action: ProbabilisticAction, goal_state: Dict[str, Any]
    ) -> float:
        """
        Compute Q(s, a) = Σ p(s'|s,a) * (reward(s') + γ * V(s')).
        Cached internally for performance.
        """
        cache_key = (action.name, state_tuple)
        if cache_key in self._q_cache:
            return self._q_cache[cache_key]

        expected = 0.0
        state_dict = dict(state_tuple)

        for prob, effect in action.outcomes:
            if prob == 0.0:
                continue
            next_dict = copy.deepcopy(state_dict)
            effect(next_dict)
            next_tuple = self._state_to_tuple(next_dict)
            reward = 1.0 if self._is_goal_state(next_tuple, goal_state) else 0.0
            expected += prob * (reward + self.gamma * self._value_function[next_tuple])

        self._q_cache[cache_key] = expected
        return expected

    # -------------------------------------------------------------------------
    # State space exploration
    # -------------------------------------------------------------------------
    def _get_reachable_states(self, initial_state: Dict[str, Any], goal_state: Dict[str, Any]) -> set:
        """
        Perform BFS from initial state to discover all reachable states.
        """
        printer.status("PPDDL", "Discovering reachable state space", "info")
        initial_tuple = self._state_to_tuple(initial_state)
        visited = {initial_tuple}
        queue = [initial_tuple]

        while queue:
            current = queue.pop(0)
            if self._is_goal_state(current, goal_state):
                continue

            for action in self._get_applicable_actions(current):
                cur_dict = dict(current)
                for prob, effect in action.outcomes:
                    if prob > 0:
                        next_dict = copy.deepcopy(cur_dict)
                        effect(next_dict)
                        next_tuple = self._state_to_tuple(next_dict)
                        if next_tuple not in visited:
                            visited.add(next_tuple)
                            queue.append(next_tuple)

        logger.info(f"Discovered {len(visited)} reachable states")
        return visited

    def _get_applicable_actions(self, state_tuple: StateTuple) -> List[ProbabilisticAction]:
        """Return actions whose preconditions hold in the given state."""
        state_dict = dict(state_tuple)
        with self._lock:
            return [a for a in self._actions.values() if a.preconditions(state_dict)]

    # -------------------------------------------------------------------------
    # Goal test and state conversion
    # -------------------------------------------------------------------------
    def _is_goal_state(self, state_tuple: StateTuple, goal_state: Dict[str, Any]) -> bool:
        """Check if state satisfies all goal conditions."""
        state_dict = dict(state_tuple)
        return all(state_dict.get(k) == v for k, v in goal_state.items())

    @staticmethod
    def _state_to_tuple(state_dict: Dict[str, Any]) -> StateTuple:
        """Convert a mutable dict to an immutable, hashable tuple."""
        return tuple(sorted(state_dict.items()))


# -------------------------------------------------------------------------
# Test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Probabilistic Planner ===\n")
    printer.status("TEST", "Starting Probabilistic Planner tests", "info")

    planner = ProbabilisticPlanner()

    # Simple world: robot at A or B
    initial_state = {"robot_at": "A"}
    goal_state = {"robot_at": "B"}

    # Action: move from A to B (succeeds 80%, stays at A 20%)
    def pre_move(state):
        return state.get("robot_at") == "A"

    def effect_success(state):
        state["robot_at"] = "B"

    def effect_fail(state):
        state["robot_at"] = "A"

    move_action = ProbabilisticAction(
        name="move_A_to_B",
        preconditions=pre_move,
        outcomes=[(0.8, effect_success), (0.2, effect_fail)],
        cost=1.0,
    )

    planner.register_action(move_action)

    task_data = {
        "initial_state": initial_state,
        "goal_state": goal_state,
        "success_threshold": 0.7,
    }

    policy = planner.perform_task(task_data)

    if policy:
        printer.status("TEST", f"Policy found with {len(policy)} state-action pairs", "success")
        for state, action in policy.items():
            print(f"  {state} -> {action.name}")
    else:
        printer.status("TEST", "No policy meeting threshold found", "warning")

    print("\n=== All tests completed successfully! ===\n")
