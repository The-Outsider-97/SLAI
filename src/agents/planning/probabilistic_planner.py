"""
Probabilistic Planner – production-grade stochastic policy planning.

This module solves finite stochastic reachability / shortest-path planning
problems with bounded value iteration.  It keeps the original public API while
adding strict action validation, bounded state-space expansion, stable state
hashing, diagnostic reports, optional memory integration, and structured
PlanningError handling.
"""

from __future__ import annotations

import copy
import threading
import time

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.planning_errors import *
from .utils.planning_helpers import *
from .planning_memory import PlanningMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Probabilistic Planner")
printer = PrettyPrinter()

StateTuple = Tuple[Tuple[str, Any], ...]
Outcome = Tuple[float, Callable[[Dict[str, Any]], None]]
Policy = Dict[StateTuple, "ProbabilisticAction"]


@dataclass
class ProbabilisticAction:
    """
    Stochastic planning action with validated probabilistic outcomes.

    Outcomes are a list of ``(probability, effect_callable)`` pairs.  Effect
    callables mutate a copied state in place.  Preconditions are evaluated on a
    read-only copy of the current state by the planner.
    """

    name: str
    preconditions: Callable[[Dict[str, Any]], bool]
    outcomes: List[Outcome]
    cost: float = 1.0
    failure_modes: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.name, "action.name")
        if not callable(self.preconditions):
            raise PlanningConfigError(
                f"Action {self.name!r} preconditions must be callable",
                config_key="preconditions",
                config_section="probabilistic_planner",
                expected_type="Callable[[dict], bool]",
            )
        require_type(self.outcomes, list, "action.outcomes")
        require_non_empty(self.outcomes, "action.outcomes")
        require_non_negative(float(self.cost), "action.cost")
        self.cost = float(self.cost)

        total = 0.0
        normalised: List[Outcome] = []
        for idx, (prob, effect) in enumerate(self.outcomes):
            p = float(prob)
            validate_probability(p, f"action.outcomes[{idx}].probability")
            if not callable(effect):
                raise PlanningConfigError(
                    f"Outcome effect for action {self.name!r} must be callable",
                    config_key=f"outcomes[{idx}].effect",
                    config_section="probabilistic_planner",
                    expected_type="Callable[[dict], None]",
                )
            total += p
            normalised.append((p, effect))

        if abs(total - 1.0) > 1e-6:
            raise PlanningConfigError(
                f"Action {self.name!r} outcome probabilities sum to {total:.8f}, expected 1.0",
                config_key="outcomes",
                config_section="probabilistic_planner",
                expected_type="probabilities summing to 1.0",
            )
        self.outcomes = normalised

        if self.failure_modes:
            for key, value in self.failure_modes.items():
                validate_probability(float(value), f"failure_modes.{key}")

    def to_policy_format(self) -> Dict[str, Any]:
        """Return a JSON-safe action summary for policy export."""
        return {
            "action": self.name,
            "cost": self.cost,
            "outcomes": [
                {
                    "probability": p,
                    "effect": getattr(effect, "__name__", str(effect)),
                }
                for p, effect in self.outcomes
            ],
            "failure_modes": dict(self.failure_modes),
            "metadata": dict(self.metadata),
        }


class ProbabilisticPlanner:
    """
    Thread-safe finite stochastic planner using bounded value iteration.

    Compatibility surface retained from the original module:
    - ``ProbabilisticAction`` dataclass.
    - ``register_action(...)`` accepts either a ProbabilisticAction or dict.
    - ``perform_task(task_data)`` returns ``Policy`` when the threshold is met.
    - ``_state_to_tuple(...)`` and other internal names remain available.
    """

    def __init__(self, memory: Optional[PlanningMemory] = None) -> None:
        self.config = load_global_config()
        self.pp_config = get_config_section("probabilistic_planner", config=self.config, default={})
        self._load_config()

        self.memory = memory if memory is not None else (PlanningMemory() if self.memory_enabled else None)
        self._actions: Dict[str, ProbabilisticAction] = {}
        self._value_function: Dict[StateTuple, float] = defaultdict(float)
        self._policy: Policy = {}
        self._q_cache: Dict[Tuple[str, StateTuple], float] = {}
        self._reachable_states: Set[StateTuple] = set()
        self._last_plan_report: Dict[str, Any] = {}
        self._trace: Deque[Dict[str, Any]] = deque(maxlen=self.trace_limit)
        self._lock = threading.RLock()

        logger.info("Probabilistic Planner successfully initialized")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def _load_config(self) -> None:
        defaults = {
            "gamma": 0.99,
            "convergence_threshold": 0.0001,
            "max_iterations": 1000,
            "default_success_threshold": 0.90,
            "max_reachable_states": 10000,
            "max_expansion_steps": 50000,
            "goal_reward": 1.0,
            "dead_end_value": 0.0,
            "action_cost_weight": 0.0,
            "cache_q_values": True,
            "raise_on_invalid_problem": True,
            "memory_enabled": False,
            "trace_limit": 250,
            "policy_tie_breaker": "value_then_cost_then_name",
        }
        cfg = deep_update(defaults, dict(self.pp_config or {}))

        self.gamma = float(cfg["gamma"])
        validate_probability(self.gamma, "probabilistic_planner.gamma")
        self.convergence_threshold = float(cfg["convergence_threshold"])
        require_positive(self.convergence_threshold, "probabilistic_planner.convergence_threshold")
        self.max_iterations = int(cfg["max_iterations"])
        require_positive(self.max_iterations, "probabilistic_planner.max_iterations")
        self.default_success_threshold = float(cfg["default_success_threshold"])
        validate_probability(self.default_success_threshold, "probabilistic_planner.default_success_threshold")
        self.max_reachable_states = int(cfg["max_reachable_states"])
        require_positive(self.max_reachable_states, "probabilistic_planner.max_reachable_states")
        self.max_expansion_steps = int(cfg["max_expansion_steps"])
        require_positive(self.max_expansion_steps, "probabilistic_planner.max_expansion_steps")
        self.goal_reward = float(cfg["goal_reward"])
        validate_probability(self.goal_reward, "probabilistic_planner.goal_reward")
        self.dead_end_value = float(cfg["dead_end_value"])
        validate_probability(self.dead_end_value, "probabilistic_planner.dead_end_value")
        self.action_cost_weight = float(cfg["action_cost_weight"])
        require_non_negative(self.action_cost_weight, "probabilistic_planner.action_cost_weight")
        self.cache_q_values = bool(cfg["cache_q_values"])
        self.raise_on_invalid_problem = bool(cfg["raise_on_invalid_problem"])
        self.memory_enabled = bool(cfg["memory_enabled"])
        self.trace_limit = int(cfg["trace_limit"])
        require_positive(self.trace_limit, "probabilistic_planner.trace_limit")
        self.policy_tie_breaker = str(cfg["policy_tie_breaker"])

    # ------------------------------------------------------------------
    # Action registration
    # ------------------------------------------------------------------
    def register_action(self, action: Union[ProbabilisticAction, Dict[str, Any]]) -> None:
        """Register or replace a stochastic action."""
        action_obj = self._coerce_action(action)
        with self._lock:
            self._actions[action_obj.name] = action_obj
            self._invalidate_solution_locked()
        logger.debug("Registered probabilistic action: %s", action_obj.name)

    def unregister_action(self, action_name: str) -> bool:
        """Remove an action by name. Returns True when one was removed."""
        require_non_empty(action_name, "action_name")
        with self._lock:
            removed = self._actions.pop(action_name, None) is not None
            if removed:
                self._invalidate_solution_locked()
            return removed

    def clear_actions(self) -> None:
        """Remove all registered actions and cached policy state."""
        with self._lock:
            self._actions.clear()
            self._invalidate_solution_locked()

    def _coerce_action(self, action: Union[ProbabilisticAction, Dict[str, Any]]) -> ProbabilisticAction:
        if isinstance(action, ProbabilisticAction):
            return action
        require_type(action, dict, "action")
        success_prob = clamp(float(action.get("probability", 1.0)), 0.0, 1.0)
        success_effect = action.get("effect", lambda state: state)
        failure_effect = action.get("failure_effect", lambda state: state)
        return ProbabilisticAction(
            name=str(action.get("name") or action.get("task_name") or "unnamed"),
            preconditions=action.get("preconditions", lambda state: True),
            outcomes=[(success_prob, success_effect), (1.0 - success_prob, failure_effect)],
            cost=float(action.get("cost", 1.0)),
            failure_modes=dict(action.get("failure_modes", {}) or {}),
            metadata=dict(action.get("metadata", {}) or {}),
        )

    # ------------------------------------------------------------------
    # Public planning entry points
    # ------------------------------------------------------------------
    def perform_task(self, task_data: Dict[str, Any]) -> Optional[Policy]:
        """
        Compute a policy for ``task_data`` and return it if it meets threshold.

        ``task_data`` must provide ``initial_state`` and ``goal_state``. Optional
        ``success_threshold`` overrides the configured default.
        """
        result = self.plan(task_data)
        if result["success_probability"] >= result["success_threshold"]:
            return dict(self._policy)
        return None

    def plan(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute a policy and return a detailed diagnostic report."""
        require_type(task_data, dict, "task_data")
        initial_state = task_data.get("initial_state")
        goal_state = task_data.get("goal_state")
        success_threshold = float(task_data.get("success_threshold", self.default_success_threshold))
        validate_probability(success_threshold, "success_threshold")

        if not isinstance(initial_state, dict) or not isinstance(goal_state, dict):
            message = "Both 'initial_state' and 'goal_state' must be dictionaries"
            if self.raise_on_invalid_problem:
                raise PlanningConfigError(
                    message,
                    config_key="task_data",
                    config_section="probabilistic_planner",
                    expected_type="dict with initial_state and goal_state",
                )
            logger.error(message)
            return self._empty_report(success_threshold, message)

        start = time.time()
        with self._lock:
            self._compute_optimal_policy(initial_state, goal_state)
            initial_tuple = self._state_to_tuple(initial_state)
            success_prob = clamp(self._value_function.get(initial_tuple, 0.0), 0.0, 1.0)
            report = {
                "success_probability": success_prob,
                "success_threshold": success_threshold,
                "threshold_met": success_prob >= success_threshold,
                "policy_size": len(self._policy),
                "reachable_states": len(self._reachable_states),
                "actions": len(self._actions),
                "elapsed_seconds": time.time() - start,
                "initial_state": self._state_to_tuple(initial_state),
                "goal_state": copy.deepcopy(goal_state),
            }
            self._last_plan_report = report
            self._record_trace("plan", report)

        printer.status("PPDDL", f"Estimated success probability: {success_prob:.4f}", "info")
        if self.memory is not None:
            self._checkpoint_plan(report)
        return report

    def get_policy(self) -> Policy:
        """Return a shallow copy of the current policy."""
        with self._lock:
            return dict(self._policy)

    def get_value_function(self) -> Dict[StateTuple, float]:
        """Return a copy of the current value function."""
        with self._lock:
            return dict(self._value_function)

    def get_last_plan_report(self) -> Dict[str, Any]:
        """Return the latest planning diagnostic report."""
        with self._lock:
            return copy.deepcopy(self._last_plan_report)

    def explain_policy(self) -> Dict[str, Any]:
        """Return a JSON-safe summary of the active policy."""
        with self._lock:
            return {
                "policy_size": len(self._policy),
                "states": [
                    {
                        "state": state,
                        "value": clamp(self._value_function.get(state, 0.0), 0.0, 1.0),
                        "action": action.to_policy_format(),
                    }
                    for state, action in self._policy.items()
                ],
            }

    # ------------------------------------------------------------------
    # Value iteration
    # ------------------------------------------------------------------
    def _compute_optimal_policy(self, initial_state: Dict[str, Any], goal_state: Dict[str, Any]) -> None:
        printer.status("PPDDL", "Starting value iteration", "info")
        if not self._actions:
            raise MethodSelectionError(
                "ProbabilisticPlanner cannot plan without registered actions.",
                candidate_methods=[],
            )

        self._q_cache.clear()
        self._reachable_states = self._get_reachable_states(initial_state, goal_state)
        if not self._reachable_states:
            self._policy.clear()
            return

        for state_tuple in self._reachable_states:
            self._value_function.setdefault(state_tuple, self.dead_end_value)

        converged = False
        for iteration in range(1, self.max_iterations + 1):
            max_delta = 0.0
            next_values: Dict[StateTuple, float] = {}

            for state_tuple in self._reachable_states:
                if self._is_goal_state(state_tuple, goal_state):
                    next_values[state_tuple] = self.goal_reward
                    continue

                applicable = self._get_applicable_actions(state_tuple)
                if not applicable:
                    next_values[state_tuple] = self.dead_end_value
                    continue

                q_values = [self._calculate_q_value(state_tuple, action, goal_state) for action in applicable]
                next_values[state_tuple] = clamp(max(q_values), 0.0, 1.0)

            for state_tuple, new_value in next_values.items():
                old_value = self._value_function.get(state_tuple, 0.0)
                max_delta = max(max_delta, abs(new_value - old_value))
                self._value_function[state_tuple] = new_value

            self._q_cache.clear()
            if max_delta < self.convergence_threshold:
                converged = True
                logger.info("Value function converged after %d iterations", iteration)
                break

        if not converged:
            logger.warning("Value iteration did not converge after %d iterations", self.max_iterations)

        self._extract_policy(self._reachable_states, goal_state)

    def _extract_policy(self, states: Set[StateTuple], goal_state: Dict[str, Any]) -> None:
        printer.status("PPDDL", "Extracting policy", "info")
        self._policy.clear()
        for state_tuple in states:
            if self._is_goal_state(state_tuple, goal_state):
                continue
            applicable = self._get_applicable_actions(state_tuple)
            if not applicable:
                continue
            scored = [(self._calculate_q_value(state_tuple, action, goal_state), action) for action in applicable]
            best_value, best_action = self._select_best_action(scored)
            if best_value > self.dead_end_value:
                self._policy[state_tuple] = best_action

    def _select_best_action(self, scored: List[Tuple[float, ProbabilisticAction]]) -> Tuple[float, ProbabilisticAction]:
        if self.policy_tie_breaker == "value_then_cost_then_name":
            return max(scored, key=lambda item: (item[0], -item[1].cost, item[1].name))
        return max(scored, key=lambda item: item[0])

    def _calculate_q_value(self, state_tuple: StateTuple, action: ProbabilisticAction,
                           goal_state: Dict[str, Any]) -> float:
        """Compute bounded reachability Q(s,a)."""
        cache_key = (action.name, state_tuple)
        if self.cache_q_values and cache_key in self._q_cache:
            return self._q_cache[cache_key]

        expected = 0.0
        state_dict = self._tuple_to_state(state_tuple)
        for prob, effect in action.outcomes:
            if prob <= 0.0:
                continue
            next_dict = copy.deepcopy(state_dict)
            try:
                effect(next_dict)
            except Exception as exc:
                logger.error("Action %s effect failed: %s", action.name, exc)
                next_dict = copy.deepcopy(state_dict)
            next_tuple = self._state_to_tuple(next_dict)
            if self._is_goal_state(next_tuple, goal_state):
                continuation = self.goal_reward
            else:
                continuation = self.gamma * self._value_function.get(next_tuple, self.dead_end_value)
            expected += prob * continuation

        if self.action_cost_weight > 0.0:
            expected -= self.action_cost_weight * max(action.cost, 0.0)
        value = clamp(expected, 0.0, 1.0)
        if self.cache_q_values:
            self._q_cache[cache_key] = value
        return value

    # ------------------------------------------------------------------
    # State-space exploration and execution
    # ------------------------------------------------------------------
    def _get_reachable_states(self, initial_state: Dict[str, Any], goal_state: Dict[str, Any]) -> Set[StateTuple]:
        printer.status("PPDDL", "Discovering reachable state space", "info")
        initial_tuple = self._state_to_tuple(initial_state)
        visited: Set[StateTuple] = {initial_tuple}
        queue: Deque[StateTuple] = deque([initial_tuple])
        expansions = 0

        while queue:
            current = queue.popleft()
            if self._is_goal_state(current, goal_state):
                continue
            for action in self._get_applicable_actions(current):
                cur_dict = self._tuple_to_state(current)
                for prob, effect in action.outcomes:
                    if prob <= 0.0:
                        continue
                    next_dict = copy.deepcopy(cur_dict)
                    try:
                        effect(next_dict)
                    except Exception as exc:
                        logger.error("Skipping failed expansion effect for %s: %s", action.name, exc)
                        continue
                    next_tuple = self._state_to_tuple(next_dict)
                    if next_tuple not in visited:
                        visited.add(next_tuple)
                        queue.append(next_tuple)
                        if len(visited) >= self.max_reachable_states:
                            raise PlanningConfigError(
                                "Reachable state limit exceeded",
                                config_key="max_reachable_states",
                                config_section="probabilistic_planner",
                                expected_type="larger finite state bound or tighter action model",
                            )
                expansions += 1
                if expansions >= self.max_expansion_steps:
                    raise PlanningConfigError(
                        "State expansion limit exceeded",
                        config_key="max_expansion_steps",
                        config_section="probabilistic_planner",
                        expected_type="larger expansion bound or tighter action model",
                    )
        logger.info("Discovered %d reachable states", len(visited))
        return visited

    def _get_applicable_actions(self, state_tuple: StateTuple) -> List[ProbabilisticAction]:
        state_dict = self._tuple_to_state(state_tuple)
        actions = list(self._actions.values())
        applicable: List[ProbabilisticAction] = []
        for action in actions:
            try:
                if bool(action.preconditions(copy.deepcopy(state_dict))):
                    applicable.append(action)
            except Exception as exc:
                logger.error("Action %s precondition failed: %s", action.name, exc)
        return applicable

    def simulate_policy(self, initial_state: Dict[str, Any], *, max_steps: int = 100) -> List[Dict[str, Any]]:
        """Deterministically follow the policy by choosing the highest-probability outcome."""
        require_positive(max_steps, "max_steps")
        trajectory: List[Dict[str, Any]] = []
        state = copy.deepcopy(initial_state)
        for step in range(int(max_steps)):
            state_tuple = self._state_to_tuple(state)
            action = self._policy.get(state_tuple)
            if action is None:
                break
            prob, effect = max(action.outcomes, key=lambda item: item[0])
            next_state = copy.deepcopy(state)
            effect(next_state)
            trajectory.append({"step": step, "state": state, "action": action.name, "probability": prob})
            state = next_state
        return trajectory

    # ------------------------------------------------------------------
    # Goal test and stable state conversion
    # ------------------------------------------------------------------
    def _is_goal_state(self, state_tuple: StateTuple, goal_state: Dict[str, Any]) -> bool:
        frozen_goal = {str(k): self._freeze_value(v) for k, v in goal_state.items()}
        return state_satisfies_goal(dict(state_tuple), frozen_goal)

    @staticmethod
    def _state_to_tuple(state_dict: Dict[str, Any]) -> StateTuple:
        require_type(state_dict, dict, "state_dict")
        return tuple(sorted((str(k), ProbabilisticPlanner._freeze_value(v)) for k, v in state_dict.items()))


    @staticmethod
    def _tuple_to_state(state_tuple: StateTuple) -> Dict[str, Any]:
        return {k: ProbabilisticPlanner._thaw_value(v) for k, v in state_tuple}

    @staticmethod
    def _thaw_value(value: Any) -> Any:
        # Frozen dictionaries are represented as tuples of (key, value) pairs.
        if isinstance(value, tuple):
            if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
                return {k: ProbabilisticPlanner._thaw_value(v) for k, v in value}
            return [ProbabilisticPlanner._thaw_value(v) for v in value]
        return value

    @staticmethod
    def _freeze_value(value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(sorted((str(k), ProbabilisticPlanner._freeze_value(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(ProbabilisticPlanner._freeze_value(v) for v in value)
        if isinstance(value, set):
            return tuple(sorted(ProbabilisticPlanner._freeze_value(v) for v in value))
        try:
            hash(value)
            return value
        except TypeError:
            return safe_json_dumps(value)

    # ------------------------------------------------------------------
    # Diagnostics and memory
    # ------------------------------------------------------------------
    def _invalidate_solution_locked(self) -> None:
        self._value_function.clear()
        self._policy.clear()
        self._q_cache.clear()
        self._reachable_states.clear()
        self._last_plan_report.clear()

    def _empty_report(self, threshold: float, reason: str) -> Dict[str, Any]:
        report = {
            "success_probability": 0.0,
            "success_threshold": threshold,
            "threshold_met": False,
            "policy_size": 0,
            "reachable_states": 0,
            "actions": len(self._actions),
            "reason": reason,
        }
        self._last_plan_report = report
        return report

    def _record_trace(self, event: str, payload: Dict[str, Any]) -> None:
        self._trace.append({
            "timestamp": time.time(),
            "event": event,
            "payload": safe_json_loads(safe_json_dumps(payload), default=str(payload)),
        })

    def _checkpoint_plan(self, report: Dict[str, Any]) -> None:
        if self.memory is None:
            return
        try:
            self.memory.save_checkpoint(label="probabilistic_plan", metadata={"report": report})
        except Exception as exc:
            logger.warning("Could not checkpoint probabilistic plan: %s", exc)


if __name__ == "__main__":
    print("\n=== Running Probabilistic Planner ===\n")
    printer.status("TEST", "Probabilistic Planner initialized", "info")

    planner = ProbabilisticPlanner()

    def can_move(state):
        return state.get("robot_at") == "A"

    def move_success(state):
        state["robot_at"] = "B"

    def move_fail(state):
        state["robot_at"] = "A"

    planner.register_action(ProbabilisticAction(
        name="move_A_to_B",
        preconditions=can_move,
        outcomes=[(0.8, move_success), (0.2, move_fail)],
        cost=1.0,
    ))

    report = planner.plan({
        "initial_state": {"robot_at": "A"},
        "goal_state": {"robot_at": "B"},
        "success_threshold": 0.7,
    })
    policy = planner.perform_task({
        "initial_state": {"robot_at": "A"},
        "goal_state": {"robot_at": "B"},
        "success_threshold": 0.7,
    })
    assert report["success_probability"] >= 0.7
    assert policy and len(policy) >= 1
    printer.pretty("Policy", planner.explain_policy(), "success")
    print("\n=== Test ran successfully ===\n")
