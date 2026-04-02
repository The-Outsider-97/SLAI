import math
import time
import random

from typing import Dict, List, Any, Callable, Optional, Tuple

from .utils.config_loader import load_global_config, get_config_section
from .utils.execution_error import InvalidContextError, ActionFailureError
from src.agents.execution.execution_memory import ExecutionMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Action Selector")
printer = PrettyPrinter

class ActionSelector:
    """
    Production‑ready action selector that chooses the best action from a list
    based on configurable strategies: priority, random, contextual, utility, hybrid.
    """
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        self.context = context or {}
        self.config = load_global_config()
        self.selector_config = get_config_section("action_selector")

        self.selection_strategy = self.selector_config.get("strategy", "hybrid")
        self.strategy_weights = self.selector_config.get("strategy_weights", {})
        self.fallback_action = self.selector_config.get("fallback_action", "idle")

        # Load action configurations (for cost/duration estimation)
        self.move_config = get_config_section("move_to_action") or {}
        self.pick_config = get_config_section("pick_object_action") or {}
        self.place_config = get_config_section("place_object_action") or {}
        self.idle_config = get_config_section("idle_action") or {}

        # Registry of available action types with their pre/postconditions
        self.action_registry: Dict[str, Dict[str, List[str]]] = {}

        # Utility function registry
        self.utility_functions: Dict[str, Callable] = {}

        # ExecutionMemory for caching and checkpointing
        self.memory = ExecutionMemory()

        # Selection history (for analysis)
        self.selection_history: List[Dict] = []
        self.max_history = self.config.get("max_history", 1000)

        # Register default utility functions
        self._register_default_utility_functions()

        # Try to restore previous selector state
        self._restore_state()

        logger.info(f"ActionSelector initialized with strategy '{self.selection_strategy}'")

    # ------------------------ Public API ------------------------------
    def select(self, actions: List[Dict], context: Optional[Dict] = None) -> Dict:
        """
        Select the best action from a list based on current context.
        Records selection in history and caches utility scores.
        """
        if context is None:
            context = self.context or {"energy": 10.0}
            logger.debug("Using default/empty context for selection")

        if context.get("force_idle"):
            return self._create_fallback_action(context)

        if not actions:
            logger.warning("No actions provided; returning fallback idle")
            return self._create_fallback_action(context)

        # Filter actions that satisfy preconditions
        valid_actions = self._filter_valid_actions(actions, context)
        if not valid_actions:
            logger.warning("No valid actions after precondition filtering")
            return self._create_fallback_action(context)

        # Apply selection strategy
        strategy = self.selection_strategy
        if strategy == "priority":
            selected = self._priority_selection(valid_actions)
        elif strategy == "random":
            selected = self._random_selection(valid_actions)
        elif strategy == "contextual":
            selected = self._contextual_selection(valid_actions, context)
        elif strategy == "utility":
            selected = self._utility_based_selection(valid_actions, context)
        elif strategy == "hybrid":
            selected = self._hybrid_selection(valid_actions, context)
        else:
            logger.warning(f"Unknown strategy '{strategy}', falling back to priority")
            selected = self._priority_selection(valid_actions)

        # Record selection
        self._record_selection(selected, context)

        logger.info(f"Selected action: {selected.get('name')} (priority {selected.get('priority')})")
        return selected

    def register_action(self, name: str, preconditions: List[str], postconditions: List[str]):
        """Register a new action type for precondition/postcondition validation."""
        self.action_registry[name] = {
            "preconditions": preconditions,
            "postconditions": postconditions
        }
        logger.debug(f"Registered action '{name}'")

    def register_utility_function(self, name: str, func: Callable[[Dict, Dict], float]):
        """Register a custom utility function (takes action, context, returns float)."""
        self.utility_functions[name] = func
        logger.debug(f"Registered utility function '{name}'")

    def set_strategy(self, strategy: str):
        """Dynamically change selection strategy and checkpoint state."""
        valid = ["priority", "random", "contextual", "utility", "hybrid"]
        if strategy in valid:
            self.selection_strategy = strategy
            logger.info(f"Selection strategy changed to '{strategy}'")
            self._checkpoint_state()  # save new strategy
        else:
            logger.warning(f"Invalid strategy '{strategy}', keeping '{self.selection_strategy}'")

    def get_selection_history(self, limit: int = 100) -> List[Dict]:
        """Return recent selection history."""
        return self.selection_history[-limit:]

    def clear_history(self):
        """Clear selection history."""
        self.selection_history = []

    # ------------------------ Core Selection Strategies -------------------
    @staticmethod
    def _priority_selection(actions: List[Dict]) -> Dict:
        """Highest priority first."""
        return max(actions, key=lambda a: a.get("priority", 0))

    @staticmethod
    def _random_selection(actions: List[Dict]) -> Dict:
        """Uniform random selection among valid actions."""
        return random.choice(actions)

    def _contextual_selection(self, actions: List[Dict], context: Dict) -> Dict:
        """
        Context‑aware heuristic selection:
        - Low energy → idle
        - Holding object and near place position → place
        - Close to destination → move
        - Object nearby and hand empty → pick
        - Otherwise fallback to priority.
        """
        energy = context.get("energy", 10.0)
        max_energy = context.get("max_energy", 10.0)
        energy_ratio = energy / max_energy if max_energy > 0 else 1.0
        dest_distance = context.get("destination_distance", float('inf'))
        object_nearby = context.get("object_nearby", False)
        carrying_items = context.get("carrying_items", 0)
        holding_object = context.get("holding_object", False)
        near_place = context.get("near_place_position", False)

        # Urgent energy recovery
        if energy_ratio < 0.3:
            idle_action = self._find_action_by_name(actions, "idle")
            if idle_action:
                logger.debug("Low energy → idle")
                return idle_action

        # Holding object and near place position → place
        if holding_object and near_place:
            place_action = self._find_action_by_name(actions, "place_object")
            if place_action:
                logger.debug("Holding object near place → place")
                return place_action

        # Close to destination and not holding object → move
        if dest_distance < 2.0 and not holding_object:
            move_action = self._find_action_by_name(actions, "move_to")
            if move_action:
                logger.debug("Close to destination → move")
                return move_action

        # Object nearby and hand empty → pick
        if object_nearby and carrying_items == 0:
            pick_action = self._find_action_by_name(actions, "pick_object")
            if pick_action:
                logger.debug("Object nearby → pick")
                return pick_action

        # Fallback to priority
        return self._priority_selection(actions)

    def _utility_based_selection(self, actions: List[Dict], context: Dict) -> Dict:
        """Select action with highest utility score (cached via ExecutionMemory)."""
        scored = []
        for action in actions:
            utility = self._calculate_utility(action, context)
            scored.append((action, utility))
        best_action, best_score = max(scored, key=lambda x: x[1])
        logger.debug(f"Utility scores: {[(a['name'], s) for a, s in scored]}")
        return best_action

    def _hybrid_selection(self, actions: List[Dict], context: Dict) -> Dict:
        """
        Hybrid: first check urgent needs (via contextual), then utility‑based.
        """
        urgent = self._contextual_selection(actions, context)
        if urgent != self._priority_selection(actions):  # if contextual made a non‑default choice
            return urgent
        return self._utility_based_selection(actions, context)

    # ------------------------ Helper Methods ----------------------------
    def _filter_valid_actions(self, actions: List[Dict], context: Dict) -> List[Dict]:
        """Return only actions whose preconditions are satisfied."""
        disallowed = set(context.get("disallowed_actions", []))
        valid = []
        for action in actions:
            name = action.get("name")
            if name in disallowed:
                continue
            preconditions = self.action_registry.get(name, {}).get("preconditions", [])
            if not preconditions:
                preconditions = action.get("preconditions", [])
            if self._check_preconditions(preconditions, context):
                valid.append(action)
        return valid


    @staticmethod
    def _check_preconditions(preconditions: List[str], context: Dict) -> bool:
        """Return True if all preconditions exist and are truthy in context."""
        return all(context.get(cond, False) for cond in preconditions)

    def _find_action_by_name(self, actions: List[Dict], name: str) -> Optional[Dict]:
        """Return first action with matching name, or None."""
        for a in actions:
            if a.get("name") == name:
                return a
        return None

    def _create_fallback_action(self, context: Dict) -> Dict:
        """Return a minimal idle action when no valid actions exist."""
        logger.warning(f"Using fallback action: {self.fallback_action}")
        return {
            "name": self.fallback_action,
            "priority": 0,
            "preconditions": [],
            "postconditions": ["has_rested"]
        }

    def _record_selection(self, action: Dict, context: Dict):
        """Store selection in history (bounded)."""
        entry = {
            "timestamp": time.time(),
            "action": action.get("name"),
            "priority": action.get("priority"),
            "context_snapshot": {
                k: context.get(k) for k in ["energy", "destination_distance", "holding_object", "object_nearby"]
            }
        }
        self.selection_history.append(entry)
        if len(self.selection_history) > self.max_history:
            self.selection_history.pop(0)

    # ------------------------ Utility Calculation ----------------------
    def _calculate_utility(self, action: Dict, context: Dict) -> float:
        """Compute utility with caching via ExecutionMemory."""
        # Create a cache key from action name and relevant context fields
        context_fingerprint = {
            k: context.get(k) for k in ["energy", "destination_distance", "holding_object",
                                        "object_nearby", "time_critical", "deadline", "current_goal"]
        }
        cache_key = f"util::{action['name']}::{hash(frozenset(context_fingerprint.items()))}"

        # Try to get from cache (memory/disk)
        cached = self.memory.get_cache(
            "utility_score",
            params={"action": action["name"], "context": context_fingerprint},
            namespace="action_selector",
        )
        if cached is not None:
            return cached

        # Compute utility
        base = min(1.0, action.get("priority", 0) / 10.0)
        total = base
        for func_name, weight in self.strategy_weights.items():
            if func_name in self.utility_functions:
                try:
                    util_val = self.utility_functions[func_name](action, context)
                    total += weight * util_val
                except Exception as e:
                    logger.warning(f"Utility function '{func_name}' failed: {e}")
            else:
                logger.debug(f"Utility function '{func_name}' not registered, skipping")

        final = max(0.0, min(1.0, total))

        # Cache with short TTL (context changes often)
        self.memory.set_cache(cache_key, final, ttl=5)
        return final

    def _register_default_utility_functions(self):
        """Register built‑in utility functions used by strategy_weights."""
        self.utility_functions["energy_efficiency"] = self._energy_efficiency_utility
        self.utility_functions["time_critical"] = self._time_critical_utility
        self.utility_functions["goal_proximity"] = self._goal_proximity_utility

    # ----- Built‑in utility functions -----
    def _energy_efficiency_utility(self, action: Dict, context: Dict) -> float:
        """Returns higher utility for low‑cost actions when energy is low."""
        energy = context.get("energy", 10.0)
        max_energy = context.get("max_energy", 10.0)
        energy_ratio = energy / max_energy if max_energy > 0 else 1.0
        cost = self._estimate_action_cost(action, context)
        norm_cost = min(1.0, cost / 2.0)

        if energy_ratio < 0.3:
            return 1.0 - norm_cost
        elif energy_ratio < 0.6:
            return 0.5 - 0.3 * norm_cost
        else:
            return 0.5

    def _time_critical_utility(self, action: Dict, context: Dict) -> float:
        """Higher utility for actions that complete quickly under deadline pressure."""
        deadline = context.get("deadline", float('inf'))
        current_time = context.get("current_time", time.time())
        time_critical = context.get("time_critical", False)

        if not time_critical and deadline == float('inf'):
            return 0.5

        time_remaining = max(0.0, deadline - current_time)
        duration = self._estimate_action_duration(action, context)
        if duration <= 0:
            return 0.5

        if duration > time_remaining:
            return 0.1
        return max(0.2, 1.0 - (duration / max(time_remaining, 0.1)))

    def _goal_proximity_utility(self, action: Dict, context: Dict) -> float:
        """Utility based on how well the action progresses current goal."""
        goal_stack = context.get("goal_stack", [])
        current_goal = context.get("current_goal")

        if not goal_stack and not current_goal:
            return 0.5

        active_goal = goal_stack[-1] if goal_stack else current_goal
        if isinstance(active_goal, dict):
            goal_type = active_goal.get("type", "")
        else:
            goal_type = str(active_goal)

        goal_action_map = {
            "navigate": "move_to", "move": "move_to", "travel": "move_to",
            "collect": "pick_object", "pickup": "pick_object", "gather": "pick_object",
            "place": "place_object", "deposit": "place_object", "deliver": "place_object",
            "rest": "idle", "recover": "idle", "wait": "idle"
        }
        preferred = goal_action_map.get(goal_type)
        if preferred and action.get("name") == preferred:
            return 1.0
        if goal_type == "deliver_package" and action.get("name") in ["move_to", "pick_object", "place_object"]:
            return 0.8
        return 0.3

    # ------------------------ Cost / Duration Estimation ----------------
    def _estimate_action_cost(self, action: Dict, context: Dict) -> float:
        name = action.get("name")
        if name == "move_to":
            distance = context.get("destination_distance", 5.0)
            cost_per_meter = self.move_config.get("energy_cost", 0.05)
            load_factor = 1.0 + 0.3 * context.get("carrying_items", 0)
            return distance * cost_per_meter * load_factor
        elif name == "pick_object":
            base = self.pick_config.get("energy_cost", 0.2)
            props = context.get("object_properties", {})
            weight_factor = props.get("weight", 0.5) / self.pick_config.get("max_weight", 5.0)
            diff_factor = props.get("grasp_difficulty", 0.0)
            return base * (1.0 + weight_factor + diff_factor)
        elif name == "place_object":
            return self.place_config.get("energy_cost", 0.1)
        elif name == "idle":
            rate = self.idle_config.get("energy_recovery_rate", 0.1)
            duration = context.get("idle_duration", self.idle_config.get("default_duration", 5.0))
            return -rate * duration
        return 1.0

    def _estimate_action_duration(self, action: Dict, context: Dict) -> float:
        name = action.get("name")
        if name == "move_to":
            distance = context.get("destination_distance", 5.0)
            speed = self.move_config.get("base_speed", 1.0)
            load_penalty = 1.0 - 0.2 * context.get("carrying_items", 0)
            effective_speed = max(0.1, speed * load_penalty)
            return distance / effective_speed
        elif name == "pick_object":
            difficulty = context.get("object_properties", {}).get("grasp_difficulty", 0.0)
            return self.pick_config.get("grasp_time", 1.0) * (1.0 + difficulty)
        elif name == "place_object":
            return self.place_config.get("place_time", 1.5)
        elif name == "idle":
            return context.get("idle_duration", self.idle_config.get("default_duration", 5.0))
        return 1.0

    # ------------------------ Persistence with ExecutionMemory ----------
    def _checkpoint_state(self):
        """Save current selector configuration (strategy, weights) to a checkpoint."""
        state = {
            "strategy": self.selection_strategy,
            "weights": self.strategy_weights,
            "registered_actions": list(self.action_registry.keys())
        }
        checkpoint_id = self.memory.create_checkpoint(
            state,
            tags=["action_selector", "config"],
            metadata={"timestamp": time.time()}
        )
        logger.debug(f"Selector state checkpointed: {checkpoint_id}")

    def _restore_state(self):
        """Attempt to restore last checkpointed selector state."""
        checkpoints = self.memory.find_checkpoints(tag="action_selector", max_age=86400)  # last 24h
        if checkpoints:
            latest = checkpoints[0]['id']
            state = self.memory.restore_checkpoint(latest)
            if state:
                self.selection_strategy = state.get("strategy", self.selection_strategy)
                self.strategy_weights = state.get("weights", self.strategy_weights)
                logger.info(f"Restored selector state from {latest}")
            else:
                logger.debug("No valid checkpoint state found")

    def export_selector_state(self, path: str):
        """Export selector state (including history) using ExecutionMemory export."""
        self.memory.export_memory(path)

    def import_selector_state(self, path: str):
        """Import selector state from file."""
        self.memory.import_memory(path)
        self._restore_state()


if __name__ == "__main__":
    print("\n=== Testing ActionSelector with Memory ===\n")
    printer.status("TEST", "Starting ActionSelector tests", "info")

    selector = ActionSelector()

    # Register known actions
    selector.register_action("move_to", ["has_destination"], ["at_destination"])
    selector.register_action("pick_object", ["object_detected", "hand_empty"], ["holding_object"])
    selector.register_action("place_object", ["holding_object"], ["hand_empty", "object_placed"])
    selector.register_action("idle", [], ["has_rested"])

    test_actions = [
        {"name": "move_to", "priority": 3},
        {"name": "pick_object", "priority": 5},
        {"name": "place_object", "priority": 4},
        {"name": "idle", "priority": 0}
    ]

    test_scenarios = [
        {"has_destination": True, "object_detected": False, "hand_empty": True, "energy": 10.0},
        {"has_destination": True, "holding_object": True, "destination_distance": 0.5, "near_place_position": True, "energy": 10.0},
        {"object_detected": True, "hand_empty": True, "energy": 2.0, "max_energy": 10.0},
        {"has_destination": True, "destination_distance": 1.0, "energy": 10.0, "time_critical": True, "deadline": time.time() + 2.0},
        {"current_goal": "deliver_package", "has_destination": True, "holding_object": True, "near_place_position": True}
    ]

    for i, ctx in enumerate(test_scenarios):
        print(f"\n--- Scenario {i+1} ---")
        selected = selector.select(test_actions, ctx)
        printer.pretty(f"Selected", selected["name"], "success")

    # Checkpoint and restore test
    selector._checkpoint_state()
    new_selector = ActionSelector()  # should auto‑restore
    print("\n--- Restored Selector Strategy ---")
    printer.pretty("Strategy", new_selector.selection_strategy, "info")