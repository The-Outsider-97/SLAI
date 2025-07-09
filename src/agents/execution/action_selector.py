

import math
import random

import time
from typing import Dict, List, Any, Callable, Optional

from src.agents.execution.utils.config_loader import load_global_config, get_config_section
from src.agents.execution.execution_memory import ExecutionMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Action Selector")
printer = PrettyPrinter

class ActionSelector:
    def __init__(self):
        self.config = load_global_config()
        self.config = get_config_section("action_selector")
        self.selection_strategy = self.config.get("strategy", "priority")
        self.strategy_weights = self.config.get("strategy_weights", {})
        self.fallback_action = "idle"

        # Load action-specific configurations
        self.move_config = get_config_section("move_to_action")
        self.pick_config = get_config_section("pick_object_action")
        self.place_config = get_config_section("place_object_action")
        self.idle_config = get_config_section("idle_action")

        # Register available action types
        self.action_registry = {
            "move_to": {
                "preconditions": ["has_destination"],
                "postconditions": ["at_destination"],
            },
            "pick_object": {
                "preconditions": ["object_detected", "hand_empty"],
                "postconditions": ["holding_object"],
            },
            "place_object": {
                "preconditions": ["holding_object"],
                "postconditions": ["hand_empty", "object_placed"],
             },
            "idle": {
                "preconditions": [],
                "postconditions": ["has_rested"]
            }
        }

        self.memory = ExecutionMemory()
        self.utility_cache_key = "action_utility_cache"

        # Configure utility functions
        self.utility_functions = {
            "energy_efficiency": self._energy_efficiency_utility,
            "time_critical": self._time_critical_utility,
            "goal_proximity": self._goal_proximity_utility
        }
        
        logger.info(f"Action Selector initialized with strategy: {self.selection_strategy}")

    def select(self, actions: List[Dict], context: Dict = None) -> Dict:
        """
        Select the best action from a list with context awareness.
        Each action should contain: 'name', 'priority', 'preconditions', etc.
        """
        if not context:
            context = {"energy": 10.0, "max_energy": 10.0}  # Default context
            logger.warning("Using fallback context for action selection")

        if not actions:
            logger.warning("No actions available for selection")
            return self._create_fallback_action(context)

        # Filter out actions with unmet preconditions
        valid_actions = self._filter_valid_actions(actions, context)
        
        if not valid_actions:
            logger.warning("No valid actions after precondition filtering")
            return self._create_fallback_action(context)

        # Apply selection strategy
        if self.selection_strategy == "priority":
            selected = self._priority_selection(valid_actions)
        elif self.selection_strategy == "random":
            selected = self._random_selection(valid_actions)
        elif self.selection_strategy == "contextual":
            selected = self._context_aware_selection(valid_actions, context)
        elif self.selection_strategy == "utility":
            selected = self._utility_based_selection(valid_actions, context)
        elif self.selection_strategy == "hybrid":
            selected = self._hybrid_selection(valid_actions, context)
        else:
            selected = valid_actions[0]  # default fallback

        logger.info(f"Selected action: {selected.get('name')} with priority {selected.get('priority')}")
        return selected

    def _filter_valid_actions(self, actions: List[Dict], context: Dict) -> List[Dict]:
        """Filter actions based on preconditions and context"""
        valid_actions = []
        for action in actions:
            action_name = action.get("name")
            preconditions = self.action_registry.get(action_name, {}).get("preconditions", [])
            
            # If no registered preconditions, use action's own
            if not preconditions:
                preconditions = action.get("preconditions", [])
            
            if self._check_preconditions(preconditions, context):
                valid_actions.append(action)
        return valid_actions

    def _check_preconditions(self, preconditions: List[str], context: Dict) -> bool:
        """Verify all preconditions are satisfied in current context"""
        return all(context.get(cond, False) for cond in preconditions)

    def _priority_selection(self, actions: List[Dict]) -> Dict:
        """Select action with highest priority"""
        return max(actions, key=lambda x: x.get("priority", 0))

    def _random_selection(self, actions: List[Dict]) -> Dict:
        """Select a random valid action"""
        return random.choice(actions)

    def _context_aware_selection(self, actions: List[Dict], context: Dict) -> Dict:
        """
        Context-aware selection based on current state:
        - Prioritizes energy conservation when low on energy
        - Prefers movement when close to destination
        - Favors object interaction when objects are nearby
        """
        # Get context values with defaults
        energy = context.get("energy", 10.0)
        max_energy = context.get("max_energy", 10.0)
        energy_ratio = energy / max_energy
        dest_distance = context.get("destination_distance", float('inf'))
        object_near = context.get("object_nearby", False)
        carrying_items = context.get("carrying_items", 0)
        holding_object = context.get("holding_object", False)
        
        # Decision logic
        if energy_ratio < 0.3:
            logger.debug("Low energy - prioritizing conservation")
            return self._find_action_by_name(actions, "idle") or actions[0]

        # If holding an object and at the destination, placing it is high priority
        if holding_object and dest_distance < 2.0 and self._find_action_by_name(actions, "place_object"):
            logger.debug("At destination while holding object - prioritizing place")
            return self._find_action_by_name(actions, "place_object")

        if dest_distance < 2.0 and self._find_action_by_name(actions, "move_to"):
            logger.debug("Close to destination - prioritizing movement")
            return self._find_action_by_name(actions, "move_to")
        
        if object_near and carrying_items == 0 and self._find_action_by_name(actions, "pick_object"):
            logger.debug("Object nearby - prioritizing pick up")
            return self._find_action_by_name(actions, "pick_object")
        
        # Default to priority selection
        return self._priority_selection(actions)

    def _utility_based_selection(self, actions: List[Dict], context: Dict) -> Dict:
        """Select action based on calculated utility score"""
        scored_actions = []
        for action in actions:
            utility = self._calculate_utility(action, context)
            # Ensure utility is float for logging
            utility_float = float(utility) if isinstance(utility, bytes) else utility
            logger.debug(f"Action {action['name']} utility: {utility_float:.2f}")
            scored_actions.append((action, utility_float))
        
        return max(scored_actions, key=lambda x: x[1])[0]

    def _calculate_utility(self, action: Dict, context: Dict) -> float:
        """Calculate utility score for an action based on multiple factors"""
        cache_key = f"{self.utility_cache_key}::{action['name']}::{hash(str(context))}"

        try:
            cached = self.memory.get_cache(cache_key)
            
            if cached is not None:
                # Convert bytes to float if needed
                if isinstance(cached, bytes):
                    try:
                        cached = float(cached.decode('utf-8'))
                    except (ValueError, UnicodeDecodeError):
                        cached = None
            
            if cached is not None:
                return cached
        
            base_utility = action.get("priority", 0) / 10.0  # Normalize priority
        
            # Apply registered utility functions
            for func_name, weight in self.strategy_weights.items():
                if func_name in self.utility_functions:
                    utility_func = self.utility_functions[func_name]
                    base_utility += weight * utility_func(action, context)
            
            # Cache with context-dependent TTL
            ttl = 300 if "urgent" in context else 3600
            self.memory.set_cache(cache_key, base_utility, ttl=ttl)
        
            return base_utility
    
        except Exception as e:
            logger.warning(f"Cache error for {cache_key}: {str(e)}. Recalculating utility.")
            # Proceed with calculation if cache fails
            return self._calculate_utility_uncached(action, context)

    def _calculate_utility_uncached(self, action: Dict, context: Dict) -> float:
        """Uncached utility calculation"""
        base_utility = action.get("priority", 0) / 10.0  # Normalize priority
    
        # Apply registered utility functions
        for func_name, weight in self.strategy_weights.items():
            if func_name in self.utility_functions:
                utility_func = self.utility_functions[func_name]
                base_utility += weight * utility_func(action, context)
        
        return base_utility

    def create_checkpoint(self, state: Dict):
        """Create action selection checkpoint"""
        return self.memory.create_checkpoint(
            state, 
            tags=["action_selector"],
            metadata={"timestamp": time.time()}
        )

    def _energy_efficiency_utility(self, action: Dict, context: Dict) -> float:
        """Favor actions that conserve energy when reserves are low"""
        energy = context.get("energy", 10.0)
        max_energy = context.get("max_energy", 10.0)
        energy_ratio = energy / max_energy
        
        # Higher utility for low-energy actions when energy is low
        if energy_ratio < 0.4:
            cost = self._estimate_action_cost(action, context)
            return 1.0 - (cost / 2.0)  # Max cost assumed as 2.0
        return 0.5  # Neutral when energy is sufficient

    def _time_critical_utility(self, action: Dict, context: Dict) -> float:
        """Calculate time-critical utility with deadline awareness and action duration estimation"""
        # Get time context with defaults
        current_time = context.get("current_time", time.time())
        deadline = context.get("deadline", float('inf'))
        time_critical = context.get("time_critical", False)
        
        # Calculate time pressure (0-1 normalized)
        time_remaining = max(0, deadline - current_time)
        max_expected_delay = context.get("max_expected_delay", 60.0)  # 1 minute default
        time_pressure = 1.0 - min(time_remaining / max_expected_delay, 1.0) if time_remaining < max_expected_delay else 0.0
        
        # Estimate action duration
        estimated_duration = self._estimate_action_duration(action, context)
        
        # Calculate utility components
        if not time_critical and time_pressure < 0.3:
            return 0.5  # Neutral utility when no time pressure
        
        # Base utility based on action type
        if action["name"] == "idle":
            base_utility = 0.2  # Low utility for idle during critical periods
        elif action["name"] == "move_to":
            # Prioritize movement more when close to destination
            distance = context.get("destination_distance", 10.0)
            proximity_factor = 1.0 / (1.0 + distance) 
            base_utility = 0.6 * proximity_factor
        else:
            base_utility = 0.7
        
        # Apply time pressure scaling
        time_utility = base_utility * (1.0 + time_pressure)
        
        # Penalize long-duration actions under high time pressure
        if time_pressure > 0.7:
            duration_penalty = min(estimated_duration / 10.0, 0.5)  # Max 50% penalty
            time_utility -= duration_penalty
        
        return max(0.1, min(time_utility, 1.0))  # Clamp between 0.1-1.0
    
    def _goal_proximity_utility(self, action: Dict, context: Dict) -> float:
        """Calculate goal utility using precondition matching and contribution weighting"""
        goal_stack = context.get("goal_stack", [])
        current_goal = context.get("current_goal", "")
        
        if not goal_stack and not current_goal:
            return 0.5  # Neutral utility when no goals
        
        # Resolve active goal from stack or single goal
        active_goal = goal_stack[-1] if goal_stack else current_goal
        goal_type = active_goal.get("type") if isinstance(active_goal, dict) else active_goal
        
        # Get action metadata from registry
        action_name = action["name"]
        action_meta = self.action_registry.get(action_name, {})
        action_postconditions = set(action_meta.get("postconditions", []))
        
        # Calculate goal relevance
        relevance = 0.0
        if isinstance(active_goal, dict):
            # Detailed goal matching
            goal_preconditions = set(active_goal.get("preconditions", []))
            goal_requirements = set(active_goal.get("requirements", []))
            
            # Direct contribution (postconditions satisfy goal preconditions)
            direct_contribution = len(action_postconditions & goal_preconditions)
            
            # Partial contribution (satisfies requirements)
            partial_contribution = 0.3 * len(action_postconditions & goal_requirements)
            
            relevance = min(1.0, 0.7 * direct_contribution + partial_contribution)
        else:
            # Simple goal-action mapping
            goal_actions = {
                # Navigation goals
                "navigate": "move_to",
                "move": "move_to",
                "travel_to": "move_to",
                "approach": "move_to",
            
                # Object collection goals
                "collect": "pick_object",
                "pickup": "pick_object",
                "gather": "pick_object",
                "retrieve": "pick_object",
                "grab": "pick_object",
            
                # Object placement goals
                "deposit": "place_object",
                "place": "place_object",
                "drop": "place_object",
                "deliver": "place_object",
                "store": "place_object",
            
                # Rest and idle-related
                "rest": "idle",
                "wait": "idle",
                "recover": "idle",
                "pause": "idle",
                "recharge": "idle",
            
                # Composite or abstract intent
                "deliver_package": ["move_to", "pick_object", "move_to", "place_object"],
                "relocate_object": ["pick_object", "move_to", "place_object"],
                "reposition": ["pick_object", "move_to", "place_object"],
                "reset": "idle",
            
                # Optional: diagnostic/testing/neutral actions
                "do_nothing": "idle",
                "standby": "idle"
            }
            relevance = 1.0 if action_name == goal_actions.get(goal_type, "") else 0.3
        
        # Weight by goal priority (if available)
        goal_priority = active_goal.get("priority", 1.0) if isinstance(active_goal, dict) else 1.0
        return min(1.0, relevance * goal_priority)
    
    def _estimate_action_cost(self, action: Dict, context: Dict) -> float:
        """Estimate action cost using config values and context-sensitive factors"""
        action_name = action.get("name")
        energy = context.get("energy", 10.0)
        
        # Base costs from configuration
        base_costs = {
            "move_to": self.move_config.get("energy_cost", 0.05),
            "pick_object": self.pick_config.get("energy_cost", 0.2),
            "place_object": self.place_config.get("energy_cost", 0.1),
            "idle": -self.idle_config.get("energy_recovery_rate", 0.1)  # Negative cost = gain
        }
        
        # Context adjustments
        cost_adjustment = 1.0
        if action_name == "move_to":
            distance = context.get("destination_distance", 5.0)
            # Consider terrain difficulty
            terrain_factor = context.get("terrain_difficulty", 1.0)
            # Adjust for carrying load
            load_factor = 1.0 + 0.3 * context.get("carrying_items", 0)
            cost_adjustment = distance * terrain_factor * load_factor
            
        elif action_name == "pick_object":
            # Adjust for object properties
            obj_props = context.get("object_properties", {})
            weight_factor = obj_props.get("weight", 0.0) / self.pick_config.get("max_weight", 5.0)
            size_factor = obj_props.get("size", 0.0) / self.pick_config.get("max_size", 1.0)
            difficulty_factor = obj_props.get("grasp_difficulty", 0.0)
            cost_adjustment = 1.0 + weight_factor + size_factor + difficulty_factor

        elif action_name == "place_object":
            # Cost could be adjusted by weight of held object
            obj_id = context.get("held_object")
            if obj_id and "inventory" in context and obj_id in context["inventory"]:
                weight = context["inventory"][obj_id].get("weight", 0.1)
                cost_adjustment = 1.0 + (weight / 2.0)  # Minor adjustment for weight

        elif action_name == "idle":
            # Scale with actual idle duration
            idle_duration = context.get("idle_duration", self.idle_config.get("default_duration", 5.0))
            cost_adjustment = idle_duration / 5.0  # Normalized to default duration
        
        # Calculate final cost
        base_cost = base_costs.get(action_name, 1.0)
        return base_cost * cost_adjustment
    
    # New helper method for time estimation
    def _estimate_action_duration(self, action: Dict, context: Dict) -> float:
        """Estimate action duration using config values and context"""
        action_name = action.get("name")
        
        # Base durations from configuration (seconds)
        base_durations = {
            "move_to": 0.0,  # Calculated dynamically
            "pick_object": self.pick_config.get("grasp_time", 1.0),
            "place_object": self.place_config.get("place_time", 1.5),
            "idle": context.get("idle_duration", self.idle_config.get("default_duration", 5.0))
        }
        
        if action_name == "move_to":
            # Calculate based on distance and speed
            distance = context.get("destination_distance", 5.0)
            base_speed = self.move_config.get("base_speed", 1.0)
            
            # Adjust speed for load
            carrying_items = context.get("carrying_items", 0)
            effective_speed = base_speed * (1.0 - 0.2 * carrying_items)  # 20% reduction per item
            
            # Add obstacle avoidance factor
            avoidance_factor = 1.0 + (0.1 * len(context.get("obstacles", [])))
            
            return (distance / max(effective_speed, 0.1)) * avoidance_factor
        
        # Add 20% variance for non-deterministic actions
        base_duration = base_durations.get(action_name, 1.0)
        return base_duration * random.uniform(0.8, 1.2)

    def _hybrid_selection(self, actions: List[Dict], context: Dict) -> Dict:
        """Hybrid strategy combining priority and context awareness"""
        # First pass: filter by urgency
        urgent_actions = [a for a in actions if self._is_urgent_action(a, context)]
        if urgent_actions:
            return self._priority_selection(urgent_actions)
        
        # Second pass: utility-based selection
        return self._utility_based_selection(actions, context)

    def _is_urgent_action(self, action: Dict, context: Dict) -> bool:
        """Check if an action addresses an urgent need"""
        urgent_needs = context.get("urgent_needs", [])
        
        # Simple urgency mapping
        need_action_map = {
            "low_energy": "idle",
            "near_destination": "move_to",
            "object_available": "pick_object"
        }
        
        for need in urgent_needs:
            if need in need_action_map and action["name"] == need_action_map[need]:
                return True
        return False

    def _find_action_by_name(self, actions: List[Dict], name: str) -> Optional[Dict]:
        """Find a specific action by name"""
        for action in actions:
            if action.get("name") == name:
                return action
        return None

    def _create_fallback_action(self, context: Dict) -> Dict:
        """Create a fallback idle action when no valid actions exist"""
        logger.warning(f"Using fallback action: {self.fallback_action}")
        return {
            "name": self.fallback_action,
            "priority": 0,
            "preconditions": [],
            "postconditions": ["has_rested"]
        }

    def register_action(self, name: str, preconditions: List[str], postconditions: List[str]):
        """Register a new action type for the selector"""
        self.action_registry[name] = {
            "preconditions": preconditions,
            "postconditions": postconditions
        }
        logger.info(f"Registered new action type: {name}")

    def register_utility_function(self, name: str, func: Callable[[Dict, Dict], float]):
        """Register a custom utility function"""
        self.utility_functions[name] = func
        logger.info(f"Registered new utility function: {name}")

    def set_strategy(self, strategy: str):
        """Dynamically change selection strategy"""
        valid_strategies = ["priority", "random", "contextual", "utility", "hybrid"]
        if strategy in valid_strategies:
            self.selection_strategy = strategy
            logger.info(f"Selection strategy changed to: {strategy}")
        else:
            logger.warning(f"Invalid strategy: {strategy}. Keeping current: {self.selection_strategy}")


if __name__ == "__main__":
    print("\n=== Running Execution Action Selector ===\n")
    printer.status("TEST", "Starting Execution Action Selector tests", "info")

    selector = ActionSelector()
    
    # Register test actions
    test_actions = [
        {"name": "move_to", "priority": 3, "preconditions": ["has_destination"]},
        {"name": "pick_object", "priority": 5, "preconditions": ["object_detected", "hand_empty"]},
        {"name": "place_object", "priority": 4, "preconditions": ["holding_object"]},
        {"name": "idle", "priority": 0, "preconditions": []}
    ]
    
    # Test contexts
    test_contexts = [
        {"has_destination": True, "object_detected": False, "hand_empty": True},
        {"has_destination": True, "holding_object": True, "destination_distance": 1.0},
        {"has_destination": False, "object_detected": True, "hand_empty": True, "energy": 3.0},
        {"has_destination": True, "object_detected": True, "hand_empty": True, "destination_distance": 1.5},
        {"has_destination": True, "object_detected": True, "hand_empty": True, "current_goal": "collect_object"}
    ]
    
    # Test strategies
    strategies = ["priority", "random", "contextual", "utility", "hybrid"]
    
    print("\n=== Action Selector Test ===\n")
    for i, context in enumerate(test_contexts):
        printer.status("CONTEXT", f"Test Context {i+1}", "info")
        for strategy in strategies:
            selector.set_strategy(strategy)
            selected = selector.select(test_actions, context)
            printer.pretty(f"Strategy: {strategy}", selected["name"], "success")
        print()
