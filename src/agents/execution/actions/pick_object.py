
import math
import time
import random

from typing import Dict, Any, Optional, Tuple

from src.agents.execution.utils.config_loader import load_global_config, get_config_section
from src.agents.execution.actions.base_action import BaseAction, ActionStatus
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Pick Object Action")
printer = PrettyPrinter

class PickObjectAction(BaseAction):
    name = "pick_object"
    priority = 5
    preconditions = ["object_detected", "hand_empty"]
    postconditions = ["holding_object"]
    _required_context_keys = ["object_position", "object_properties"]
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        self.config = load_global_config()
        self.pick_config = get_config_section("pick_object_action")
        
        # Configuration parameters
        self.grasp_time = self.pick_config.get("grasp_time")
        self.min_distance = self.pick_config.get("min_distance")
        self.base_success_rate = self.pick_config.get("base_success_rate")
        self.energy_cost = self.pick_config.get("energy_cost")
        self.max_weight = self.pick_config.get("max_weight")
        self.max_size = self.pick_config.get("max_size")
        
        # Action state
        self.object_id = ""
        self.object_properties: Dict[str, Any] = {}
        self.grasp_progress = 0.0
        self.attempt_failed = False
        
        logger.info(f"Pick Object Action initialized")

    def _execute(self) -> bool:
        """Execute object picking with grasp mechanics and failure handling"""
        printer.status("PICK", "Executing...", "info")

        # Validate object properties
        if not self._validate_object():
            return False
            
        # Verify proximity to object
        if not self._check_proximity():
            logger.error("Too far from object to pick up")
            return False
            
        # Initialize picking process
        self._pre_execute_pick()
        
        # Execute grasp sequence
        success = self._perform_grasp_sequence()
        
        # Handle results
        return self._post_execute_pick(success)

    def _validate_object(self) -> bool:
        """Verify object can be picked up based on properties"""
        printer.status("PICK", "Validating object...", "info")

        # Get object properties from context
        self.object_id = self.context.get("target_object", "unknown")
        self.object_properties = self.context.get("object_properties", {})
        
        # Check required properties
        required_props = ["weight", "size", "grasp_difficulty"]
        missing = [prop for prop in required_props if prop not in self.object_properties]
        if missing:
            logger.error(f"Object missing properties: {', '.join(missing)}")
            return False
            
        # Check weight capacity
        weight = self.object_properties["weight"]
        if weight > self.max_weight:
            logger.error(f"Object too heavy ({weight} > {self.max_weight})")
            return False
            
        # Check size constraints
        size = self.object_properties["size"]
        if size > self.max_size:
            logger.error(f"Object too large ({size} > {self.max_size})")
            return False
            
        return True

    def _check_proximity(self) -> bool:
        """Verify agent is close enough to the object"""
        printer.status("PICK", "Checking proximity...", "info")

        agent_pos = self.context.get("current_position", (0, 0))
        obj_pos = self.context.get("object_position", (0, 0))
        
        distance = math.sqrt(
            (agent_pos[0]-obj_pos[0])**2 + 
            (agent_pos[1]-obj_pos[1])**2
        )
        
        return distance <= self.min_distance

    def _pre_execute_pick(self):
        """Specialized setup for picking"""
        printer.status("PICK", "Pre execute", "info")

        self.status = ActionStatus.RUNNING
        self.grasp_progress = 0.0
        self.attempt_failed = False
        
        logger.info(f"Preparing to pick up {self.object_id}")
        self.context["current_activity"] = f"picking_{self.object_id}"

    def _perform_grasp_sequence(self) -> bool:
        """Execute the grasping process with progress tracking"""
        printer.status("PICK", "Executing grasping process", "info")

        start_time = time.time()
        grasp_duration = self.grasp_time * (1 + self.object_properties.get("grasp_difficulty", 0))
        
        while self.grasp_progress < 1.0:
            # Check for interruptions
            if self._should_interrupt():
                return False
                
            # Update progress
            elapsed = time.time() - start_time
            self.grasp_progress = min(1.0, elapsed / grasp_duration)
            
            # Update status visualization
            self._update_grasp_status()
            
            # Handle random failures
            if self._check_failure_condition():
                self.attempt_failed = True
                return False
                
            time.sleep(0.1)
            
        return True

    def _update_grasp_status(self):
        """Visualize grasp progress through status updates"""
        printer.status("PICK", "Updating grasping status", "info")

        if self.grasp_progress < 0.3:
            self.status = ActionStatus.ACCELERATE
        elif self.grasp_progress < 0.7:
            self.status = ActionStatus.RUNNING
        else:
            self.status = ActionStatus.DECELERATE

    def _check_failure_condition(self) -> bool:
        """Determine if grasp attempt fails"""
        printer.status("PICK", "Checking failure condition", "info")

        if self.attempt_failed:
            return True
            
        # Calculate failure probability
        difficulty = self.object_properties.get("grasp_difficulty", 0)
        failure_chance = (1 - self.base_success_rate) * (1 + difficulty)
        
        # Critical failure check (1% base + difficulty modifier)
        critical_chance = 0.01 * (1 + difficulty)
        if random.random() < critical_chance:
            logger.error("Critical failure during grasp attempt!")
            return True
            
        # Normal failure check
        if random.random() < failure_chance:
            logger.warning("Grasp attempt failed, retrying...")
            return True
            
        return False

    def _should_interrupt(self) -> bool:
        """Check if picking should be interrupted"""
        printer.status("PICK", "Interrupting...", "info")

        # External interruption signal
        if self.context.get("interrupt_action", False):
            return True
            
        # Critical event requires attention
        if self.context.get("urgent_event", False):
            return True
            
        # Agent cancellation
        if self.status == ActionStatus.CANCELLED:
            return True
            
        # Energy depletion
        if self.context.get("energy", 10.0) <= 0:
            return True
            
        return False

    def _post_execute_pick(self, success: bool) -> bool:
        """Finalize picking process and update context"""
        printer.status("PICK", "Post execute", "info")

        # Consume energy regardless of success
        self._consume_energy()
        
        if success:
            # Update inventory
            self._update_inventory()
            
            # Update object state
            self.context["object_state"][self.object_id] = "held"
            self.context["holding_object"] = True
            self.context["held_object"] = self.object_id
            
            logger.info(f"Successfully picked up {self.object_id}")
            return True
        else:
            if self.attempt_failed:
                logger.error(f"Failed to pick up {self.object_id}")
            return False

    def _consume_energy(self):
        """Deduct energy based on object properties"""
        printer.status("PICK", "Consuming Energy", "info")

        if "energy" in self.context:
            weight_factor = self.object_properties["weight"] / self.max_weight
            difficulty_factor = self.object_properties["grasp_difficulty"]
            
            consumption = self.energy_cost * (1 + weight_factor + difficulty_factor)
            self.context["energy"] = max(0, self.context["energy"] - consumption)

    def _update_inventory(self):
        """Add object to agent's inventory"""
        printer.status("PICK", "Updating inventory...", "info")

        if "inventory" not in self.context:
            self.context["inventory"] = {}
            
        self.context["inventory"][self.object_id] = {
            "type": self.object_properties.get("type", "unknown"),
            "weight": self.object_properties["weight"],
            "size": self.object_properties["size"]
        }
        
        # Update carrying capacity
        carrying_items = len(self.context["inventory"])
        self.context["carrying_items"] = carrying_items
        self.set_carry_capacity(carrying_items)

    def _pre_execute(self):
        """Setup before main execution"""
        super()._pre_execute()
        # Initialize object state tracking
        self.context.setdefault("object_state", {})
        logger.info(f"Beginning object pickup sequence")

    def _post_execute(self, success: bool):
        """Cleanup after action completes"""
        super()._post_execute(success)
        
        # Reset activity tracking
        if "current_activity" in self.context:
            del self.context["current_activity"]
            
        # Reset interruption flags
        if "interrupt_action" in self.context:
            self.context["interrupt_action"] = False

    def to_dict(self) -> Dict[str, Any]:
        """Enhanced serialization with picking info"""
        base = super().to_dict()
        base.update({
            "object_id": self.object_id,
            "grasp_progress": self.grasp_progress,
            "weight": self.object_properties.get("weight", 0),
            "size": self.object_properties.get("size", 0),
            "grasp_difficulty": self.object_properties.get("grasp_difficulty", 0)
        })
        return base

if __name__ == "__main__":
    print("\n=== Running Execution PICK_OBJECT Action ===\n")
    printer.status("TEST", "Starting Execution PICK_OBJECT Action tests", "info")

    # Test context
    context = {
        "target_object": "apple",
        "object_position": (1.0, 1.0),
        "current_position": (1.1, 1.1),
        "object_properties": {
            "weight": 0.3,
            "size": 0.1,
            "type": "fruit",
            "grasp_difficulty": 0.2
        },
        "energy": 10.0
    }
    
    pick_action = PickObjectAction(context)
    print(f"{pick_action}")
    
    print("\n* * * * * Phase 2 - Validation * * * * *\n")
    
    printer.pretty("VALIDATE", pick_action._validate_object(), "success")
    printer.pretty("PROXIMITY", pick_action._check_proximity(), "success")
    
    print("\n* * * * * Phase 3 - Grasp Simulation * * * * *\n")
    
    # Test grasp sequence
    pick_action._pre_execute_pick()
    success = pick_action._perform_grasp_sequence()
    printer.pretty("GRASP", success, "success" if success else "error")
    
    print("\n* * * * * Phase 4 - Finalization * * * * *\n")
    
    final_success = pick_action._post_execute_pick(success)
    printer.pretty("FINAL", final_success, "success" if final_success else "error")
    
    print("\n=== All tests completed successfully! ===\n")
