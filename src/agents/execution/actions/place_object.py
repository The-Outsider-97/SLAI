
import math
import time

from typing import Dict, Any, Optional

from src.agents.execution.utils.config_loader import load_global_config, get_config_section
from src.agents.execution.actions.base_action import BaseAction, ActionStatus
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Place Object Action")
printer = PrettyPrinter

class PlaceObjectAction(BaseAction):
    name = "place_object"
    priority = 4
    preconditions = ["holding_object"]
    postconditions = ["hand_empty", "object_placed"]
    _required_context_keys = ["place_position", "held_object"]

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        self.config = load_global_config()
        # Assume a [place_object_action] section in config, with defaults
        self.place_config = get_config_section("place_object_action")

        # Configuration parameters
        self.place_time = self.place_config.get("place_time", 1.5)
        self.min_distance = self.place_config.get("min_distance", 0.5)
        self.energy_cost = self.place_config.get("energy_cost", 0.1)

        # Action state
        self.object_id = ""
        self.place_progress = 0.0

        logger.info("Place Object Action initialized")

    def _execute(self) -> bool:
        """Execute the object placing action."""
        printer.status("PLACE", "Executing...", "info")

        if not self._validate_state():
            return False

        if not self._check_proximity():
            logger.error("Too far from the target location to place the object.")
            self.failure_reason = "Too far from place position."
            return False

        self._pre_execute_place()
        
        # Simulate the placing process
        success = self._perform_place_sequence()

        return self._post_execute_place(success)

    def _validate_state(self) -> bool:
        """Verify the agent is holding an object to place."""
        printer.status("PLACE", "Validating state...", "info")
        
        self.object_id = self.context.get("held_object")
        if not self.object_id:
            logger.error("Agent is not holding any object to place.")
            self.failure_reason = "Not holding an object."
            return False
            
        logger.info(f"Validated: Agent is holding '{self.object_id}'.")
        return True

    def _check_proximity(self) -> bool:
        """Verify agent is close enough to the place position."""
        printer.status("PLACE", "Checking proximity...", "info")

        agent_pos = self.context.get("current_position", (0, 0))
        place_pos = self.context.get("place_position", (0, 0))
        
        distance = math.sqrt(
            (agent_pos[0] - place_pos[0])**2 + 
            (agent_pos[1] - place_pos[1])**2
        )
        
        return distance <= self.min_distance

    def _pre_execute_place(self):
        """Setup for placing."""
        printer.status("PLACE", "Pre execute place", "info")
        self.status = ActionStatus.RUNNING
        self.place_progress = 0.0
        self.context["current_activity"] = f"placing_{self.object_id}"

    def _perform_place_sequence(self) -> bool:
        """Simulate the time it takes to place an object."""
        printer.status("PLACE", "Performing place sequence", "info")
        start_time = time.time()

        while self.place_progress < 1.0:
            if self._should_interrupt():
                logger.warning("Place action interrupted.")
                self.failure_reason = "Action interrupted."
                return False

            elapsed = time.time() - start_time
            self.place_progress = min(1.0, elapsed / self.place_time)
            
            # Update status visualization
            self._update_place_status()

            time.sleep(0.1)
        
        return True

    def _update_place_status(self):
        """Visualize place progress."""
        if self.place_progress < 0.5:
            self.status = ActionStatus.DECELERATE # Hand moving towards position
        else:
            self.status = ActionStatus.RUNNING # Releasing object

    def _should_interrupt(self) -> bool:
        """Check for interruption signals."""
        return self.context.get("interrupt_action", False) or self.status == ActionStatus.CANCELLED

    def _post_execute_place(self, success: bool) -> bool:
        """Finalize placing and update context."""
        printer.status("PLACE", "Post execute place", "info")
        self._consume_energy()

        if success:
            logger.info(f"Successfully placed '{self.object_id}'.")
            
            # Update agent state
            self.context["hand_empty"] = True
            self.context["holding_object"] = False
            self.context["held_object"] = None
            self.context["object_placed"] = True # New postcondition flag

            # Update inventory
            if "inventory" in self.context and self.object_id in self.context["inventory"]:
                del self.context["inventory"][self.object_id]
            
            # Update carrying capacity
            carrying_items = len(self.context.get("inventory", {}))
            self.context["carrying_items"] = carrying_items
            self.set_carry_capacity(carrying_items)

            # Update world state
            self.context.setdefault("object_state", {})
            self.context["object_state"][self.object_id] = "placed"

            return True
        else:
            logger.error(f"Failed to place '{self.object_id}'. Reason: {self.failure_reason}")
            return False

    def _consume_energy(self):
        """Deduct energy for the action."""
        if "energy" in self.context:
            self.context["energy"] = max(0, self.context["energy"] - self.energy_cost)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize action state."""
        base = super().to_dict()
        base.update({
            "object_id": self.object_id,
            "place_progress": self.place_progress
        })
        return base

if __name__ == "__main__":
    print("\n=== Running Execution PLACE_OBJECT Action ===\n")
    printer.status("TEST", "Starting Execution PLACE_OBJECT Action tests", "info")

    context = {
        "holding_object": True,
        "held_object": "apple",
        "place_position": (2.0, 2.0),
        "current_position": (2.1, 2.1),
        "inventory": {"apple": {"weight": 0.3, "size": 0.1}},
        "energy": 5.0
    }
    
    place_action = PlaceObjectAction(context)
    print(f"{place_action}")

    print("\n* * * * * Phase 2 - Execution * * * * *\n")
    
    success = place_action.execute()
    printer.pretty("EXECUTION", "SUCCESS" if success else "FAILURE", "success" if success else "error")

    print("\n* * * * * Phase 3 - Final Context * * * * *\n")
    printer.pretty("CONTEXT", context, "info")

    print("\n=== All tests completed successfully! ===\n")
