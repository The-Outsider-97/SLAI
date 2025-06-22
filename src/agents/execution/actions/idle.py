
import time

from typing import Dict, Any, Optional

from src.agents.execution.utils.config_loader import load_global_config, get_config_section
from src.agents.execution.actions.base_action import BaseAction, ActionStatus
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Idle Action")
printer = PrettyPrinter

class IdleAction(BaseAction):
    name = "idle"
    priority = 0
    preconditions = []
    postconditions = ["has_rested"]
    _required_context_keys = ["max_idle_time"]  # Required context parameter
    energy_recovery_rate = 0.1  # Energy recovered per second
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        self.config = load_global_config()
        self.idle_config = get_config_section("idle_action")
        self.idle_duration = 0.0
        self.time_elapsed = 0.0
        self.interruptible = True
        self.energy_gained = 0.0
        
        # Configure from settings
        self.default_duration = self.idle_config.get("default_duration")
        self.min_rest_threshold = self.idle_config.get("min_rest_threshold")
        self.max_rest_threshold = self.idle_config.get("max_rest_threshold")

        logger.info(f"Idle Action initialized")

    def _execute(self) -> bool:
        """Core idle implementation with duration and energy recovery"""
        printer.status("IDLE", "Execute", "info")

        # Determine idle duration
        self.idle_duration = self._calculate_duration()
        logger.info(f"Idling for {self.idle_duration:.1f} seconds...")
        
        # Initialize state
        self.time_elapsed = 0.0
        self.energy_gained = 0.0
        start_time = time.time()
        
        # Idling loop with periodic checks
        while self.time_elapsed < self.idle_duration:
            if self._should_interrupt():
                logger.info("Idle interrupted by external event")
                return False
                
            # Update state
            self.time_elapsed = time.time() - start_time
            self._recover_energy()
            
            # Update status for monitoring
            self._update_status()
            
            # Sleep to avoid busy waiting (short intervals for responsiveness)
            time.sleep(0.1)
        
        logger.info(f"Finished idling. Gained {self.energy_gained:.1f} energy")
        return True

    def _calculate_duration(self) -> float:
        """Determine how long to idle based on context and needs"""
        # Priority: context setting > agent need > default
        if "idle_duration" in self.context:
            return float(self.context["idle_duration"])
        
        # Calculate based on energy deficit
        current_energy = self.context.get("energy", 1.0)
        max_energy = self.context.get("max_energy", 10.0)
        energy_deficit = max_energy - current_energy
        
        if energy_deficit > 0:
            needed_time = energy_deficit / self.energy_recovery_rate
            return min(needed_time, self.context.get("max_idle_time", self.default_duration))
        
        return self.default_duration

    def _recover_energy(self):
        """Gradually restore agent's energy during idle"""
        delta = min(0.1, self.idle_duration - self.time_elapsed)
        energy_gain = delta * self.energy_recovery_rate
        
        if "energy" in self.context:
            self.context["energy"] = min(
                self.context.get("max_energy", 10.0),
                self.context["energy"] + energy_gain
            )
            self.energy_gained += energy_gain

    def _should_interrupt(self) -> bool:
        """Check if idle should be interrupted"""
        # External interruption signal
        if self.context.get("interrupt_idle", False):
            return True
            
        # Critical event requires attention
        if self.context.get("urgent_event", False):
            return True
            
        # Agent cancellation
        if self.status == ActionStatus.CANCELLED:
            return True
            
        return False

    def _update_status(self):
        """Update action status during execution"""
        progress = self.time_elapsed / self.idle_duration
        
        if progress < 0.3:
            self.status = ActionStatus.ACCELERATE  # "Warming up" state
        elif progress > 0.7:
            self.status = ActionStatus.DECELERATE  # "Winding down" state
        else:
            self.status = ActionStatus.RUNNING  # Steady state

    def _pre_execute(self):
        """Specialized setup for idle action"""
        super()._pre_execute()
        logger.info("Entering idle state")
        
        # Initialize energy tracking
        self.context.setdefault("energy", 8.0)
        self.context.setdefault("max_energy", 10.0)

    def _post_execute(self, success: bool):
        """Cleanup after idle completes"""
        super()._post_execute(success)
        
        # Reset interruption flags
        if "interrupt_idle" in self.context:
            self.context["interrupt_idle"] = False
            
        # Always apply rest benefit even if interrupted
        if self.time_elapsed > self.min_rest_threshold:
            self.context["has_rested"] = True
            logger.debug("Applied rest benefits")

    def to_dict(self) -> Dict[str, Any]:
        """Enhanced serialization with idle-specific info"""
        base = super().to_dict()
        base.update({
            "idle_duration": self.idle_duration,
            "time_elapsed": self.time_elapsed,
            "energy_gained": self.energy_gained,
            "recovery_rate": self.energy_recovery_rate
        })
        return base

if __name__ == "__main__":
    print("\n=== Running Execution IDLE Action ===\n")
    printer.status("TEST", "Starting Execution IDLE Action tests", "info")
    context={"idle_duration": 8.0, "energy": 5.0}

    idle = IdleAction()
    print(f"{idle}")

    print("\n* * * * * Phase 2 * * * * *\n")

    success= True

    printer.pretty("EXEC", idle._execute(), "success")
    printer.pretty("CALC", idle._calculate_duration(), "success")
    printer.pretty("RECOVER", idle._recover_energy(), "success")
    printer.pretty("INTERRUPT", idle._should_interrupt(), "success")
    printer.pretty("STATUS", idle._update_status(), "success")
    printer.pretty("PRE", idle._pre_execute(), "success")
    printer.pretty("POST", idle._post_execute(success=success), "success")
    printer.pretty("DICT", idle.to_dict(), "success")

    print("\n=== All tests completed successfully! ===\n")
