
import inspect
import time

from enum import Enum
from typing import List, Dict, Any, Optional, Set, Tuple

from src.agents.execution.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Base Action")
printer = PrettyPrinter

class SoftInterrupt(Exception):
    pass

class ActionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    WALKING = "walking"
    JUGGING = "jugging"
    DECELERATE = "decelerate"
    ACCELERATE = "accelerate"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BaseAction:
    name: str = "base"
    priority: int = 1
    preconditions: List[str] = []
    postconditions: List[str] = []
    cost: float = 1.0  # Action cost (time/energy/resources)
    timeout: float = 10.0  # Default timeout in seconds
    retry_attempts: int = 0
    movement_speed: float = 1.0
    carry_capacity: int = 0
    _required_context_keys: List[str] = []
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        self.config = load_global_config()
        self.base_config = get_config_section('base_action')
        self.context = context or {}
        self.status = ActionStatus.PENDING
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.attempt_count: int = 0
        self.failure_reason: Optional[str] = None
        self._target_speed: float = 1.0
        self._current_speed: float = 0.0

        logger.info(f"Base Action initialized")

    def check_preconditions(self) -> bool:
        """Verify all preconditions are satisfied in current context"""
        printer.status("CHECK", "Checking preconditions...", "info")

        missing = [cond for cond in self.preconditions if not self.context.get(cond)]
        if missing:
            logger.warning(f"Missing preconditions: {', '.join(missing)}")
            return False
        return True

    def apply_postconditions(self):
        """Update context with action results"""
        printer.status("BASE", "Apply postconditions...", "info")

        for condition in self.postconditions:
            self.context[condition] = True
        logger.debug(f"Applied postconditions: {self.postconditions}")

    def update_movement(self, delta_time: float) -> Tuple[float, ActionStatus]:
        """
        Update movement state and return (current_speed, status)
        Handles transitions between movement states automatically
        """
        printer.status("BASE", "Updating movement...", "info")

        if self.status not in [ActionStatus.WALKING, ActionStatus.JUGGING, 
                              ActionStatus.DECELERATE, ActionStatus.ACCELERATE]:
            return self._current_speed, self.status

        # Handle speed transitions
        acceleration = 0.5 * delta_time  # Normalized acceleration rate
        if abs(self._current_speed - self._target_speed) < 0.01:
            self._current_speed = self._target_speed
            if self.carry_capacity > 0:
                self.status = ActionStatus.JUGGING
            else:
                self.status = ActionStatus.WALKING
        elif self._current_speed < self._target_speed:
            self._current_speed = min(self._current_speed + acceleration, self._target_speed)
            self.status = ActionStatus.ACCELERATE
        else:
            self._current_speed = max(self._current_speed - acceleration, self._target_speed)
            self.status = ActionStatus.DECELERATE

        return self._current_speed, self.status

    def set_movement_target(self, target_speed: float):
        """Set desired movement speed (0-1 normalized)"""
        printer.status("BASE", "Set movement target", "info")

        self._target_speed = max(0.0, min(1.0, target_speed))
        logger.debug(f"Movement target set to {self._target_speed}")

    def set_carry_capacity(self, count: int):
        """Update carrying status which affects movement"""
        self.carry_capacity = max(0, count)
        if count > 0:
            # Reduce max speed when carrying items
            self._target_speed = min(self._target_speed, 0.7)  # 70% max speed when carrying
            logger.info(f"Now carrying {count} items, adjusting speed")
        self.movement_speed = 1.0 - (0.2 * count)  # 20% reduction per item

    def get_required_context(self) -> List[str]:
        """
        Dynamically collects all required context keys from the action's
        entire class hierarchy. This allows subclasses to define their own
        specific needs without overwriting the requirements of their parents.
        The method traverses the MRO (Method Resolution Order) and aggregates
        the '_required_context_keys' from each class.
        """
        required: Set[str] = set()
        for cls in inspect.getmro(self.__class__):
            if hasattr(cls, '_required_context_keys'):
                keys = getattr(cls, '_required_context_keys', [])
                required.update(keys)
        return sorted(list(required))

    def validate_context(self) -> bool:
        """Ensure required context parameters exist"""
        printer.status("BASE", "Context validation...", "info")

        required_params = self.get_required_context()
        if not required_params:
            return True
        missing = [param for param in required_params if param not in self.context]
        if missing:
            logger.error(f"Missing context parameters: {', '.join(missing)}")
            return False
        return True

    def execute(self) -> bool:
        """Main execution workflow with full lifecycle management"""
        printer.status("BASE", "Execute workflow...", "info")

        if self.status == ActionStatus.RUNNING:
            logger.warning("Action already in progress")
            return False

        try:
            if not self.validate_context():
                return False
                
            if not self.check_preconditions():
                return False
                
            self._pre_execute()
            result = self._execute()
            self._post_execute(result)
            
            return result
        except Exception as e:
            self._handle_failure(str(e))
            return False

    def _pre_execute(self):
        """Setup before main execution with movement initialization"""
        printer.status("BASE", "Pre execution", "info")

        self.status = ActionStatus.RUNNING
        self.attempt_count += 1
        self.start_time = time.time()
        self._current_speed = 0.0
        self._target_speed = self.movement_speed
        logger.info(f"Starting {self.name} action (Attempt {self.attempt_count})")

    def _post_execute(self, success: bool):
        """Cleanup after main execution with movement stopping"""
        printer.status("BASE", "Post execution", "info")

        # Reset movement flags
        self.context["cancel_movement"] = False
        self.context["urgent_event"] = False

        self._current_speed = 0.0
        self._target_speed = 0.0
        self.end_time = time.time()
        
        if success:
            self.status = ActionStatus.SUCCESS
            self.apply_postconditions()
            logger.info(f"Action completed in {self.elapsed_time:.2f}s")
        else:
            self.status = ActionStatus.FAILED
            logger.error(f"Action failed after {self.elapsed_time:.2f}s")

    def _handle_failure(self, error_msg: str):
        """Manage execution failures"""
        printer.status("BASE", "Handling failure...", "warning")

        self.status = ActionStatus.FAILED
        self.failure_reason = error_msg
        self.end_time = time.time()
        logger.error(f"Action failed: {error_msg}")

    def cancel(self):
        """Terminate ongoing action"""
        printer.status("BASE", "Canceling...", "warning")

        if self.status == ActionStatus.RUNNING:
            self.status = ActionStatus.CANCELLED
            self.end_time = time.time()
            logger.warning(f"Action cancelled after {self.elapsed_time:.2f}s")

    def reset(self):
        """Reset action to initial state"""
        self.status = ActionStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.failure_reason = None

    @property
    def elapsed_time(self) -> float:
        """Calculate action duration"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        if self.start_time:
            return time.time() - self.start_time
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize action state to dictionary with movement info"""
        base_dict = {
            "name": self.name,
            "priority": self.priority,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "cost": self.cost,
            "status": self.status.value,
            "elapsed_time": self.elapsed_time,
            "context_keys": list(self.context.keys()),
            "class": self.__class__.__name__,
            "current_speed": self._current_speed,
            "target_speed": self._target_speed,
            "carry_capacity": self.carry_capacity,
        }
        return base_dict

    def __str__(self) -> str:
        return f"{self.name} (Priority: {self.priority}, Status: {self.status.value})"


if __name__ == "__main__":
    print("\n=== Running Execution Base Action ===\n")
    printer.status("TEST", "Starting Execution Base Action tests", "info")
    context=None

    base = BaseAction()
    print(f"{base}")

    print("\n* * * * * Phase 2 - conditions * * * * *\n")

    printer.pretty("PRE", base.check_preconditions(), "success")
    printer.pretty("POST", base.apply_postconditions(), "success")

    print("All tests completed successfully!")
