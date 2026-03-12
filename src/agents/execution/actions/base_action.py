
import inspect
import time

from enum import Enum, auto
from typing import List, Dict, Any, Optional, Set, Tuple, TypeVar, Generic

from src.agents.execution.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Base Action")
printer = PrettyPrinter

ContextType = TypeVar('ContextType', bound=Dict[str, Any])

class SoftInterrupt(Exception):
    pass

class ActionStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    WALKING = auto()
    JUGGING = auto()
    DECELERATE = auto()
    ACCELERATE = auto()
    SUCCESS = auto()
    FAILED = auto()
    CANCELLED = auto()
    INTERRUPTED = auto()

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

    def __init__(self, context: Optional[ContextType] = None):
        self.context = context or {}
        self.status = ActionStatus.PENDING
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.attempt_count: int = 0
        self.failure_reason: Optional[str] = None
        self._state_history: List[Tuple[float, ActionStatus]] = []

        # Movement parameters
        self._target_speed: float = 0.0
        self._current_speed: float = 0.0
        self.carry_capacity: int = 0
        self.movement_speed: float = 1.0

        # Cache for expensive operations
        self._required_context_cache: Optional[List[str]] = None
        logger.info(f"Base Action initialized")

    def _update_status(self, new_status: ActionStatus):
        """Update status with history tracking"""
        self.status = new_status
        self._state_history.append((time.time(), new_status))

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
        for condition in self.postconditions:
            self.context[condition] = True
        logger.debug(f"Applied postconditions: {self.postconditions}")

    def update_movement(self, delta_time: float) -> Tuple[float, ActionStatus]:
        """
        Update movement state and return (current_speed, status)
        Handles transitions between movement states automatically
        """
        printer.status("BASE", "Updating movement...", "info")

        if self.status not in [
            ActionStatus.WALKING, ActionStatus.JUGGING, 
            ActionStatus.DECELERATE, ActionStatus.ACCELERATE
        ]:
            return self._current_speed, self.status

        # Calculate acceleration based on current state
        acceleration_factor = 2.5  # Base acceleration constant
        if self.carry_capacity > 0:
            acceleration_factor *= 0.7  # Reduce acceleration when carrying items
            
        acceleration = acceleration_factor * delta_time
        
        # Handle speed transitions with damping
        speed_diff = self._target_speed - self._current_speed
        if abs(speed_diff) < 0.01:
            self._current_speed = self._target_speed
            new_status = ActionStatus.JUGGING if self.carry_capacity > 0 else ActionStatus.WALKING
            self._update_status(new_status)
        elif speed_diff > 0:
            self._current_speed = min(
                self._current_speed + acceleration, 
                self._target_speed
            )
            self._update_status(ActionStatus.ACCELERATE)
        else:
            self._current_speed = max(
                self._current_speed - acceleration, 
                self._target_speed
            )
            self._update_status(ActionStatus.DECELERATE)

        return self._current_speed, self.status

    def set_movement_target(self, target_speed: float):
        """Set desired movement speed (0-1 normalized)"""
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
        """Dynamically collect required context keys from MRO"""
        if self._required_context_cache is not None:
            return self._required_context_cache
            
        required: Set[str] = set()
        for cls in inspect.getmro(self.__class__):
            if hasattr(cls, '_required_context_keys'):
                required.update(getattr(cls, '_required_context_keys', []))
                
        self._required_context_cache = sorted(list(required))
        return self._required_context_cache

    def validate_context(self) -> bool:
        """Ensure required context parameters exist"""
        printer.status("BASE", "Context validation...", "info")

        required_params = self.get_required_context()
        if not required_params:
            return True
            
        missing = [param for param in required_params if param not in self.context]
        if missing:
            self.failure_reason = f"Missing context keys: {', '.join(missing)}"
            return False
        return True

    def execute(self) -> bool:
        """Main execution workflow with full lifecycle management"""
        printer.status("BASE", "Execute workflow...", "info")

        if self.status == ActionStatus.RUNNING:
            return False

        try:
            if not self.validate_context():
                self._update_status(ActionStatus.FAILED)
                return False
                
            if not self.check_preconditions():
                self._update_status(ActionStatus.FAILED)
                return False
                
            self._pre_execute()
            result = self._execute()
            self._post_execute(result)
            
            return result
        except SoftInterrupt:
            self._handle_interruption()
            return False
        except Exception as e:
            self._handle_failure(str(e))
            return False

    def _pre_execute(self):
        """Setup before main execution"""
        self._update_status(ActionStatus.RUNNING)
        self.attempt_count += 1
        self.start_time = time.time()
        self._current_speed = 0.0
        self._target_speed = self.movement_speed
        logger.info(f"Starting {self.name} action (Attempt {self.attempt_count})")

    def _post_execute(self, success: bool):
        """Cleanup after main execution with movement stopping"""
        # Reset movement flags
        self.context["cancel_movement"] = False
        self.context["urgent_event"] = False

        self._current_speed = 0.0
        self._target_speed = 0.0
        self.end_time = time.time()
        
        if success:
            self._update_status(ActionStatus.SUCCESS)
            self.apply_postconditions()
            logger.info(f"Action completed in {self.elapsed_time:.2f}s")
        else:
            self._update_status(ActionStatus.FAILED)
            logger.error(f"Action failed after {self.elapsed_time:.2f}s")

    def _handle_failure(self, error_msg: str):
        """Manage execution failures"""
        self._update_status(ActionStatus.FAILED)
        self.failure_reason = error_msg
        self.end_time = time.time()
        
        # Auto-retry logic
        if self.attempt_count <= self.retry_attempts:
            self.reset()
            self.execute()
        logger.error(f"Action failed: {error_msg}")

    def _handle_interruption(self):
        """Handle graceful interruptions"""
        self._update_status(ActionStatus.INTERRUPTED)
        self.end_time = time.time()
        self.failure_reason = "Action interrupted by external event"

    def cancel(self):
        """Terminate ongoing action"""
        if self.status == ActionStatus.RUNNING:
            self._update_status(ActionStatus.CANCELLED)
            self.end_time = time.time()
            logger.warning(f"Action cancelled after {self.elapsed_time:.2f}s")

    def reset(self):
        """Reset action to initial state"""
        self._update_status(ActionStatus.PENDING)
        self.start_time = None
        self.end_time = None
        self.failure_reason = None
        self._required_context_cache = None

    @property
    def elapsed_time(self) -> float:
        """Calculate action duration"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        if self.start_time:
            return time.time() - self.start_time
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize action state to dictionary"""
        base_dict = {
            "name": self.name,
            "priority": self.priority,
            "status": self.status.name,
            "elapsed_time": self.elapsed_time,
            "attempt_count": self.attempt_count,
            "failure_reason": self.failure_reason,
            "current_speed": self._current_speed,
            "target_speed": self._target_speed,
            "carry_capacity": self.carry_capacity,
            "state_history": [(t, s.name) for t, s in self._state_history]
        }
        return base_dict

    def __str__(self) -> str:
        return f"{self.name} (Priority: {self.priority}, Status: {self.status.name})"


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
