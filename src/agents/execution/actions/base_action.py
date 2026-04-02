import inspect
import time
import math

from enum import Enum, auto
from typing import (
    Dict, Any, Optional, List, Set, Tuple, TypeVar, Generic, Callable, Union
)
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import deque

from ..utils.execution_error import (ExecutionError, SoftInterrupt, TimeoutError as CustomTimeoutError,
                                        InvalidContextError, ActionFailureError, UnreachableTargetError,
                                        CorruptedContextStateError, ExecutionLoopLockError)
from ..utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Base Action")
printer = PrettyPrinter

ContextType = TypeVar('ContextType', bound=Dict[str, Any])


# ======================== Enums and Data Structures ========================
class ActionStatus(Enum):
    """Extended action status for fine‑grained state tracking."""
    PENDING = auto()
    RUNNING = auto()
    WALKING = auto()       # normal movement
    JOGGING = auto()       # faster, higher energy
    JUGGING = auto()       # carrying items
    ACCELERATING = auto()  # speeding up
    DECELERATING = auto()  # slowing down
    SUCCESS = auto()
    FAILED = auto()
    CANCELLED = auto()
    INTERRUPTED = auto()
    PAUSED = auto()        # external pause
    ACTUATING = auto()     # performing actuator primitive


class MovementProfile(Enum):
    LINEAR = auto()
    S_CURVE = auto()
    TRAPEZOIDAL = auto()


@dataclass
class MovementState:
    """Kinematic state for movement control."""
    position: Tuple[float, float] = (0.0, 0.0)
    velocity: float = 0.0
    acceleration: float = 0.0
    target_speed: float = 0.0
    current_speed: float = 0.0
    heading: float = 0.0          # radians
    profile: MovementProfile = MovementProfile.S_CURVE


@dataclass
class ActuatorCommand:
    """Generic actuator command."""
    type: str                     # e.g., "gripper", "joint", "linear"
    action: str                   # e.g., "open", "close", "set_position"
    value: Union[float, int, bool, Tuple[float, ...]]
    duration: float = 0.5         # seconds
    max_effort: Optional[float] = None
    block: bool = True            # wait for completion?


# ============================= BaseAction Class =============================
class BaseAction(ABC):
    """
    Production‑ready action base class with movement physics, actuator primitives,
    robust interruption handling, and full integration with custom error classes.
    """
    name: str = "base"
    priority: int = 1
    preconditions: List[str] = []
    postconditions: List[str] = []
    cost: float = 1.0              # base resource cost
    timeout: float = 10.0          # seconds
    retry_attempts: int = 0
    retry_backoff: float = 1.0     # seconds between retries
    _required_context_keys: List[str] = []

    def __init__(self, context: Optional[ContextType] = None):
        self.context = context or {}
        self.status = ActionStatus.PENDING
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.attempt_count: int = 0
        self.failure_reason: Optional[str] = None
        self._state_history: List[Tuple[float, ActionStatus]] = []
        self._interrupt_requested: bool = False
        self.carry_capacity = 0

        # Movement system
        self._movement = MovementState()
        self._movement_config: Dict[str, Any] = {}
        self._load_movement_config()

        # Actuator simulation (can be overridden by hardware interface)
        self._actuator_busy: bool = False
        self._actuator_callbacks: Dict[str, Callable[[ActuatorCommand], bool]] = {}

        # Simulation mode flag (False = real‑time, True = step‑driven)
        self.simulation_mode: bool = self.context.get("simulation_mode", False)

        # Performance / logging
        self._step_time: float = 0.05   # default simulation step
        self._last_update: float = 0.0

        logger.info(f"{self.name} action initialized (simulation_mode={self.simulation_mode})")

    # ------------------------ Configuration & Helpers ------------------------
    def _load_movement_config(self):
        """Load movement parameters from global config."""
        self.config = load_global_config()
        self.move_cfg = get_config_section("movement")
        self._movement_config = {
            "max_speed": self.move_cfg.get("max_speed", 2.0),          # m/s
            "max_acceleration": self.move_cfg.get("max_acceleration", 4.0),  # m/s²
            "max_deceleration": self.move_cfg.get("max_deceleration", 5.0),
            "carry_speed_factor": self.move_cfg.get("carry_speed_factor", 0.7),
            "energy_cost_per_meter": self.move_cfg.get("energy_cost_per_meter", 0.5),
            "profile": MovementProfile[self.move_cfg.get("profile", "S_CURVE").upper()]
        }
        self._movement.profile = self._movement_config["profile"]

    def get_required_context(self) -> List[str]:
        """Collect required context keys from class hierarchy."""
        if hasattr(self, "_required_context_cache"):
            return self._required_context_cache
        required: Set[str] = set()
        for cls in inspect.getmro(self.__class__):
            if hasattr(cls, '_required_context_keys'):
                required.update(getattr(cls, '_required_context_keys', []))
        self._required_context_cache = sorted(required)
        return self._required_context_cache

    def validate_context(self) -> bool:
        """
        Ensure all required context keys exist.
        Raises InvalidContextError if missing.
        """
        missing = [k for k in self.get_required_context() if k not in self.context]
        if missing:
            raise InvalidContextError(self.name, missing)
        return True

    def check_preconditions(self) -> bool:
        """Verify preconditions from context. Raises ActionFailureError if unmet."""
        missing = [cond for cond in self.preconditions if not self.context.get(cond)]
        if missing:
            raise ActionFailureError(
                self.name,
                f"Preconditions not satisfied: {missing}"
            )
        return True

    def apply_postconditions(self):
        """Set postcondition flags in context."""
        for cond in self.postconditions:
            self.context[cond] = True
        logger.debug(f"Postconditions applied: {self.postconditions}")

    # ------------------------ Movement System ------------------------
    def set_movement_target(self, target_speed: float):
        """Request a target speed (0.0–1.0 normalized to max_speed)."""
        self._movement.target_speed = max(0.0, min(1.0, target_speed))
        logger.debug(f"Movement target set to {self._movement.target_speed}")

    def set_carry_capacity(self, count: int):
        """Adjust movement speed factor when carrying objects."""
        self.carry_capacity = max(0, count)
        if count > 0:
            factor = self._movement_config["carry_speed_factor"] ** count
            self._movement.target_speed = min(self._movement.target_speed, factor)
            logger.info(f"Carrying {count} items, speed factor = {factor:.2f}")

    def update_movement(self, delta_time: float) -> Tuple[float, ActionStatus]:
        """
        Update kinematic state using the configured acceleration profile.
        Returns (current_speed, new_status).
        """
        if self.status not in (ActionStatus.WALKING, ActionStatus.JOGGING,
                               ActionStatus.JUGGING, ActionStatus.ACCELERATING,
                               ActionStatus.DECELERATING):
            return self._movement.current_speed, self.status

        max_acc = self._movement_config["max_acceleration"]
        max_dec = self._movement_config["max_deceleration"]
        max_speed = self._movement_config["max_speed"]
        target_abs = self._movement.target_speed * max_speed

        speed_error = target_abs - self._movement.current_speed
        if abs(speed_error) < 0.01:
            self._movement.current_speed = target_abs
            self._movement.acceleration = 0.0
            new_status = ActionStatus.JUGGING if self.carry_capacity > 0 else ActionStatus.WALKING
        elif speed_error > 0:
            # Accelerating
            acc = max_acc
            if self._movement.profile == MovementProfile.S_CURVE:
                acc *= min(1.0, self._movement.current_speed / (0.5 * max_speed) + 0.2)
            self._movement.acceleration = acc
            self._movement.current_speed += acc * delta_time
            if self._movement.current_speed > target_abs:
                self._movement.current_speed = target_abs
            new_status = ActionStatus.ACCELERATING
        else:
            # Decelerating
            dec = max_dec
            if self._movement.profile == MovementProfile.S_CURVE:
                dec *= min(1.0, (max_speed - self._movement.current_speed) / (0.3 * max_speed) + 0.2)
            self._movement.acceleration = -dec
            self._movement.current_speed -= dec * delta_time
            if self._movement.current_speed < target_abs:
                self._movement.current_speed = target_abs
            new_status = ActionStatus.DECELERATING

        self._movement.velocity = self._movement.current_speed
        self._update_status(new_status)
        return self._movement.current_speed, self.status

    def _apply_movement_delta(self, delta_time: float, direction: Tuple[float, float]) -> Tuple[float, float]:
        """
        Move the agent based on current speed and direction.
        Returns new position (x, y).
        """
        if not direction or self._movement.current_speed == 0.0:
            return self.context.get("current_position", (0.0, 0.0))
        dx, dy = direction
        norm = math.hypot(dx, dy)
        if norm > 0:
            dx /= norm
            dy /= norm
        step = self._movement.current_speed * delta_time
        pos = self.context.get("current_position", (0.0, 0.0))
        new_pos = (pos[0] + dx * step, pos[1] + dy * step)
        self._consume_movement_energy(step)
        return new_pos

    def _consume_movement_energy(self, distance: float):
        """Deduct energy based on distance and speed."""
        if "energy" in self.context:
            energy_cost = self._movement_config["energy_cost_per_meter"] * distance
            energy_cost *= (1.0 + 0.2 * self._movement.current_speed)
            self.context["energy"] = max(0.0, self.context["energy"] - energy_cost)

    # ------------------------ Actuator Primitives ------------------------
    def register_actuator(self, name: str, handler: Callable[[ActuatorCommand], bool]):
        """Register a hardware/simulation handler for actuator commands."""
        self._actuator_callbacks[name] = handler

    def actuate(self, command: ActuatorCommand) -> bool:
        """
        Send an actuator command. If blocking, waits for completion or timeout.
        Returns True if successful.
        """
        if self._actuator_busy and command.block:
            logger.warning(f"Actuator busy, command {command.type}:{command.action} ignored")
            return False

        handler = self._actuator_callbacks.get(command.type)
        if not handler:
            if not self.simulation_mode:
                time.sleep(command.duration)
            else:
                self._simulate_actuator_delay(command.duration)
            return True

        self._actuator_busy = True
        try:
            success = handler(command)
            if command.block and not self.simulation_mode and command.duration > 0:
                time.sleep(command.duration)
            return success
        finally:
            self._actuator_busy = False

    def _simulate_actuator_delay(self, duration: float):
        """For step‑based simulation, advance a virtual timer."""
        pass

    def set_gripper(self, open: bool, force: float = 1.0, duration: float = 0.3) -> bool:
        return self.actuate(ActuatorCommand(
            type="gripper", action="open" if open else "close",
            value=force, duration=duration
        ))

    def set_joint_angle(self, joint_id: int, angle_rad: float, speed: float = 1.0, duration: float = 0.5) -> bool:
        return self.actuate(ActuatorCommand(
            type="joint", action="set_position",
            value=(joint_id, angle_rad, speed), duration=duration
        ))

    def set_linear_actuator(self, position: float, speed: float = 0.1, duration: float = 1.0) -> bool:
        return self.actuate(ActuatorCommand(
            type="linear", action="set_position",
            value=(position, speed), duration=duration
        ))

    # ------------------------ Execution Lifecycle ------------------------
    def execute(self) -> bool:
        """
        Main entry point with full lifecycle, retries, and custom error handling.
        Raises appropriate ExecutionError subclasses on failures.
        """
        if self.status == ActionStatus.RUNNING:
            logger.warning(f"{self.name} already running")
            return False

        while self.attempt_count <= self.retry_attempts:
            try:
                # Phase 1: Validation
                self.validate_context()
                self.check_preconditions()

                # Phase 2: Execution with timeout
                self._pre_execute()
                result = self._execute_with_timeout()
                self._post_execute(result)
                return result

            except SoftInterrupt as e:
                self._handle_interruption(str(e))
                return False

            except CustomTimeoutError as e:
                # Timeout is a hard failure, no retry
                self._handle_failure(str(e))
                raise  # Re-raise to let upper layer know

            except (InvalidContextError, ActionFailureError, UnreachableTargetError,
                    CorruptedContextStateError, ExecutionLoopLockError) as e:
                self._handle_failure(str(e))
                if self.attempt_count >= self.retry_attempts:
                    raise
                # Backoff before retry
                if not self.simulation_mode:
                    time.sleep(self.retry_backoff)
                self.reset()

            except Exception as e:
                # Catch-all for unexpected errors
                self._handle_failure(f"Unexpected error: {e}")
                if self.attempt_count >= self.retry_attempts:
                    raise ActionFailureError(self.name, str(e))
                if not self.simulation_mode:
                    time.sleep(self.retry_backoff)
                self.reset()

        return False

    def _execute_with_timeout(self) -> bool:
        """Wrap _execute() with a timeout mechanism."""
        if self.timeout <= 0:
            return self._execute()

        start = time.time()
        # We run _execute in a non‑blocking manner; for true timeout we'd need threading.
        # Here we assume _execute itself checks elapsed time internally.
        # To keep it simple, we raise TimeoutError if _execute exceeds the limit.
        result = self._execute()
        elapsed = time.time() - start
        if elapsed > self.timeout:
            raise CustomTimeoutError(self.name, self.timeout, elapsed)
        return result

    @abstractmethod
    def _execute(self) -> bool:
        """Core action logic implemented by subclasses. Should periodically check for interruptions."""
        pass

    def _pre_execute(self):
        """Setup before execution."""
        self._update_status(ActionStatus.RUNNING)
        self.attempt_count += 1
        self.start_time = time.time()
        self._interrupt_requested = False
        self._movement.current_speed = 0.0
        self._movement.velocity = 0.0
        self._movement.acceleration = 0.0
        logger.info(f"Starting {self.name} (attempt {self.attempt_count})")

    def _post_execute(self, success: bool):
        """Cleanup after execution."""
        self.set_movement_target(0.0)
        self._movement.current_speed = 0.0
        self.end_time = time.time()
        if success:
            self._update_status(ActionStatus.SUCCESS)
            self.apply_postconditions()
            logger.info(f"{self.name} succeeded in {self.elapsed_time:.2f}s")
        else:
            self._update_status(ActionStatus.FAILED)
            logger.error(f"{self.name} failed after {self.elapsed_time:.2f}s: {self.failure_reason}")

    def _handle_failure(self, error_msg: str):
        """Record failure reason and update status."""
        self._update_status(ActionStatus.FAILED)
        self.failure_reason = error_msg
        self.end_time = time.time()
        logger.error(f"Action failure: {error_msg}")

    def _handle_interruption(self, reason: str = "external"):
        """Graceful interruption handling."""
        self._update_status(ActionStatus.INTERRUPTED)
        self.end_time = time.time()
        self.failure_reason = f"Interrupted: {reason}"
        logger.warning(self.failure_reason)

    def _should_interrupt(self) -> bool:
        """Check if action should be interrupted (cancelled, paused, or external signal)."""
        if self._interrupt_requested:
            return True
        if self.status == ActionStatus.CANCELLED:
            return True
        if self.status == ActionStatus.PAUSED:
            return True
        # Subclasses may add more checks (e.g., energy depletion)
        return False

    def cancel(self):
        """Request cancellation from outside."""
        self._interrupt_requested = True
        self._update_status(ActionStatus.CANCELLED)
        logger.info(f"{self.name} cancellation requested")

    def pause(self):
        if self.status == ActionStatus.RUNNING:
            self._update_status(ActionStatus.PAUSED)

    def resume(self):
        if self.status == ActionStatus.PAUSED:
            self._update_status(ActionStatus.RUNNING)

    def reset(self):
        """Reset action to pristine state for retry."""
        self.status = ActionStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.failure_reason = None
        self._interrupt_requested = False
        self._movement.current_speed = 0.0
        self._movement.target_speed = 0.0

    # ------------------------ Utilities & Monitoring ------------------------
    def _update_status(self, new_status: ActionStatus):
        self.status = new_status
        self._state_history.append((time.time(), new_status))

    @property
    def elapsed_time(self) -> float:
        if self.start_time:
            if self.end_time:
                return self.end_time - self.start_time
            return time.time() - self.start_time
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "priority": self.priority,
            "status": self.status.name,
            "elapsed_time": self.elapsed_time,
            "attempt_count": self.attempt_count,
            "failure_reason": self.failure_reason,
            "current_speed": self._movement.current_speed,
            "target_speed": self._movement.target_speed,
            "carry_capacity": self.carry_capacity,
            "state_history": [(t, s.name) for t, s in self._state_history],
            "movement_profile": self._movement.profile.name,
        }

    def __str__(self) -> str:
        return f"{self.name} (prio={self.priority}, status={self.status.name})"


if __name__ == "__main__":
    print("\n=== Running Execution Base Action ===\n")
    printer.status("TEST", "Starting Execution Base Action tests", "info")

    # Create a concrete dummy action for testing
    class DummyAction(BaseAction):
        name = "dummy"
        _required_context_keys = ["dummy_data"]

        def _execute(self) -> bool:
            printer.status("DUMMY", "Executing dummy action", "info")
            return True

    context = {"dummy_data": 42, "energy": 10.0}
    action = DummyAction(context)
    print(f"{action}")

    print("\n* * * * * Phase 2 - conditions * * * * *\n")

    dtime=600

    printer.pretty("VALIDATE", action.validate_context(), "success")
    printer.pretty("PRE", action.check_preconditions(), "success")
    printer.pretty("POST", action.apply_postconditions(), "success")
    printer.pretty("MOVE", action.update_movement(delta_time=dtime), "success")
    printer.pretty("DICT", action.to_dict(), "success")

    print("\n* * * * * Phase 3 - Execute * * * * *\n")
    result = action.execute()
    printer.pretty("EXECUTION", "SUCCESS" if result else "FAILURE", "success" if result else "error")

    print("\n=== All tests completed successfully! ===\n")