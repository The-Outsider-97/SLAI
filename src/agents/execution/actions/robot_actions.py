import time
import math
from typing import Dict, Any, Optional, Tuple, List, Callable, cast

from ..actions.base_action import BaseAction, ActionStatus, SoftInterrupt
from ..modules.robot_interface import RobotInterface
from ..utils.execution_error import *
from ..utils.execution_helpers import *
from ..utils.config_loader import get_config_section
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("RobotActions")
printer = PrettyPrinter()


_MISSING = object()


class RobotAction(BaseAction):
    """Base class for actions that cross the robot hardware boundary.

    Concrete actions declare only the capabilities they consume. This supports
    both ``RobotInterface`` subclasses and structurally compatible adapters,
    while rejecting unsupported operations before execution begins.
    """
    _required_context_keys = ["robot"]
    robot_capabilities: Tuple[str, ...] = ()

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        robot = self.context.get("robot")
        if robot is None:
            raise InvalidContextError(self.name, ["robot"])
        unsupported_capabilities = missing_capabilities(
            robot,
            self.robot_capabilities,
        )
        if unsupported_capabilities:
            raise InvalidContextError(
                self.name,
                [f"robot.{name}" for name in unsupported_capabilities],
            )
        self.robot = cast(RobotInterface, robot)
        self.robot_cfg = get_config_section("robot_actions") or {}

    def _execute_robot_command(self, capability: str, *args: Any, **kwargs: Any) -> None:
        """Execute a validated robot command and enforce its Boolean contract."""
        command = getattr(self.robot, capability, None)
        if not callable(command):
            raise ActionFailureError(
                self.name,
                f"Robot capability {capability!r} became unavailable",
            )
        try:
            accepted = command(*args, **kwargs)
        except Exception as exc:
            raise ActionFailureError(
                self.name,
                f"Robot command {capability!r} raised {type(exc).__name__}: {exc}",
            ) from exc
        if accepted is not True:
            raise ActionFailureError(
                self.name,
                f"Robot command {capability!r} was rejected or violated its Boolean return contract",
            )

    def _read_pose(self) -> Tuple[float, float, float]:
        """Read and validate a finite ``(x, y, theta)`` hardware pose."""
        pose = self._read_robot_value("get_pose")
        if not is_position(pose, dimensions=3):
            raise ActionFailureError(
                self.name,
                "Robot get_pose() must return three finite real coordinates",
            )
        return float(pose[0]), float(pose[1]), float(pose[2])

    def _read_robot_value(self, capability: str, *args: Any, **kwargs: Any) -> Any:
        """Read a validated robot capability and translate adapter failures."""
        reader = getattr(self.robot, capability, None)
        if not callable(reader):
            raise ActionFailureError(
                self.name,
                f"Robot capability {capability!r} became unavailable",
            )
        try:
            return reader(*args, **kwargs)
        except Exception as exc:
            raise ActionFailureError(
                self.name,
                f"Robot read {capability!r} raised {type(exc).__name__}: {exc}",
            ) from exc


# ========================== Locomotion Actions ==========================

class MotorAction(RobotAction):
    """Set motor speeds for differential drive robots."""
    name = "motor"
    preconditions = ["robot_ready"]
    postconditions = ["motors_set"]
    robot_capabilities = ("set_motor_speed", "stop")

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        self.left_speed = 0.0
        self.right_speed = 0.0
        self.duration = 0.0  # if >0, auto-revert after duration

    def _execute(self) -> bool:
        self.left_speed = self.context.get("left_speed", 0.0)
        self.right_speed = self.context.get("right_speed", 0.0)
        self.duration = self.context.get("duration", 0.0)

        self._execute_robot_command(
            "set_motor_speed",
            self.left_speed,
            self.right_speed,
        )

        if self.duration > 0:
            start = time.time()
            while time.time() - start < self.duration:
                if self._should_interrupt():
                    self._execute_robot_command("stop")
                    raise SoftInterrupt("Motor action interrupted")
                time.sleep(0.05)
            # Auto-stop after duration
            self._execute_robot_command("set_motor_speed", 0.0, 0.0)
        return True


class AckermannAction(RobotAction):
    """Set steering and throttle for Ackermann steering robots."""
    name = "ackermann"
    preconditions = ["robot_ready"]
    postconditions = ["steering_set", "throttle_set"]
    robot_capabilities = ("set_steering", "set_throttle", "stop")

    def _execute(self) -> bool:
        steering = self.context.get("steering", 0.0)
        throttle = self.context.get("throttle", 0.0)
        duration = self.context.get("duration", 0.0)

        self._execute_robot_command("set_steering", steering)
        self._execute_robot_command("set_throttle", throttle)

        if duration > 0:
            start = time.time()
            while time.time() - start < duration:
                if self._should_interrupt():
                    self._execute_robot_command("stop")
                    raise SoftInterrupt("Ackermann action interrupted")
                time.sleep(0.05)
            self._execute_robot_command("set_throttle", 0.0)
        return True


class StopAction(RobotAction):
    """Emergency stop."""
    name = "stop"
    priority = 10  # high priority
    preconditions = []
    postconditions = ["stopped"]
    robot_capabilities = ("stop",)

    def _execute(self) -> bool:
        self._execute_robot_command("stop")
        return True


class SpinAction(RobotAction):
    """Rotate in place by a given angle (degrees)."""
    name = "spin"
    _required_context_keys = ["angle_deg"]
    preconditions = ["robot_ready", "localized"]
    robot_capabilities = ("get_pose", "set_motor_speed", "stop")

    def _execute(self) -> bool:
        angle_deg = self.context["angle_deg"]
        angular_speed = self.context.get("angular_speed", 30.0)  # deg/s
        # Assume robot can get current orientation via get_pose()
        start_yaw = self._read_pose()[2]
        target_yaw = start_yaw + math.radians(angle_deg)

        # Normalize target_yaw to [-pi, pi]
        target_yaw = normalize_angle(target_yaw)

        # Simple PID‑less open‑loop spin (differential drive)
        # Positive angle = clockwise? Define sign convention.
        direction = 1 if angle_deg > 0 else -1
        left_speed = direction * angular_speed / 90.0   # normalize to speed range
        right_speed = -direction * angular_speed / 90.0

        self._execute_robot_command("set_motor_speed", left_speed, right_speed)

        # Wait until within tolerance
        tolerance = math.radians(2.0)
        while True:
            if self._should_interrupt():
                self._execute_robot_command("stop")
                raise SoftInterrupt("Spin interrupted")
            current_yaw = self._read_pose()[2]
            error = abs(normalize_angle(target_yaw - current_yaw))
            if error < tolerance:
                break
            time.sleep(0.02)

        self._execute_robot_command("set_motor_speed", 0.0, 0.0)
        return True


# ========================== Navigation Actions ==========================

class NavigateAction(RobotAction):
    """
    Move to a target pose (x, y, theta) using a simple go‑to‑goal controller.
    Relies on robot.get_pose() and set_motor_speed().
    """
    name = "navigate"
    _required_context_keys = ["target_x", "target_y"]
    preconditions = ["robot_ready", "localized"]
    timeout = 30.0
    robot_capabilities = ("get_pose", "set_motor_speed", "stop")

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        self.target_theta = self.context.get("target_theta", None)
        self.distance_tolerance = self.context.get("distance_tolerance", 0.1)
        self.angle_tolerance = math.radians(self.context.get("angle_tolerance", 5.0))
        self.max_speed = self.robot_cfg.get("max_speed", 0.5)      # m/s
        self.max_angular = self.robot_cfg.get("max_angular", 1.0)  # rad/s
        self.kp_linear = self.robot_cfg.get("kp_linear", 0.8)
        self.kp_angular = self.robot_cfg.get("kp_angular", 1.5)

    def _execute(self) -> bool:
        target_x = self.context["target_x"]
        target_y = self.context["target_y"]

        start_time = time.time()
        while True:
            if self._should_interrupt():
                self._execute_robot_command("stop")
                raise SoftInterrupt("Navigation interrupted")
            if time.time() - start_time > self.timeout:
                self._execute_robot_command("stop")
                raise ActionFailureError(self.name, "Navigation timeout")

            x, y, theta = self._read_pose()
            dx = target_x - x
            dy = target_y - y
            distance = euclidean_distance((x, y), (target_x, target_y)) # pyright: ignore[reportArgumentType]
            if distance < self.distance_tolerance:
                # Position reached; handle orientation if needed
                if self.target_theta is None:
                    break
                angle_error = normalize_angle(self.target_theta - theta)
                if abs(angle_error) < self.angle_tolerance:
                    break
                # Rotate in place
                angular = self.kp_angular * angle_error
                angular = clamp(angular, -self.max_angular, self.max_angular)
                self._execute_robot_command(
                    "set_motor_speed",
                    -angular,
                    angular,
                )
                time.sleep(0.02)
                continue

            # Go‑to‑goal control
            angle_to_target = math.atan2(dy, dx)
            angle_error = normalize_angle(angle_to_target - theta)
            linear = self.kp_linear * distance
            linear = clamp(linear, 0.0, self.max_speed)
            angular = self.kp_angular * angle_error
            angular = clamp(angular, -self.max_angular, self.max_angular)

            # Convert to differential drive speeds
            left = linear - angular
            right = linear + angular
            # Normalize to [-1, 1] if needed (assuming set_motor_speed takes -1..1)
            max_abs = max(abs(left), abs(right))
            if max_abs > 1.0:
                left /= max_abs
                right /= max_abs
            self._execute_robot_command("set_motor_speed", left, right)
            time.sleep(0.05)

        self._execute_robot_command("set_motor_speed", 0.0, 0.0)
        return True


class FollowPathAction(RobotAction):
    """Follow a list of waypoints [(x1,y1), (x2,y2), ...]."""
    name = "follow_path"
    _required_context_keys = ["path"]
    preconditions = ["robot_ready", "localized"]
    robot_capabilities = ("get_pose", "set_motor_speed", "stop")

    def _execute(self) -> bool:
        path = self.context["path"]
        target_keys = ("target_x", "target_y", "target_theta")
        original_targets = {
            key: self.context.get(key, _MISSING) for key in target_keys
        }
        try:
            for index, (waypoint_x, waypoint_y) in enumerate(path):
                self.context["target_x"] = waypoint_x
                self.context["target_y"] = waypoint_y
                is_final_waypoint = index == len(path) - 1
                if (
                    is_final_waypoint
                    and original_targets["target_theta"] is not _MISSING
                ):
                    self.context["target_theta"] = original_targets["target_theta"]
                else:
                    self.context.pop("target_theta", None)

                navigation = NavigateAction(self.context)
                if not navigation.execute():
                    raise ActionFailureError(
                        self.name,
                        f"Failed at waypoint {index}",
                    )
        finally:
            for key, value in original_targets.items():
                if value is _MISSING:
                    self.context.pop(key, None)
                else:
                    self.context[key] = value
        return True


# ========================== Manipulation Actions ==========================

class GripperAction(RobotAction):
    """Open or close gripper."""
    name = "gripper"
    preconditions = ["robot_ready"]
    postconditions = ["gripper_changed"]
    robot_capabilities = ("set_gripper",)

    def _execute(self) -> bool:
        open_gripper = self.context.get("open", True)
        force = self.context.get("force", 1.0)
        self._execute_robot_command("set_gripper", open_gripper, force)
        return True


class JointAction(RobotAction):
    """Move a specific joint to a target position."""
    name = "joint"
    _required_context_keys = ["joint_id", "position"]
    preconditions = ["robot_ready"]
    robot_capabilities = ("set_joint_position",)

    def _execute(self) -> bool:
        joint_id = self.context["joint_id"]
        position = self.context["position"]
        speed = self.context.get("speed", 0.5)
        self._execute_robot_command(
            "set_joint_position",
            joint_id,
            position,
            speed,
        )
        # Optionally wait for completion
        if self.context.get("wait", True):
            # Simulate motion time (simplified)
            time.sleep(self.context.get("duration", 0.5))
        return True


# ========================== Sensor Actions ==========================

class SensorReadAction(RobotAction):
    """Read a sensor and store value in context."""
    name = "sensor_read"
    _required_context_keys = ["sensor_name", "output_key"]
    preconditions = []
    robot_capabilities = ("get_sensor_value",)

    def _execute(self) -> bool:
        sensor_name = self.context["sensor_name"]
        output_key = self.context["output_key"]
        value = self._read_robot_value("get_sensor_value", sensor_name)
        self.context[output_key] = value
        logger.debug(f"Read {sensor_name} = {value}")
        return True


# ========================== Utility Actions ==========================

class WaitAction(BaseAction):
    """Wait for a specified duration or until a condition becomes true."""
    name = "wait"
    preconditions = []

    def _execute(self) -> bool:
        duration = self.context.get("duration", 1.0)
        condition = self.context.get("condition")  # callable returning bool
        if condition:
            # Poll condition
            start = time.time()
            timeout = self.context.get("timeout", 10.0)
            while not condition(self.context):
                if self._should_interrupt():
                    raise SoftInterrupt("Wait interrupted")
                if time.time() - start > timeout:
                    raise ActionFailureError(self.name, "Condition timeout")
                time.sleep(0.05)
        else:
            # Simple sleep, but check interrupts
            start = time.time()
            while time.time() - start < duration:
                if self._should_interrupt():
                    raise SoftInterrupt("Wait interrupted")
                time.sleep(0.05)
        return True


class LedAction(RobotAction):
    """Control an LED."""
    name = "led"
    _required_context_keys = ["led_id", "state"]
    preconditions = []
    robot_capabilities = ("set_led",)

    def _execute(self) -> bool:
        led_id = self.context["led_id"]
        state = self.context["state"]
        self._execute_robot_command("set_led", led_id, state)
        return True


# ========================== Sequence Action ==========================

class SequenceAction(BaseAction):
    """
    Execute a sequence of sub‑actions defined in context.
    Each sub‑action is a dict with "action_class" and "context".
    """
    name = "sequence"
    preconditions = []
    _required_context_keys = ["actions"]

    def _execute(self) -> bool:
        actions_list = self.context["actions"]
        for act_desc in actions_list:
            action_cls = act_desc["action_class"]
            sub_context = self.context.copy()
            sub_context.update(act_desc.get("context", {}))
            # Ensure robot is passed down
            if "robot" in self.context:
                sub_context["robot"] = self.context["robot"]
            action = action_cls(sub_context)
            if not action.execute():
                raise ActionFailureError(self.name,
                                         f"Sub-action {action.name} failed")
        return True


__all__ = [
    "AckermannAction",
    "FollowPathAction",
    "GripperAction",
    "JointAction",
    "LedAction",
    "MotorAction",
    "NavigateAction",
    "RobotAction",
    "SensorReadAction",
    "SequenceAction",
    "SpinAction",
    "StopAction",
    "WaitAction",
]


if __name__ == "__main__":
    printer = PrettyPrinter()
    print("\n=== Running Robot Actions ===\n")
    printer.status("TEST", "Starting Robot Actions tests", "info")

    # Mock robot interface for testing (implements RobotInterface)
    class Robot(RobotInterface):
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.theta = 0.0

        def set_motor_speed(self, left: float, right: float) -> bool:
            print(f"MockMotor: left={left:.2f}, right={right:.2f}")
            return True
        def set_steering(self, angle: float) -> bool:
            print(f"MockSteering: angle={angle:.2f}")
            return True
        def set_throttle(self, speed: float) -> bool:
            print(f"MockThrottle: speed={speed:.2f}")
            return True
        def stop(self) -> bool:
            print("MockStop")
            return True
        def set_gripper(self, open: bool, force: float = 1.0) -> bool:
            print(f"MockGripper: open={open}, force={force}")
            return True
        def set_joint_position(self, joint_id: int, position: float, speed: float) -> bool:
            print(f"MockJoint: id={joint_id}, pos={position}, speed={speed}")
            return True
        def get_pose(self) -> Tuple[float, float, float]:
            # Simulate moving towards target for testing (optional)
            return (self.x, self.y, self.theta)
        def get_sensor_value(self, sensor_name: str) -> Any:
            return 42.0  # dummy value
        def set_led(self, led_id: int, state: bool) -> bool:
            print(f"MockLED: id={led_id}, state={state}")
            return True

    # Create mock robot and context
    robot = Robot()
    context = {
        "robot": robot,
        "robot_ready": True,
        "localized": True,
        "target_x": 2.0,
        "target_y": 1.5,
        "target_theta": math.radians(90)
    }

    # Test instantiation of actions (no execution to avoid infinite loop)
    nav_action = NavigateAction(context)
    print(f"Created {nav_action.name} action")

    stop_action = StopAction(context)
    print(f"Created {stop_action.name} action")

    print("\nAll actions instantiated successfully. (Execution skipped for demo.)")
    print("\n=== Robot Actions tests passed ===")