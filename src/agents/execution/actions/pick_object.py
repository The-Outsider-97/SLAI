import math
import time
import random

from typing import Dict, Any, Optional, Tuple

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.execution_error import ActionFailureError, InvalidContextError, UnreachableTargetError
from ..actions.base_action import BaseAction, ActionStatus, SoftInterrupt
from ..actions.robot_actions import RobotAction
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Pick Object Action")
printer = PrettyPrinter

class PickObjectAction(RobotAction):
    """
    Pick an object using a robotic gripper and optional arm movement.
    Requires robot interface with set_gripper(), set_joint_position(), get_pose().
    """
    name = "pick_object"
    priority = 5
    preconditions = ["robot_ready", "hand_empty"]
    postconditions = ["holding_object", "hand_empty"]  # hand_empty becomes False
    _required_context_keys = ["object_position", "object_properties", "target_object"]

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        self.config = load_global_config()
        self.pick_object = get_config_section("pick_object_action") or {}

        # Configuration with defaults
        self.grasp_time = self.pick_object.get("grasp_time", 1.0)
        self.min_distance = self.pick_object.get("min_distance", 0.5)
        self.base_success_rate = self.pick_object.get("base_success_rate", 0.95)
        self.energy_cost = self.pick_object.get("energy_cost", 0.2)
        self.max_weight = self.pick_object.get("max_weight", 5.0)
        self.max_size = self.pick_object.get("max_size", 1.0)
        self.arm_prepick_joint = self.pick_object.get("arm_prepick_joint", None)  # e.g., {"joint_id": 2, "angle": 0.5}
        self.arm_pick_joint = self.pick_object.get("arm_pick_joint", None)

        # Internal state
        self.object_id: str = ""
        self.object_properties: Dict[str, Any] = {}
        self.grasp_progress: float = 0.0
        self._pick_start_time: float = 0.0

    def _execute(self) -> bool:
        """Execute the pick sequence with validation, arm positioning, and gripper closure."""
        logger.info(f"Starting pick action for '{self.context.get('target_object')}'")

        # Validate object and proximity
        self._validate_object()
        self._check_proximity()

        # Pre‑pick arm positioning (if configured)
        if self.arm_prepick_joint:
            self._move_arm(self.arm_prepick_joint, "pre‑pick")

        # Perform grasp
        success = self._perform_grasp_sequence()

        # Post‑pick arm retraction (optional)
        if success and self.arm_pick_joint:
            self._move_arm(self.arm_pick_joint, "pick")

        return self._finalize_pick(success)

    def _validate_object(self) -> None:
        """Validate object properties and raise appropriate errors."""
        self.object_id = self.context.get("target_object", "unknown")
        self.object_properties = self.context.get("object_properties", {})

        required = ["weight", "size", "grasp_difficulty"]
        missing = [p for p in required if p not in self.object_properties]
        if missing:
            raise InvalidContextError(self.name, missing)

        weight = self.object_properties["weight"]
        if weight > self.max_weight:
            raise ActionFailureError(self.name, f"Object too heavy ({weight} > {self.max_weight})")

        size = self.object_properties["size"]
        if size > self.max_size:
            raise ActionFailureError(self.name, f"Object too large ({size} > {self.max_size})")

    def _check_proximity(self) -> None:
        """Ensure robot is close enough to object using robot.get_pose()."""
        try:
            robot_pose = self.robot.get_pose()  # (x, y, theta)
        except AttributeError:
            # Fallback: use context position if robot lacks get_pose
            robot_pose = self.context.get("current_position", (0.0, 0.0))
            if len(robot_pose) == 2:
                robot_pose = (robot_pose[0], robot_pose[1], 0.0)

        obj_pos = self.context.get("object_position")
        if not obj_pos:
            raise InvalidContextError(self.name, ["object_position"])

        dx = robot_pose[0] - obj_pos[0]
        dy = robot_pose[1] - obj_pos[1]
        distance = math.hypot(dx, dy)
        if distance > self.min_distance:
            raise UnreachableTargetError(self.name, obj_pos, robot_pose[:2])

    def _move_arm(self, joint_config: Dict[str, Any], stage: str) -> None:
        """Move a joint to a specified angle."""
        joint_id = joint_config.get("joint_id")
        angle = joint_config.get("angle")
        speed = joint_config.get("speed", 0.5)
        if joint_id is None or angle is None:
            logger.warning(f"Invalid arm config for {stage}: {joint_config}")
            return
        if not self.robot.set_joint_position(joint_id, angle, speed):
            raise ActionFailureError(self.name, f"Arm movement failed during {stage}")

    def _perform_grasp_sequence(self) -> bool:
        """Simulate or actually close gripper with timing and failure chance."""
        difficulty = self.object_properties.get("grasp_difficulty", 0.0)
        grasp_duration = self.grasp_time * (1.0 + difficulty)
        self._pick_start_time = time.time()
        self.grasp_progress = 0.0

        # Random failure check before starting
        if self._should_fail():
            logger.warning("Grasp attempt failed (pre‑failure)")
            return False

        # Close gripper (actual hardware command)
        if not self.robot.set_gripper(open=False, force=self.object_properties.get("weight", 1.0)):
            raise ActionFailureError(self.name, "Failed to close gripper")

        # Wait for grasp duration with progress updates
        while self.grasp_progress < 1.0:
            if self._should_interrupt():
                self.robot.set_gripper(open=True)  # release
                raise SoftInterrupt("Pick interrupted")

            elapsed = time.time() - self._pick_start_time
            self.grasp_progress = min(1.0, elapsed / grasp_duration)
            self._update_grasp_status()
            time.sleep(0.05)

            # Mid‑grasp failure check
            if self._should_fail():
                self.robot.set_gripper(open=True)
                return False

        return True

    def _should_fail(self) -> bool:
        """Determine if grasp should fail based on difficulty and random chance."""
        difficulty = self.object_properties.get("grasp_difficulty", 0.0)
        failure_chance = (1.0 - self.base_success_rate) * (1.0 + difficulty)
        # Critical failure (e.g., object slips)
        if random.random() < 0.01 * (1.0 + difficulty):
            logger.error("Critical grasp failure: object dropped")
            return True
        return random.random() < failure_chance

    def _update_grasp_status(self) -> None:
        """Update action status based on grasp progress."""
        if self.grasp_progress < 0.3:
            self.status = ActionStatus.ACCELERATING
        elif self.grasp_progress < 0.7:
            self.status = ActionStatus.RUNNING
        else:
            self.status = ActionStatus.DECELERATING

    def _finalize_pick(self, success: bool) -> bool:
        """Update context and inventory after pick attempt."""
        self._consume_energy()

        if not success:
            logger.error(f"Failed to pick {self.object_id}")
            return False

        # Update inventory
        inventory = self.context.setdefault("inventory", {})
        inventory[self.object_id] = {
            "type": self.object_properties.get("type", "unknown"),
            "weight": self.object_properties["weight"],
            "size": self.object_properties["size"]
        }
        self.context["carrying_items"] = len(inventory)
        self.set_carry_capacity(self.context["carrying_items"])

        # Update state flags
        self.context["holding_object"] = True
        self.context["held_object"] = self.object_id
        self.context["hand_empty"] = False
        self.context.setdefault("object_state", {})[self.object_id] = "held"

        logger.info(f"Successfully picked up {self.object_id}")
        return True

    def _consume_energy(self) -> None:
        """Deduct energy based on object weight and difficulty."""
        if "energy" in self.context:
            weight_factor = self.object_properties["weight"] / self.max_weight
            difficulty = self.object_properties.get("grasp_difficulty", 0.0)
            consumption = self.energy_cost * (1.0 + weight_factor + difficulty)
            self.context["energy"] = max(0.0, self.context["energy"] - consumption)

    def _should_interrupt(self) -> bool:
        """Extended interrupt checks including energy depletion."""
        if super()._should_interrupt():
            return True
        if self.context.get("energy", 10.0) <= 0:
            logger.warning("Pick interrupted: energy depleted")
            return True
        return False

if __name__ == "__main__":
    print("\n=== Running Execution PICK_OBJECT Action ===\n")
    printer.status("TEST", "Starting Execution PICK_OBJECT Action tests", "info")
    from ..utils.robot_interface import RobotInterface

    # Mock robot interface
    class MockRobot(RobotInterface):
        def set_motor_speed(self, left: float, right: float) -> bool:
            print(f"MockMotor: left={left}, right={right}")
            return True
        def set_steering(self, angle: float) -> bool:
            print(f"MockSteering: angle={angle}")
            return True
        def set_throttle(self, speed: float) -> bool:
            print(f"MockThrottle: speed={speed}")
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
            return (1.1, 1.1, 0.0)  # close to object at (1.0, 1.0)
        def get_sensor_value(self, sensor_name: str) -> Any:
            return 42.0
        def set_led(self, led_id: int, state: bool) -> bool:
            print(f"MockLED: id={led_id}, state={state}")
            return True

    # Test context with robot
    context = {
        "robot": MockRobot(),
        "robot_ready": True,
        "hand_empty": True,
        "target_object": "apple",
        "object_position": (1.0, 1.0),
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

    print("\n* * * * * Phase 2 - Execute * * * * *\n")
    try:
        success = pick_action.execute()
        printer.pretty("EXECUTION", "SUCCESS" if success else "FAILURE",
                       "success" if success else "error")
    except Exception as e:
        printer.pretty("EXECUTION", f"Exception: {e}", "error")

    print("\n=== All tests completed successfully! ===\n")