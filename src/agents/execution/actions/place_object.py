import math
import time

from typing import Dict, Any, Optional

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.execution_error import ActionFailureError, InvalidContextError, UnreachableTargetError
from ..actions.base_action import ActionStatus, SoftInterrupt
from ..actions.robot_actions import RobotAction
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Place Object Action")
printer = PrettyPrinter

class PlaceObjectAction(RobotAction):
    """
    Place a held object at a target position using gripper and optional arm.
    """
    name = "place_object"
    priority = 4
    preconditions = ["robot_ready", "holding_object"]
    postconditions = ["hand_empty", "object_placed"]
    _required_context_keys = ["place_position", "held_object"]

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(context)
        cfg = get_config_section("place_object_action") or {}

        self.place_time = cfg.get("place_time", 1.5)
        self.min_distance = cfg.get("min_distance", 0.5)
        self.energy_cost = cfg.get("energy_cost", 0.1)
        self.arm_preplace_joint = cfg.get("arm_preplace_joint", None)
        self.arm_release_joint = cfg.get("arm_release_joint", None)

        self.object_id: str = ""
        self.place_progress: float = 0.0
        self._place_start_time: float = 0.0

    def _execute(self) -> bool:
        """Execute place sequence: validate, move arm, open gripper."""
        logger.info(f"Starting place action for '{self.context.get('held_object')}'")

        self._validate_state()
        self._check_proximity()

        # Optional arm positioning before release
        if self.arm_preplace_joint:
            self._move_arm(self.arm_preplace_joint, "pre‑place")

        success = self._perform_place_sequence()

        # Optional arm retract after release
        if success and self.arm_release_joint:
            self._move_arm(self.arm_release_joint, "post‑place")

        return self._finalize_place(success)

    def _validate_state(self) -> None:
        """Ensure robot is holding an object."""
        self.object_id = self.context.get("held_object")
        if not self.object_id:
            raise InvalidContextError(self.name, ["held_object"])
        logger.info(f"Placing object: {self.object_id}")

    def _check_proximity(self) -> None:
        """Check distance to place position."""
        try:
            robot_pose = self.robot.get_pose()
        except AttributeError:
            robot_pose = self.context.get("current_position", (0.0, 0.0))
            if len(robot_pose) == 2:
                robot_pose = (robot_pose[0], robot_pose[1], 0.0)

        place_pos = self.context.get("place_position")
        if not place_pos:
            raise InvalidContextError(self.name, ["place_position"])

        dx = robot_pose[0] - place_pos[0]
        dy = robot_pose[1] - place_pos[1]
        distance = math.hypot(dx, dy)
        if distance > self.min_distance:
            raise UnreachableTargetError(self.name, place_pos, robot_pose[:2])

    def _move_arm(self, joint_config: Dict[str, Any], stage: str) -> None:
        """Move a joint to specified angle."""
        joint_id = joint_config.get("joint_id")
        angle = joint_config.get("angle")
        speed = joint_config.get("speed", 0.5)
        if joint_id is None or angle is None:
            logger.warning(f"Invalid arm config for {stage}: {joint_config}")
            return
        if not self.robot.set_joint_position(joint_id, angle, speed):
            raise ActionFailureError(self.name, f"Arm movement failed during {stage}")

    def _perform_place_sequence(self) -> bool:
        """Open gripper and wait for release duration."""
        # Open gripper
        if not self.robot.set_gripper(open=True):
            raise ActionFailureError(self.name, "Failed to open gripper")

        self._place_start_time = time.time()
        self.place_progress = 0.0

        while self.place_progress < 1.0:
            if self._should_interrupt():
                raise SoftInterrupt("Place interrupted")

            elapsed = time.time() - self._place_start_time
            self.place_progress = min(1.0, elapsed / self.place_time)
            self._update_place_status()
            time.sleep(0.05)

        return True

    def _update_place_status(self) -> None:
        """Update action status based on progress."""
        if self.place_progress < 0.5:
            self.status = ActionStatus.DECELERATING
        else:
            self.status = ActionStatus.RUNNING

    def _finalize_place(self, success: bool) -> bool:
        """Update context and inventory after placement."""
        self._consume_energy()

        if not success:
            logger.error(f"Failed to place {self.object_id}")
            return False

        # Remove from inventory
        inventory = self.context.get("inventory", {})
        if self.object_id in inventory:
            del inventory[self.object_id]

        self.context["carrying_items"] = len(inventory)
        self.set_carry_capacity(self.context["carrying_items"])

        # Update state flags
        self.context["holding_object"] = False
        self.context["held_object"] = None
        self.context["hand_empty"] = True
        self.context["object_placed"] = True
        self.context.setdefault("object_state", {})[self.object_id] = "placed"

        logger.info(f"Successfully placed {self.object_id}")
        return True

    def _consume_energy(self) -> None:
        """Deduct energy for placing."""
        if "energy" in self.context:
            self.context["energy"] = max(0.0, self.context["energy"] - self.energy_cost)

    def _should_interrupt(self) -> bool:
        """Check for interruption signals."""
        if super()._should_interrupt():
            return True
        if self.context.get("energy", 10.0) <= 0:
            logger.warning("Place interrupted: energy depleted")
            return True
        return False

if __name__ == "__main__":
    print("\n=== Running Execution PLACE_OBJECT Action ===\n")
    printer.status("TEST", "Starting Execution PLACE_OBJECT Action tests", "info")

    from ..utils.robot_interface import RobotInterface
    from typing import Any, Tuple

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
            return (2.1, 2.1, 0.0)  # close to place position (2.0, 2.0)
        def get_sensor_value(self, sensor_name: str) -> Any:
            return 42.0
        def set_led(self, led_id: int, state: bool) -> bool:
            print(f"MockLED: id={led_id}, state={state}")
            return True

    context = {
        "robot": MockRobot(),
        "robot_ready": True,
        "holding_object": True,
        "held_object": "apple",
        "place_position": (2.0, 2.0),
        "inventory": {"apple": {"weight": 0.3, "size": 0.1}},
        "energy": 5.0
    }

    place_action = PlaceObjectAction(context)
    print(f"{place_action}")

    print("\n* * * * * Phase 2 - Execution * * * * *\n")

    try:
        success = place_action.execute()
        printer.pretty("EXECUTION", "SUCCESS" if success else "FAILURE",
                       "success" if success else "error")
    except Exception as e:
        printer.pretty("EXECUTION", f"Exception: {e}", "error")

    print("\n* * * * * Phase 3 - Final Context * * * * *\n")
    printer.pretty("CONTEXT", context, "info")

    print("\n=== All tests completed successfully! ===\n")