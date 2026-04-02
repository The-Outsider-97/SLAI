from typing import Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod

class RobotInterface(ABC):
    """Abstract interface for robot hardware (simulated or real)."""

    @abstractmethod
    def set_motor_speed(self, left: float, right: float) -> bool:
        """Set left/right wheel speeds (differential drive)."""
        pass

    @abstractmethod
    def set_steering(self, angle: float) -> bool:
        """Set steering angle (Ackermann)."""
        pass

    @abstractmethod
    def set_throttle(self, speed: float) -> bool:
        """Set throttle (Ackermann)."""
        pass

    @abstractmethod
    def stop(self) -> bool:
        """Emergency stop."""
        pass

    @abstractmethod
    def set_gripper(self, open: bool, force: float = 1.0) -> bool:
        """Control gripper."""
        pass

    @abstractmethod
    def set_joint_position(self, joint_id: int, position: float, speed: float) -> bool:
        """Move a joint to target position."""
        pass

    @abstractmethod
    def get_pose(self) -> Tuple[float, float, float]:
        """Return (x, y, theta) in world frame."""
        pass

    @abstractmethod
    def get_sensor_value(self, sensor_name: str) -> Any:
        """Read a sensor by name."""
        pass

    @abstractmethod
    def set_led(self, led_id: int, state: bool) -> bool:
        """Control an LED."""
        pass