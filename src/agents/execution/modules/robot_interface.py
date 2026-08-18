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

    @abstractmethod
    def get_battery_level(self) -> float:
        """Return battery level as a percentage."""
        pass

    @abstractmethod
    def get_joint_position(self, joint_id: int) -> float:
        """Return current position of a joint."""
        pass

    @abstractmethod
    def get_joint_velocity(self, joint_id: int) -> float:
        """Return current velocity of a joint."""
        pass

    @abstractmethod
    def get_joint_torque(self, joint_id: int) -> float:
        """Return current torque of a joint."""
        pass

    @abstractmethod
    def get_all_joint_states(self) -> Dict[int, Dict[str, float]]:
        """Return a dictionary of all joint states, keyed by joint_id."""
        pass

    @abstractmethod
    def reset(self) -> bool:
        """Reset the robot to a safe state."""
        pass

    @abstractmethod
    def is_operational(self) -> bool:
        """Check if the robot is operational."""
        pass

    @abstractmethod
    def execute_trajectory(self, trajectory: Any) -> bool:
        """Execute a given trajectory."""
        pass

    @abstractmethod
    def get_camera_image(self, camera_id: int) -> Any:
        """Capture an image from a specified camera."""
        pass

    @abstractmethod
    def get_lidar_scan(self) -> Any:
        """Return a LIDAR scan."""
        pass

    @abstractmethod
    def get_ultrasonic_distance(self, sensor_id: int) -> float:
        """Return distance reading from an ultrasonic sensor."""
        pass

    @abstractmethod
    def get_tof_distance(self, sensor_id: int) -> float:
        """Return distance reading from a time-of-flight sensor e.g., VL53L0X."""
        pass

    @abstractmethod
    def get_imu_data(self) -> Dict[str, float]:
        """Return IMU data (accelerometer, gyroscope, magnetometer)."""
        pass

    @abstractmethod
    def get_temperature(self, sensor_id: int) -> float:
        """Return temperature reading from a specified sensor."""
        pass

    @abstractmethod
    def get_humidity(self, sensor_id: int) -> float:    
        """Return humidity reading from a specified sensor."""
        pass

    @abstractmethod
    def get_pressure(self, sensor_id: int) -> float:
        """Return pressure reading from a specified sensor."""
        pass

    @abstractmethod
    def get_gps_coordinates(self) -> Tuple[float, float]:
        """Return GPS coordinates (latitude, longitude)."""
        pass

    @abstractmethod
    def get_altitude(self) -> float:
        """Return altitude reading from a specified sensor."""
        pass

    @abstractmethod
    def get_magnetometer_data(self) -> Dict[str, float]:
        """Return magnetometer data."""
        pass