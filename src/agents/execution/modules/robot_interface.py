"""Hardware boundary contract for execution-agent robot adapters.

The interface contains the stable v2.2 actuation/pose surface required by the
existing robot actions. Additional telemetry and sensor methods are optional
capabilities: adapters implement only those supported by their hardware. Action
classes validate the exact capabilities they need before issuing commands.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NoReturn


class RobotInterface(ABC):
    """Abstract boundary for simulated or physical robot adapters.

    Commands are synchronous and return ``True`` only when the adapter accepts
    the command. Implementations are responsible for documenting units, valid
    ranges, frame conventions, and whether acceptance implies completion.
    """

    def _unsupported(self, capability: str) -> NoReturn:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement robot capability {capability!r}"
        )

    def supports_capability(self, capability: str) -> bool:
        """Return whether this adapter implements a named robot operation."""
        implementation = getattr(type(self), str(capability), None)
        placeholder = getattr(RobotInterface, str(capability), None)
        return callable(implementation) and implementation is not placeholder

    @abstractmethod
    def set_motor_speed(self, left: float, right: float) -> bool:
        """Set left/right wheel speeds (differential drive)."""
        raise NotImplementedError

    @abstractmethod
    def set_steering(self, angle: float) -> bool:
        """Set steering angle (Ackermann)."""
        raise NotImplementedError

    @abstractmethod
    def set_throttle(self, speed: float) -> bool:
        """Set throttle (Ackermann)."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> bool:
        """Emergency stop."""
        raise NotImplementedError

    @abstractmethod
    def set_gripper(self, open: bool, force: float = 1.0) -> bool:
        """Control gripper."""
        raise NotImplementedError

    @abstractmethod
    def set_joint_position(self, joint_id: int, position: float, speed: float) -> bool:
        """Move a joint to target position."""
        raise NotImplementedError

    @abstractmethod
    def get_pose(self) -> tuple[float, float, float]:
        """Return ``(x, y, theta)`` in the adapter's documented world frame."""
        raise NotImplementedError

    @abstractmethod
    def get_sensor_value(self, sensor_name: str) -> Any:
        """Read a sensor by name."""
        raise NotImplementedError

    @abstractmethod
    def set_led(self, led_id: int, state: bool) -> bool:
        """Control an LED."""
        raise NotImplementedError

    def get_battery_level(self) -> float:
        """Return battery level as a percentage when supported."""
        self._unsupported("get_battery_level")

    def get_joint_position(self, joint_id: int) -> float:
        """Return current position of a joint when supported."""
        self._unsupported("get_joint_position")

    def get_joint_velocity(self, joint_id: int) -> float:
        """Return current velocity of a joint when supported."""
        self._unsupported("get_joint_velocity")

    def get_joint_torque(self, joint_id: int) -> float:
        """Return current torque of a joint when supported."""
        self._unsupported("get_joint_torque")

    def get_all_joint_states(self) -> dict[int, dict[str, float]]:
        """Return all joint states, keyed by joint identifier, when supported."""
        self._unsupported("get_all_joint_states")

    def reset(self) -> bool:
        """Reset the robot to a safe state when supported."""
        self._unsupported("reset")

    def is_operational(self) -> bool:
        """Return the adapter's operational state when supported."""
        self._unsupported("is_operational")

    def execute_trajectory(self, trajectory: Any) -> bool:
        """Execute a trajectory when supported."""
        self._unsupported("execute_trajectory")

    def get_camera_image(self, camera_id: int) -> Any:
        """Capture an image from a specified camera when supported."""
        self._unsupported("get_camera_image")

    def get_lidar_scan(self) -> Any:
        """Return a LiDAR scan when supported."""
        self._unsupported("get_lidar_scan")

    def get_ultrasonic_distance(self, sensor_id: int) -> float:
        """Return ultrasonic distance when supported."""
        self._unsupported("get_ultrasonic_distance")

    def get_tof_distance(self, sensor_id: int) -> float:
        """Return time-of-flight distance when supported."""
        self._unsupported("get_tof_distance")

    def get_imu_data(self) -> dict[str, float]:
        """Return IMU data when supported."""
        self._unsupported("get_imu_data")

    def get_temperature(self, sensor_id: int) -> float:
        """Return temperature when supported."""
        self._unsupported("get_temperature")

    def get_humidity(self, sensor_id: int) -> float:
        """Return humidity when supported."""
        self._unsupported("get_humidity")

    def get_pressure(self, sensor_id: int) -> float:
        """Return pressure when supported."""
        self._unsupported("get_pressure")

    def get_gps_coordinates(self) -> tuple[float, float]:
        """Return ``(latitude, longitude)`` when supported."""
        self._unsupported("get_gps_coordinates")

    def get_altitude(self) -> float:
        """Return altitude when supported."""
        self._unsupported("get_altitude")

    def get_magnetometer_data(self) -> dict[str, float]:
        """Return magnetometer data when supported."""
        self._unsupported("get_magnetometer_data")

    def get_proximity_data(self) -> dict[str, float]:
        """Return proximity sensor data when supported."""
        self._unsupported("get_proximity_data")

    def get_wheel_encoder_counts(self) -> dict[str, int]:
        """Return wheel encoder counts when supported."""
        self._unsupported("get_wheel_encoder_counts")

    def get_motor_currents(self) -> dict[str, float]:
        """Return motor currents when supported."""
        self._unsupported("get_motor_currents")

    def get_servo_positions(self) -> dict[str, float]:
        """Return servo positions when supported."""
        self._unsupported("get_servo_positions")

    def get_servo_velocities(self) -> dict[str, float]:
        """Return servo velocities when supported."""
        self._unsupported("get_servo_velocities")

    def get_servo_torques(self) -> dict[str, float]:
        """Return servo torques when supported."""
        self._unsupported("get_servo_torques")

    def get_servo_states(self) -> dict[str, dict[str, float]]:
        """Return all servo states when supported."""
        self._unsupported("get_servo_states")

    def get_joint_states(self) -> dict[int, dict[str, float]]:
        """Return all joint states when supported."""
        self._unsupported("get_joint_states")

    def get_actuator_states(self) -> dict[str, dict[str, float]]:
        """Return all actuator states when supported."""
        self._unsupported("get_actuator_states")

    def get_esc_states(self) -> dict[str, dict[str, float]]:
        """Return all ESC states when supported."""
        self._unsupported("get_esc_states")

    def get_bms_data(self) -> dict[str, float]:
        """Return battery management system data when supported."""
        self._unsupported("get_bms_data")

    def get_power_distribution_data(self) -> dict[str, float]:
        """Return power distribution data when supported."""
        self._unsupported("get_power_distribution_data")

    def get_system_health(self) -> dict[str, Any]:
        """Return system health metrics when supported."""
        self._unsupported("get_system_health")

    def get_diagnostics(self) -> dict[str, Any]:
        """Return diagnostics information when supported."""
        self._unsupported("get_diagnostics")

    def get_error_logs(self) -> list[str]:
        """Return error logs when supported."""
        self._unsupported("get_error_logs")

    def get_event_logs(self) -> list[str]:
        """Return event logs when supported."""
        self._unsupported("get_event_logs")

    def get_system_time(self) -> float:
        """Return system time in seconds when supported."""
        self._unsupported("get_system_time")

    def get_firmware_version(self) -> str:
        """Return firmware version when supported."""
        self._unsupported("get_firmware_version")

    def get_hardware_version(self) -> str:
        """Return hardware version when supported."""
        self._unsupported("get_hardware_version")

    def get_serial_number(self) -> str:
        """Return serial number when supported."""
        self._unsupported("get_serial_number")

    def get_calibration_data(self) -> dict[str, float]:
        """Return calibration data when supported."""
        self._unsupported("get_calibration_data")


__all__ = ["RobotInterface"]
