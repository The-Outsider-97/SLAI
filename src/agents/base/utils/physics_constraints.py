"""
Physics constraints and environmental effects for the SLAI Environment.

This module provides functions to simulate physical phenomena such as
gravity, friction, boundary collisions, relativistic corrections, and quantum effects.
The state vector is assumed to have components at specific indices:

- state[0]: x-position
- state[1]: y-position
- state[2]: x-velocity (if state_dim >= 3)
- state[3]: y-velocity (if state_dim >= 4)
- state[4]: angle (radians) (if state_dim >= 5)
- state[5]: angular velocity (rad/s) (if state_dim >= 6)
- state[6]: moment of inertia (if state_dim >= 7)
- state[7]: electric charge (if state_dim >= 8)

All physics functions are designed to work with the SLAIEnv class, but can be used
independently given a compatible state vector and configuration.
"""

import numpy as np

from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field

from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Physics Constraints")
printer = PrettyPrinter


@dataclass
class PhysicsConfig:
    """Configuration parameters for the physics engine."""
    # Gravity and environment
    gravity: float = 9.80665          # m/s^2
    friction_coeff: float = 0.02      # linear friction (dimensionless, per second)
    rotational_friction: float = 0.01 # angular damping (dimensionless, per second)

    # Wind
    wind_strength: float = 0.0        # m/s
    wind_direction: float = 0.0       # radians (0 = east, π/2 = north)

    # Drag (quadratic)
    drag_coeff: float = 0.01          # drag coefficient (kg/m)
    terminal_velocity: float = 50.0   # m/s

    # Collisions
    elasticity: float = 0.8           # coefficient of restitution (0 = inelastic, 1 = elastic)

    # Particle properties (used when not provided in state)
    default_mass: float = 1.0         # kg
    default_charge: float = 0.0       # C

    # Quantum tunneling (simplified model)
    enable_tunneling: bool = False
    tunneling_probability: float = 0.05
    barrier_positions: Tuple[float, ...] = (-8.0, 8.0)
    barrier_width: float = 0.1

    # Relativistic effects
    enable_relativistic: bool = True
    relativistic_threshold: float = 0.1  # fraction of c

    # Electromagnetic effects
    enable_electromagnetic: bool = False
    electric_field: Tuple[float, float] = (0.0, 10.0)  # V/m
    magnetic_field: float = 0.5          # Tesla

    # Time step (will be overridden by env.dt)
    dt: float = 0.02

    # Boundary thresholds (relative to observation space)
    boundary_margin: float = 0.01
    corner_threshold: float = 0.05


class PhysicsEngine:
    """
    Centralized physics engine for the SLAI Environment.
    Applies gravity, friction, drag, wind, collisions, and optional advanced effects.
    """

    # Physical constants (from https://en.wikipedia.org/wiki/List_of_physical_constants)
    CONSTANTS = {
        # Gravitational and related constants
        "G": 6.67430e-11,                     # m^3·kg^−1·s^−2
        "g": 9.80665,                         # m/s^2
        "lambda": 1.1056e-52,                 # 1/m^2

        # Electromagnetic constants
        "epsilon0": 8.8541878128e-12,         # F/m
        "mu0": 1.25663706212e-6,              # N/A^2
        "ke": 8.9875517923e9,                 # N·m^2·C^−2
        "c": 299792458,                       # m/s
        "e": 1.602176634e-19,                 # C
        "alpha": 7.2973525693e-3,             # dimensionless
        "phi0": 2.067833848e-15,              # Wb

        # Thermodynamic constants
        "kB": 1.380649e-23,                   # J/K
        "R": 8.31446261815324,                # J/mol·K
        "NA": 6.02214076e23,                  # mol^−1
        "F": 96485.3321233100184,             # C/mol
        "sigma": 5.670374419e-8,              # W/m²·K⁴
        "kW": 2.897771955e-3,                 # m·K

        # Quantum constants
        "h": 6.62607015e-34,                  # J·s
        "hbar": 1.054571817e-34,              # J·s

        # Particle masses
        "me": 9.1093837015e-31,               # kg
        "mp": 1.67262192369e-27,              # kg
        "mn": 1.67492749804e-27,              # kg
        "u": 1.66053906660e-27,               # kg

        # Time and frequency
        "Hz": 1.0,                            # s^−1
        "day": 86400,                         # s
        "year": 31557600,                     # s

        # Temperature and pressure
        "T0": 273.15,                         # K
        "P0": 101325,                         # Pa

        # Other constants
        "Z0": 376.730313668,                  # ohm
        "lP": 1.616255e-35,                   # m
        "tP": 5.391247e-44,                   # s
        "mP": 2.176434e-8,                    # kg
        "TP": 1.416784e32,                    # K
    }

    def __init__(self, config: Optional[Union[PhysicsConfig, Dict]] = None):
        """
        Initialize the physics engine.

        Args:
            config: PhysicsConfig instance or dictionary of parameters.
        """
        if config is None:
            self.config = PhysicsConfig()
        elif isinstance(config, dict):
            self.config = PhysicsConfig(**config)
        else:
            self.config = config

        self.constants = self.CONSTANTS.copy()
        self._c = self.constants["c"]
        self._hbar = self.constants["hbar"]

    def apply_environmental_effects(self, state: np.ndarray, dt: float,
                                    mass: Optional[float] = None) -> np.ndarray:
        """
        Apply environmental effects (gravity, friction, wind, drag, relativistic corrections, etc.).

        Args:
            state: The state vector (modified in-place).
            dt: Time step (seconds).
            mass: Mass of the object (kg). If None, uses config.default_mass.

        Returns:
            The modified state vector (same array, for chaining).
        """
        if mass is None:
            mass = self.config.default_mass

        # Ensure dt is positive
        if dt <= 0:
            return state

        # Helper to get velocity components safely
        has_vel2d = len(state) >= 4
        has_vel1d = len(state) >= 3
        has_angle = len(state) >= 5
        has_ang_vel = len(state) >= 6
        has_charge = len(state) >= 8

        # 1. Gravity (affects y-velocity)
        if has_vel2d:
            state[3] -= self.config.gravity * dt

        # 2. Linear friction (viscous damping)
        if has_vel1d:
            state[2] *= (1 - self.config.friction_coeff * dt)
        if has_vel2d:
            state[3] *= (1 - self.config.friction_coeff * dt)

        # 3. Wind forces (directional + turbulence)
        if has_vel2d and self.config.wind_strength > 0:
            wind_x = self.config.wind_strength * np.cos(self.config.wind_direction)
            wind_y = self.config.wind_strength * np.sin(self.config.wind_direction)
            turbulence = np.random.normal(0, 0.1 * self.config.wind_strength, 2)
            state[2] += (wind_x + turbulence[0]) * dt
            state[3] += (wind_y + turbulence[1]) * dt

        # 4. Quadratic drag (air resistance)
        if has_vel2d and self.config.drag_coeff > 0:
            vx, vy = state[2], state[3]
            speed = np.hypot(vx, vy)
            if speed > 0.01:
                drag_magnitude = self.config.drag_coeff * speed ** 2
                drag_x = -drag_magnitude * vx / speed
                drag_y = -drag_magnitude * vy / speed
                state[2] += drag_x * dt
                state[3] += drag_y * dt

        # 5. Terminal velocity limit
        if has_vel2d and self.config.terminal_velocity > 0:
            vx, vy = state[2], state[3]
            speed = np.hypot(vx, vy)
            if speed > self.config.terminal_velocity:
                scale = self.config.terminal_velocity / speed
                state[2] *= scale
                state[3] *= scale

        # 6. Rotational effects
        if has_ang_vel:
            state[5] *= (1 - self.config.rotational_friction * dt)

        # 7. Relativistic corrections (Lorentz factor) at high speeds
        if self.config.enable_relativistic and has_vel2d:
            vx, vy = state[2], state[3]
            speed = np.hypot(vx, vy)
            if speed > self.config.relativistic_threshold * self._c:
                gamma = 1 / np.sqrt(1 - (speed ** 2) / (self._c ** 2))
                state[2] *= gamma
                state[3] *= gamma

        # 8. Quantum tunneling (simplified)
        if self.config.enable_tunneling and has_vel1d:
            for barrier in self.config.barrier_positions:
                if abs(state[0] - barrier) < self.config.barrier_width:
                    if np.random.random() < self.config.tunneling_probability:
                        state[0] = barrier + self.config.barrier_width * np.sign(state[2] if abs(state[2]) > 0 else 1)

        # 9. Electromagnetic forces
        if self.config.enable_electromagnetic and has_charge and has_vel2d:
            charge = state[7] if has_charge else self.config.default_charge
            if abs(charge) > 1e-9:
                # Electric force
                Ex, Ey = self.config.electric_field
                state[2] += (charge * Ex * dt / mass)
                state[3] += (charge * Ey * dt / mass)
                # Magnetic force (Lorentz) - simplified: perpendicular acceleration
                B = self.config.magnetic_field
                vx, vy = state[2], state[3]
                state[2] += (charge * B * vy * dt / mass)
                state[3] -= (charge * B * vx * dt / mass)

        return state

    def enforce_boundary_constraints(self, state: np.ndarray,
                                     low_bound: np.ndarray, high_bound: np.ndarray) -> np.ndarray:
        """
        Enforce boundary collisions (walls, ground, ceiling) with inelastic bouncing.

        Args:
            state: The state vector (modified in-place).
            low_bound: Lower bounds of the observation space (x, y, ...).
            high_bound: Upper bounds of the observation space.

        Returns:
            The modified state vector.
        """
        # Only handle 2D boundaries for now (x, y)
        if len(state) < 2:
            return state

        x, y = state[0], state[1]
        has_vel_x = len(state) >= 3
        has_vel_y = len(state) >= 4
        elasticity = self.config.elasticity
        margin = self.config.boundary_margin
        corner_thresh = self.config.corner_threshold

        # Helper to detect corner collisions
        def in_corner(x, y):
            corners = [
                (low_bound[0], low_bound[1]),
                (low_bound[0], high_bound[1]),
                (high_bound[0], low_bound[1]),
                (high_bound[0], high_bound[1])
            ]
            for cx, cy in corners:
                if abs(x - cx) < corner_thresh and abs(y - cy) < corner_thresh:
                    return True
            return False

        # Ground (lower y)
        ground = low_bound[1] + margin
        if y < ground:
            state[1] = ground
            if has_vel_y:
                if in_corner(state[0], state[1]):
                    # Diagonal bounce
                    if has_vel_x:
                        state[2] = -state[2] * elasticity
                    state[3] = -state[3] * elasticity
                else:
                    state[3] = -state[3] * elasticity
                    if has_vel_x:
                        state[2] *= (1 - (1 - elasticity))  # horizontal friction on impact

        # Ceiling (upper y)
        ceiling = high_bound[1] - margin
        if y > ceiling:
            state[1] = ceiling
            if has_vel_y:
                if in_corner(state[0], state[1]):
                    if has_vel_x:
                        state[2] = -state[2] * elasticity
                    state[3] = -state[3] * elasticity
                else:
                    state[3] = -state[3] * elasticity
                    if has_vel_x:
                        state[2] *= (1 - (1 - elasticity))

        # Left wall (lower x)
        left = low_bound[0] + margin
        if x < left:
            state[0] = left
            if has_vel_x:
                if in_corner(state[0], state[1]):
                    if has_vel_y:
                        state[3] = -state[3] * elasticity
                    state[2] = -state[2] * elasticity
                else:
                    state[2] = -state[2] * elasticity
                    if has_vel_y:
                        state[3] *= (1 - (1 - elasticity))

        # Right wall (upper x)
        right = high_bound[0] - margin
        if x > right:
            state[0] = right
            if has_vel_x:
                if in_corner(state[0], state[1]):
                    if has_vel_y:
                        state[3] = -state[3] * elasticity
                    state[2] = -state[2] * elasticity
                else:
                    state[2] = -state[2] * elasticity
                    if has_vel_y:
                        state[3] *= (1 - (1 - elasticity))

        # Angular constraints (if present)
        if len(state) >= 5:
            state[4] = state[4] % (2 * np.pi)  # Normalize angle
            if len(state) >= 6:
                max_ang_vel = 5.0
                if abs(state[5]) > max_ang_vel:
                    state[5] = np.sign(state[5]) * max_ang_vel

        return state

    def apply_all(self, state: np.ndarray, dt: float, low_bound: np.ndarray, high_bound: np.ndarray,
                  mass: Optional[float] = None) -> np.ndarray:
        """
        Apply both environmental effects and boundary constraints in the correct order.

        Args:
            state: State vector (modified in-place).
            dt: Time step.
            low_bound: Lower bounds.
            high_bound: Upper bounds.
            mass: Mass (optional, defaults to config.default_mass).

        Returns:
            Modified state.
        """
        self.apply_environmental_effects(state, dt, mass)
        self.enforce_boundary_constraints(state, low_bound, high_bound)
        return state


# ========== Legacy API (for backward compatibility with SLAIEnv) ==========

_engine = None  # Singleton engine instance

def _get_engine() -> PhysicsEngine:
    """Get or create a singleton PhysicsEngine instance."""
    global _engine
    if _engine is None:
        _engine = PhysicsEngine()
    return _engine


def apply_constants(env_instance: Any) -> Dict[str, float]:
    """
    Register physical constants in the environment instance.
    This function is kept for backward compatibility.

    Args:
        env_instance: An SLAIEnv instance (will receive a .constants attribute).

    Returns:
        The constants dictionary.
    """
    engine = _get_engine()
    env_instance.constants = engine.constants
    return env_instance.constants


def apply_environmental_effects(env_instance: Any, state_array: np.ndarray) -> np.ndarray:
    """
    Apply environmental effects using the environment's parameters.

    Args:
        env_instance: SLAIEnv instance (must have dt, gravity, friction_coeff, etc.)
        state_array: State vector (modified in-place).

    Returns:
        Modified state.
    """
    engine = _get_engine()
    # Update engine config from environment if attributes exist
    if hasattr(env_instance, 'dt'):
        engine.config.dt = env_instance.dt
    if hasattr(env_instance, 'gravity'):
        engine.config.gravity = env_instance.gravity
    if hasattr(env_instance, 'friction_coeff'):
        engine.config.friction_coeff = env_instance.friction_coeff
    if hasattr(env_instance, 'wind_strength'):
        engine.config.wind_strength = env_instance.wind_strength
    if hasattr(env_instance, 'wind_direction'):
        engine.config.wind_direction = env_instance.wind_direction
    if hasattr(env_instance, 'drag_coeff'):
        engine.config.drag_coeff = env_instance.drag_coeff
    if hasattr(env_instance, 'terminal_velocity'):
        engine.config.terminal_velocity = env_instance.terminal_velocity
    if hasattr(env_instance, 'rotational_friction'):
        engine.config.rotational_friction = env_instance.rotational_friction
    mass = getattr(env_instance, 'mass', engine.config.default_mass)

    engine.apply_environmental_effects(state_array, engine.config.dt, mass)
    return state_array


def enforce_physics_constraints(env_instance: Any, state_array: np.ndarray) -> np.ndarray:
    """
    Enforce boundary constraints using the environment's observation space.

    Args:
        env_instance: SLAIEnv instance (must have observation_space and elasticity).
        state_array: State vector (modified in-place).

    Returns:
        Modified state.
    """
    engine = _get_engine()
    if hasattr(env_instance, 'elasticity'):
        engine.config.elasticity = env_instance.elasticity
    if hasattr(env_instance, 'observation_space'):
        low = env_instance.observation_space.low
        high = env_instance.observation_space.high
    else:
        # Fallback to reasonable defaults if no observation space
        low = np.array([-10.0, -10.0])
        high = np.array([10.0, 10.0])

    engine.enforce_boundary_constraints(state_array, low, high)
    return state_array


if __name__ == "__main__":
    print("\n=== Running Physics Constraints ===\n")
    printer.status("TEST", "Starting Physics Constraints tests", "info")

    engine = PhysicsEngine()

    # Starting at origin with a diagonal velocity
    state = np.array([0.0, 0.0, 5.0, 10.0], dtype=float)
    dt = 0.02
    low_bound = np.array([-10.0, -10.0])
    high_bound = np.array([10.0, 10.0])
    
    print(f"Initial state: {state}")

    # Apply physics for a few steps
    for step in range(10):
        state = engine.apply_all(state, dt, low_bound, high_bound)
        print(f"Step {step+1}: x={state[0]:6.2f}, y={state[1]:6.2f}, vx={state[2]:6.2f}, vy={state[3]:6.2f}")
    
    printer.pretty("SNAPSHOT", "Physics test completed", "success")
