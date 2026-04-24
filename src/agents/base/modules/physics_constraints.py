"""
Physics constraints and environmental effects for the SLAI Environment.

This module provides a production-ready physics engine used to simulate
core environmental and boundary interactions for SLAI-style state vectors.
It models gravity, friction, drag, wind, collisions, optional
relativistic/electromagnetic effects, and simplified tunneling behavior.
The state vector is assumed to have components at specific indices:

- state[0]: x-position
- state[1]: y-position
- state[2]: x-velocity (if state_dim >= 3)
- state[3]: y-velocity (if state_dim >= 4)
- state[4]: angle (radians) (if state_dim >= 5)
- state[5]: angular velocity (rad/s) (if state_dim >= 6)
- state[6]: moment of inertia (if state_dim >= 7)
- state[7]: electric charge (if state_dim >= 8)

All physics functions are designed to work with the SLAIEnv class, but can be
used independently given a compatible state vector and configuration. The
implementation stays intentionally generic so higher-level environment modules
can reuse one validated engine instead of scattering physical rules across the
codebase.
"""

from __future__ import annotations

import math
import numpy as np

from dataclasses import dataclass, field
from collections import deque
from collections.abc import Mapping
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.base_errors import *
from ..utils.base_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Physics Constraints")
printer = PrettyPrinter()


@dataclass
class PhysicsConfig:
    """Configuration parameters for the physics engine."""

    gravity: float = 9.80665
    friction_coeff: float = 0.02
    rotational_friction: float = 0.01

    wind_strength: float = 0.0
    wind_direction: float = 0.0
    wind_turbulence_ratio: float = 0.1

    drag_coeff: float = 0.01
    terminal_velocity: float = 50.0
    min_speed_for_drag: float = 0.01

    elasticity: float = 0.8
    tangential_damping: float = 0.2
    boundary_margin: float = 0.01
    corner_threshold: float = 0.05
    max_angular_velocity: float = 5.0

    default_mass: float = 1.0
    default_charge: float = 0.0

    enable_tunneling: bool = False
    tunneling_probability: float = 0.05
    barrier_positions: Tuple[float, ...] = (-8.0, 8.0)
    barrier_width: float = 0.1

    enable_relativistic: bool = True
    relativistic_threshold: float = 0.1
    relativistic_safety_factor: float = 0.999999

    enable_electromagnetic: bool = False
    electric_field: Tuple[float, float] = (0.0, 10.0)
    magnetic_field: float = 0.5

    dt: float = 0.02
    enable_history: bool = True
    history_limit: int = 200
    random_seed: Optional[int] = None


@dataclass(frozen=True)
class PhysicsStepSummary:
    """Structured audit record for one physics application step."""

    timestamp: str
    dt: float
    collisions: int
    relativistic_clamp_applied: bool
    tunneling_events: int
    electromagnetic_applied: bool
    speed_before: float
    speed_after: float
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "dt": self.dt,
            "collisions": self.collisions,
            "relativistic_clamp_applied": self.relativistic_clamp_applied,
            "tunneling_events": self.tunneling_events,
            "electromagnetic_applied": self.electromagnetic_applied,
            "speed_before": self.speed_before,
            "speed_after": self.speed_after,
            "notes": to_json_safe(self.notes),
        }


class PhysicsEngine:
    """
    Centralized physics engine for the SLAI Environment.

    The engine is config-driven, validates runtime inputs, records bounded step
    history, and preserves the legacy API functions expected by existing env
    wrappers.
    """

    CONSTANTS: Dict[str, float] = {
        "G": 6.67430e-11,
        "g": 9.80665,
        "lambda": 1.1056e-52,
        "epsilon0": 8.8541878128e-12,
        "mu0": 1.25663706212e-6,
        "ke": 8.9875517923e9,
        "c": 299792458.0,
        "e": 1.602176634e-19,
        "alpha": 7.2973525693e-3,
        "phi0": 2.067833848e-15,
        "kB": 1.380649e-23,
        "R": 8.31446261815324,
        "NA": 6.02214076e23,
        "F": 96485.3321233100184,
        "sigma": 5.670374419e-8,
        "kW": 2.897771955e-3,
        "h": 6.62607015e-34,
        "hbar": 1.054571817e-34,
        "me": 9.1093837015e-31,
        "mp": 1.67262192369e-27,
        "mn": 1.67492749804e-27,
        "u": 1.66053906660e-27,
        "Hz": 1.0,
        "day": 86400.0,
        "year": 31557600.0,
        "T0": 273.15,
        "P0": 101325.0,
        "Z0": 376.730313668,
        "lP": 1.616255e-35,
        "tP": 5.391247e-44,
        "mP": 2.176434e-8,
        "TP": 1.416784e32,
    }

    def __init__(self, config: Optional[Union[PhysicsConfig, Mapping[str, Any]]] = None):
        self.global_config = load_global_config()
        self.physics_config = get_config_section("physics_constraints") or {}

        if config is None:
            self.config = self._build_config_from_mapping(self.physics_config)
        elif isinstance(config, PhysicsConfig):
            self.config = config
        elif isinstance(config, Mapping):
            merged = deep_merge_dicts(self.physics_config, dict(config))
            self.config = self._build_config_from_mapping(merged)
        else:
            raise BaseValidationError(
                "config must be None, a mapping, or PhysicsConfig.",
                self.physics_config,
                component="PhysicsEngine",
                operation="__init__",
                context={"received_type": type(config).__name__},
            )

        self.constants = dict(self.CONSTANTS)
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.config.history_limit)
        self._rng = np.random.default_rng(self.config.random_seed)
        self._stats: Dict[str, int] = {
            "environment_steps": 0,
            "boundary_steps": 0,
            "collisions": 0,
            "tunneling_events": 0,
            "relativistic_clamps": 0,
            "electromagnetic_applications": 0,
        }

        self._validate_config()
        logger.info("Physics Constraints successfully initialized")

    # ------------------------------------------------------------------
    # Configuration and validation
    # ------------------------------------------------------------------
    def _build_config_from_mapping(self, mapping: Mapping[str, Any]) -> PhysicsConfig:
        ensure_mapping(
            mapping,
            "physics_constraints",
            config=self.physics_config,
            error_cls=BaseConfigurationError,
        )
        return PhysicsConfig(
            gravity=coerce_float(mapping.get("gravity", 9.80665), 9.80665),
            friction_coeff=coerce_float(mapping.get("friction_coeff", 0.02), 0.02, minimum=0.0),
            rotational_friction=coerce_float(mapping.get("rotational_friction", 0.01), 0.01, minimum=0.0),
            wind_strength=coerce_float(mapping.get("wind_strength", 0.0), 0.0, minimum=0.0),
            wind_direction=coerce_float(mapping.get("wind_direction", 0.0), 0.0),
            wind_turbulence_ratio=coerce_float(mapping.get("wind_turbulence_ratio", 0.1), 0.1, minimum=0.0),
            drag_coeff=coerce_float(mapping.get("drag_coeff", 0.01), 0.01, minimum=0.0),
            terminal_velocity=coerce_float(mapping.get("terminal_velocity", 50.0), 50.0, minimum=0.0),
            min_speed_for_drag=coerce_float(mapping.get("min_speed_for_drag", 0.01), 0.01, minimum=0.0),
            elasticity=coerce_float(mapping.get("elasticity", 0.8), 0.8, minimum=0.0, maximum=1.0),
            tangential_damping=coerce_float(mapping.get("tangential_damping", 0.2), 0.2, minimum=0.0, maximum=1.0),
            boundary_margin=coerce_float(mapping.get("boundary_margin", 0.01), 0.01, minimum=0.0),
            corner_threshold=coerce_float(mapping.get("corner_threshold", 0.05), 0.05, minimum=0.0),
            max_angular_velocity=coerce_float(mapping.get("max_angular_velocity", 5.0), 5.0, minimum=0.0),
            default_mass=coerce_float(mapping.get("default_mass", 1.0), 1.0, minimum=1.0e-12),
            default_charge=coerce_float(mapping.get("default_charge", 0.0), 0.0),
            enable_tunneling=coerce_bool(mapping.get("enable_tunneling", False), False),
            tunneling_probability=coerce_float(mapping.get("tunneling_probability", 0.05), 0.05, minimum=0.0, maximum=1.0),
            barrier_positions=tuple(coerce_float(v, 0.0) for v in ensure_list(mapping.get("barrier_positions", (-8.0, 8.0)))),
            barrier_width=coerce_float(mapping.get("barrier_width", 0.1), 0.1, minimum=0.0),
            enable_relativistic=coerce_bool(mapping.get("enable_relativistic", True), True),
            relativistic_threshold=coerce_float(mapping.get("relativistic_threshold", 0.1), 0.1, minimum=0.0, maximum=1.0),
            relativistic_safety_factor=coerce_float(mapping.get("relativistic_safety_factor", 0.999999), 0.999999, minimum=0.0, maximum=1.0),
            enable_electromagnetic=coerce_bool(mapping.get("enable_electromagnetic", False), False),
            electric_field=self._normalize_vector(mapping.get("electric_field", (0.0, 10.0)), "electric_field"),
            magnetic_field=coerce_float(mapping.get("magnetic_field", 0.5), 0.5),
            dt=coerce_float(mapping.get("dt", 0.02), 0.02, minimum=1.0e-12),
            enable_history=coerce_bool(mapping.get("enable_history", True), True),
            history_limit=coerce_int(mapping.get("history_limit", 200), 200, minimum=1),
            random_seed=(None if mapping.get("random_seed", None) in (None, "", "none", "None") else coerce_int(mapping.get("random_seed"), 0)),
        )

    def _normalize_vector(self, value: Any, name: str) -> Tuple[float, float]:
        if isinstance(value, str):
            parts = parse_delimited_text(value)
        else:
            parts = ensure_list(value)
        ensure_condition(
            len(parts) == 2,
            f"'{name}' must contain exactly two numeric values.",
            config=self.physics_config,
            error_cls=BaseConfigurationError,
            component="PhysicsEngine",
            operation="configuration",
            context={"field": name, "received": to_json_safe(parts)},
        )
        return (coerce_float(parts[0], 0.0), coerce_float(parts[1], 0.0))

    def _validate_config(self) -> None:
        ensure_numeric_range(
            self.config.gravity,
            "gravity",
            minimum=0.0,
            config=self.physics_config,
            error_cls=BaseConfigurationError,
        )
        ensure_numeric_range(
            self.config.elasticity,
            "elasticity",
            minimum=0.0,
            maximum=1.0,
            config=self.physics_config,
            error_cls=BaseConfigurationError,
        )
        ensure_numeric_range(
            self.config.relativistic_threshold,
            "relativistic_threshold",
            minimum=0.0,
            maximum=1.0,
            config=self.physics_config,
            error_cls=BaseConfigurationError,
        )

    def _validate_state_array(self, state: np.ndarray, *, name: str = "state") -> np.ndarray:
        ensure_condition(
            isinstance(state, np.ndarray),
            f"'{name}' must be a numpy.ndarray.",
            config=self.physics_config,
            error_cls=BaseValidationError,
            component="PhysicsEngine",
            operation="validation",
            context={"field": name, "received_type": type(state).__name__},
        )
        ensure_condition(
            state.ndim == 1,
            f"'{name}' must be a 1D state vector.",
            config=self.physics_config,
            error_cls=BaseValidationError,
            component="PhysicsEngine",
            operation="validation",
            context={"field": name, "ndim": int(state.ndim)},
        )
        return state

    def _validate_bounds(self, low_bound: np.ndarray, high_bound: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        low = self._validate_state_array(np.asarray(low_bound, dtype=float), name="low_bound")
        high = self._validate_state_array(np.asarray(high_bound, dtype=float), name="high_bound")
        ensure_condition(
            len(low) >= 2 and len(high) >= 2,
            "Boundary vectors must have at least two dimensions.",
            config=self.physics_config,
            error_cls=BaseValidationError,
            component="PhysicsEngine",
            operation="validation",
        )
        ensure_condition(
            bool(np.all(high[:2] > low[:2])),
            "Each high boundary value must be greater than the corresponding low boundary value.",
            config=self.physics_config,
            error_cls=BaseValidationError,
            component="PhysicsEngine",
            operation="validation",
            context={"low_bound": low[:2].tolist(), "high_bound": high[:2].tolist()},
        )
        return low, high

    def _speed(self, state: np.ndarray) -> float:
        if len(state) < 4:
            return abs(float(state[2])) if len(state) >= 3 else 0.0
        return float(np.hypot(state[2], state[3]))

    # ------------------------------------------------------------------
    # Physics core
    # ------------------------------------------------------------------
    def apply_environmental_effects(
        self,
        state: np.ndarray,
        dt: Optional[float] = None,
        mass: Optional[float] = None,
    ) -> np.ndarray:
        """Apply gravity, damping, drag, wind, and advanced effects in-place."""
        state = self._validate_state_array(np.asarray(state, dtype=float))
        step_dt = self.config.dt if dt is None else coerce_float(dt, self.config.dt, minimum=1.0e-12)
        particle_mass = coerce_float(mass if mass is not None else self.config.default_mass, self.config.default_mass, minimum=1.0e-12)

        has_vel_x = len(state) >= 3
        has_vel_y = len(state) >= 4
        has_ang_vel = len(state) >= 6
        has_charge = len(state) >= 8
        speed_before = self._speed(state)
        relativistic_clamp_applied = False
        tunneling_events = 0
        electromagnetic_applied = False

        if has_vel_y:
            state[3] -= self.config.gravity * step_dt

        damping = max(0.0, 1.0 - self.config.friction_coeff * step_dt)
        if has_vel_x:
            state[2] *= damping
        if has_vel_y:
            state[3] *= damping

        if has_vel_x and has_vel_y and self.config.wind_strength > 0.0:
            base_wind_x = self.config.wind_strength * math.cos(self.config.wind_direction)
            base_wind_y = self.config.wind_strength * math.sin(self.config.wind_direction)
            turbulence_scale = self.config.wind_strength * self.config.wind_turbulence_ratio
            turbulence_x, turbulence_y = self._rng.normal(0.0, turbulence_scale, 2)
            state[2] += (base_wind_x + turbulence_x) * step_dt
            state[3] += (base_wind_y + turbulence_y) * step_dt

        if has_vel_x and has_vel_y and self.config.drag_coeff > 0.0:
            vx = float(state[2])
            vy = float(state[3])
            speed = float(np.hypot(vx, vy))
            if speed > self.config.min_speed_for_drag:
                drag_force = self.config.drag_coeff * (speed ** 2)
                drag_accel = drag_force / particle_mass
                state[2] += (-drag_accel * vx / speed) * step_dt
                state[3] += (-drag_accel * vy / speed) * step_dt

        if has_vel_x and has_vel_y and self.config.terminal_velocity > 0.0:
            speed = self._speed(state)
            if speed > self.config.terminal_velocity:
                scale = self.config.terminal_velocity / speed
                state[2] *= scale
                state[3] *= scale

        if has_ang_vel:
            rotational_damping = max(0.0, 1.0 - self.config.rotational_friction * step_dt)
            state[5] *= rotational_damping

        if self.config.enable_relativistic and has_vel_x and has_vel_y:
            speed = self._speed(state)
            if speed > self.config.relativistic_threshold * self.constants["c"]:
                max_allowed = self.constants["c"] * self.config.relativistic_safety_factor
                if speed > max_allowed and speed > 0:
                    scale = max_allowed / speed
                    state[2] *= scale
                    state[3] *= scale
                    relativistic_clamp_applied = True
                    self._stats["relativistic_clamps"] += 1

        if self.config.enable_tunneling and has_vel_x and self.config.barrier_width > 0.0:
            for barrier in self.config.barrier_positions:
                if abs(float(state[0]) - barrier) <= self.config.barrier_width:
                    if self._rng.random() < self.config.tunneling_probability:
                        direction = np.sign(state[2]) if abs(float(state[2])) > 0 else 1.0
                        state[0] = barrier + self.config.barrier_width * direction
                        tunneling_events += 1
                        self._stats["tunneling_events"] += 1

        if self.config.enable_electromagnetic and has_charge and has_vel_x and has_vel_y:
            charge = float(state[7]) if abs(float(state[7])) > 1.0e-12 else self.config.default_charge
            if abs(charge) > 1.0e-12:
                ex, ey = self.config.electric_field
                bz = self.config.magnetic_field
                vx = float(state[2])
                vy = float(state[3])
                ax = (charge / particle_mass) * (ex + vy * bz)
                ay = (charge / particle_mass) * (ey - vx * bz)
                state[2] += ax * step_dt
                state[3] += ay * step_dt
                electromagnetic_applied = True
                self._stats["electromagnetic_applications"] += 1

        self._stats["environment_steps"] += 1
        self._record_history(
            PhysicsStepSummary(
                timestamp=utc_now_iso(),
                dt=step_dt,
                collisions=0,
                relativistic_clamp_applied=relativistic_clamp_applied,
                tunneling_events=tunneling_events,
                electromagnetic_applied=electromagnetic_applied,
                speed_before=speed_before,
                speed_after=self._speed(state),
                notes={"phase": "environmental_effects"},
            )
        )
        return state

    def enforce_boundary_constraints(
        self,
        state: np.ndarray,
        low_bound: np.ndarray,
        high_bound: np.ndarray,
    ) -> np.ndarray:
        """Enforce wall/ground/ceiling collisions and angular normalization in-place."""
        state = self._validate_state_array(np.asarray(state, dtype=float))
        low, high = self._validate_bounds(low_bound, high_bound)

        has_vel_x = len(state) >= 3
        has_vel_y = len(state) >= 4
        has_angle = len(state) >= 5
        has_ang_vel = len(state) >= 6
        collisions = 0
        speed_before = self._speed(state)

        left = float(low[0] + self.config.boundary_margin)
        right = float(high[0] - self.config.boundary_margin)
        bottom = float(low[1] + self.config.boundary_margin)
        top = float(high[1] - self.config.boundary_margin)

        def damp_tangent(value: float) -> float:
            return value * max(0.0, 1.0 - self.config.tangential_damping)

        def in_corner(x: float, y: float) -> bool:
            corners = ((left, bottom), (left, top), (right, bottom), (right, top))
            return any(abs(x - cx) <= self.config.corner_threshold and abs(y - cy) <= self.config.corner_threshold for cx, cy in corners)

        if float(state[1]) < bottom:
            state[1] = bottom
            collisions += 1
            if has_vel_y:
                state[3] = abs(float(state[3])) * self.config.elasticity
            if has_vel_x:
                state[2] = damp_tangent(float(state[2]))

        if float(state[1]) > top:
            state[1] = top
            collisions += 1
            if has_vel_y:
                state[3] = -abs(float(state[3])) * self.config.elasticity
            if has_vel_x:
                state[2] = damp_tangent(float(state[2]))

        if float(state[0]) < left:
            state[0] = left
            collisions += 1
            if has_vel_x:
                state[2] = abs(float(state[2])) * self.config.elasticity
            if has_vel_y:
                state[3] = damp_tangent(float(state[3]))

        if float(state[0]) > right:
            state[0] = right
            collisions += 1
            if has_vel_x:
                state[2] = -abs(float(state[2])) * self.config.elasticity
            if has_vel_y:
                state[3] = damp_tangent(float(state[3]))

        if collisions > 1 and has_vel_x and has_vel_y and in_corner(float(state[0]), float(state[1])):
            state[2] *= self.config.elasticity
            state[3] *= self.config.elasticity

        if has_angle:
            state[4] = float(state[4]) % (2.0 * math.pi)
        if has_ang_vel and abs(float(state[5])) > self.config.max_angular_velocity:
            state[5] = math.copysign(self.config.max_angular_velocity, float(state[5]))

        self._stats["boundary_steps"] += 1
        self._stats["collisions"] += collisions
        self._record_history(
            PhysicsStepSummary(
                timestamp=utc_now_iso(),
                dt=self.config.dt,
                collisions=collisions,
                relativistic_clamp_applied=False,
                tunneling_events=0,
                electromagnetic_applied=False,
                speed_before=speed_before,
                speed_after=self._speed(state),
                notes={"phase": "boundary_constraints"},
            )
        )
        return state

    def apply_all(
        self,
        state: np.ndarray,
        dt: float,
        low_bound: np.ndarray,
        high_bound: np.ndarray,
        mass: Optional[float] = None,
    ) -> np.ndarray:
        """Apply environmental effects then boundary constraints."""
        state = self.apply_environmental_effects(state, dt=dt, mass=mass)
        state = self.enforce_boundary_constraints(state, low_bound, high_bound)
        return state

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    def _record_history(self, summary: PhysicsStepSummary) -> None:
        if not self.config.enable_history:
            return
        self._history.append(summary.to_dict())

    def recent_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        limit = coerce_int(limit, 20, minimum=1)
        return list(self._history)[-limit:]

    def stats(self) -> Dict[str, Any]:
        return {
            "config": to_json_safe(self.config.__dict__),
            "stats": dict(self._stats),
            "history_length": len(self._history),
            "constants_count": len(self.constants),
        }


# ========== Legacy API (for backward compatibility with SLAIEnv) ==========
_engine: Optional[PhysicsEngine] = None


def _get_engine() -> PhysicsEngine:
    """Get or create a singleton PhysicsEngine instance."""
    global _engine
    if _engine is None:
        _engine = PhysicsEngine()
    return _engine


def apply_constants(env_instance: Any) -> Dict[str, float]:
    """Register physical constants on an environment instance."""
    engine = _get_engine()
    env_instance.constants = dict(engine.constants)
    return env_instance.constants


def apply_environmental_effects(env_instance: Any, state_array: np.ndarray) -> np.ndarray:
    """Apply environmental effects using environment-provided overrides when present."""
    engine = _get_engine()
    override: Dict[str, Any] = {}
    for key in (
        "dt",
        "gravity",
        "friction_coeff",
        "rotational_friction",
        "wind_strength",
        "wind_direction",
        "drag_coeff",
        "terminal_velocity",
        "elasticity",
        "default_mass",
        "default_charge",
        "enable_tunneling",
        "tunneling_probability",
        "barrier_positions",
        "barrier_width",
        "enable_relativistic",
        "relativistic_threshold",
        "enable_electromagnetic",
        "electric_field",
        "magnetic_field",
    ):
        if hasattr(env_instance, key):
            override[key] = getattr(env_instance, key)

    if override:
        engine = PhysicsEngine(config=override)
    mass = getattr(env_instance, "mass", engine.config.default_mass)
    return engine.apply_environmental_effects(np.asarray(state_array, dtype=float), dt=engine.config.dt, mass=mass)


def enforce_physics_constraints(env_instance: Any, state_array: np.ndarray) -> np.ndarray:
    """Enforce boundary constraints using the environment observation space when present."""
    engine = _get_engine()
    if hasattr(env_instance, "elasticity"):
        engine = PhysicsEngine(config={"elasticity": getattr(env_instance, "elasticity")})

    if hasattr(env_instance, "observation_space"):
        low = np.asarray(env_instance.observation_space.low, dtype=float)
        high = np.asarray(env_instance.observation_space.high, dtype=float)
    else:
        low = np.array([-10.0, -10.0], dtype=float)
        high = np.array([10.0, 10.0], dtype=float)

    return engine.enforce_boundary_constraints(np.asarray(state_array, dtype=float), low, high)


def apply_all_physics_constraints(
    env_instance: Any,
    state_array: np.ndarray,
    low_bound: Optional[np.ndarray] = None,
    high_bound: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compatibility wrapper that applies both environment and boundary effects."""
    engine = _get_engine()
    dt = getattr(env_instance, "dt", engine.config.dt)
    mass = getattr(env_instance, "mass", engine.config.default_mass)

    if low_bound is None or high_bound is None:
        if hasattr(env_instance, "observation_space"):
            low_bound = np.asarray(env_instance.observation_space.low, dtype=float)
            high_bound = np.asarray(env_instance.observation_space.high, dtype=float)
        else:
            low_bound = np.array([-10.0, -10.0], dtype=float)
            high_bound = np.array([10.0, 10.0], dtype=float)

    return engine.apply_all(np.asarray(state_array, dtype=float), float(dt), low_bound, high_bound, mass=mass)


if __name__ == "__main__":
    print("\n=== Running Physics Constraints ===\n")
    printer.status("TEST", "Physics Constraints initialized", "info")

    engine = PhysicsEngine()
    state = np.array([0.0, 0.0, 5.0, 10.0, 0.0, 0.2, 1.0, 0.1], dtype=float)
    dt = 0.02
    low_bound = np.array([-10.0, -10.0], dtype=float)
    high_bound = np.array([10.0, 10.0], dtype=float)

    printer.pretty("INITIAL_STATE", state.tolist(), "info")

    for step in range(10):
        state[0] += state[2] * dt
        state[1] += state[3] * dt
        state = engine.apply_all(state, dt, low_bound, high_bound, mass=1.5)
        printer.pretty(
            f"STEP_{step + 1}",
            {
                "x": round(float(state[0]), 5),
                "y": round(float(state[1]), 5),
                "vx": round(float(state[2]), 5),
                "vy": round(float(state[3]), 5),
            },
            "success",
        )

    printer.pretty("RECENT_HISTORY", engine.recent_history(), "success")
    printer.pretty("PHYSICS_STATS", engine.stats(), "success")

    print("\n=== Test ran successfully ===\n")
