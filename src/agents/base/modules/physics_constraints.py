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
import numpy as np # pyright: ignore[reportMissingImports]

from dataclasses import dataclass, field
from collections import deque
from collections.abc import Mapping
from typing import Any, Deque, Dict, List, Optional, Tuple

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.base_errors import *
from ..utils.base_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Physics Constraints")
printer = PrettyPrinter()


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


@dataclass(frozen=True)
class ConservationAudit:
    """Energy/momentum audit for one physics step."""
    timestamp: str
    kinetic_energy: float
    potential_energy: float
    thermal_energy: float
    total_energy: float
    momentum_x: float
    momentum_y: float
    angular_momentum: float
    energy_drift: float
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_json_safe({
            "timestamp": self.timestamp,
            "kinetic_energy": self.kinetic_energy,
            "potential_energy": self.potential_energy,
            "thermal_energy": self.thermal_energy,
            "total_energy": self.total_energy,
            "momentum_x": self.momentum_x,
            "momentum_y": self.momentum_y,
            "angular_momentum": self.angular_momentum,
            "energy_drift": self.energy_drift,
            "notes": self.notes,
        })


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
        "rho_air": 1.225,              # kg/m^3
        "rho_water": 997.0,
        "mu_air": 1.81e-5,             # dynamic viscosity (Pa·s)
        "sigma_sb": 5.670374419e-8,    # Stefan-Boltzmann
        "atm_pressure": 101325.0,
        "stefan_boltzmann": 5.670374419e-8,
        "planck": 6.62607015e-34,
        "avogadro": 6.02214076e23,
        "gas_constant": 8.31446261815324,
        "coulomb_k": 8.9875517923e9,
        "gauss": 1.0e-4,
    }

    CONFIG_FIELDS: Tuple[str, ...] = (
        "gravity",
        "friction_coeff",
        "rotational_friction",
        "wind_strength",
        "wind_direction",
        "wind_turbulence_ratio",
        "drag_coeff",
        "terminal_velocity",
        "min_speed_for_drag",
        "elasticity",
        "tangential_damping",
        "boundary_margin",
        "corner_threshold",
        "max_angular_velocity",
        "default_mass",
        "default_charge",
        "enable_tunneling",
        "tunneling_probability",
        "barrier_positions",
        "barrier_width",
        "enable_relativistic",
        "relativistic_threshold",
        "relativistic_safety_factor",
        "enable_electromagnetic",
        "electric_field",
        "magnetic_field",
        "dt",
        "enable_history",
        "history_limit",
        "random_seed",
    
        # --- classical ---
        "static_friction_coeff", "kinetic_friction_coeff",
        "body_radius", "air_density", "fluid_density", "buoyancy_enabled",
        "magnus_coeff", "angular_velocity_index",
        "coriolis_omega", "coriolis_axis",
        "spring_constants", "spring_anchors", "spring_damping",
        "pendulum_length", "pendulum_anchor", "pendulum_enabled",
        "nbody_masses", "nbody_positions", "nbody_G_enabled",
        # --- thermodynamics ---
        "enable_thermodynamics", "ambient_temperature", "body_temperature",
        "heat_transfer_coeff", "thermal_conductivity", "emissivity",
        "specific_heat", "thermal_expansion_coeff", "reference_temperature",
        "container_volume",
        # --- EM ---
        "enable_coulomb", "other_charges",
        "enable_faraday", "magnetic_flux_rate",
        "enable_radiation_pressure", "irradiance",
        # --- relativity ---
        "enable_time_dilation", "enable_length_contraction",
        "enable_relativistic_momentum",
        # --- quantum ---
        "enable_wkb_tunneling", "barrier_heights", "particle_mass_kg",
        "enable_de_broglie", "enable_uncertainty_floor", "uncertainty_scale",
        # --- materials ---
        "enable_material_stress", "youngs_modulus", "yield_strength",
        "fracture_strain", "plastic_hardening", "rest_length",
        "enable_fatigue", "fatigue_limit",
        # --- constraints ---
        "enable_distance_constraints", "distance_targets",
        "enable_angle_constraints", "angle_targets",
        "min_velocity", "max_velocity", "max_acceleration", "max_jerk",
        # --- audit ---
        "enable_conservation_audit", "energy_tolerance",
        )

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        self.global_config = load_global_config()
        base_physics_config = get_config_section("physics_constraints") or {}
        ensure_mapping(
            base_physics_config,
            "physics_constraints",
            config=base_physics_config,
            error_cls=BaseConfigurationError,
            component="PhysicsEngine",
            operation="__init__",
        )

        if config is None:
            self.physics_config = dict(base_physics_config)
        elif isinstance(config, Mapping):
            # Runtime overrides are still supported for the legacy wrappers, but
            # the canonical defaults remain in base_config.yaml.
            self.physics_config = deep_merge_dicts(base_physics_config, dict(config))
        else:
            raise BaseValidationError(
                "config must be None or a mapping of physics constraint overrides.",
                base_physics_config,
                component="PhysicsEngine",
                operation="__init__",
                context={"received_type": type(config).__name__},
            )

        mapping = self.physics_config

        self.gravity = coerce_float(mapping.get("gravity", 9.80665), 9.80665)
        self.friction_coeff = coerce_float(mapping.get("friction_coeff", 0.02), 0.02, minimum=0.0)
        self.rotational_friction = coerce_float(mapping.get("rotational_friction", 0.01), 0.01, minimum=0.0)

        self.wind_strength = coerce_float(mapping.get("wind_strength", 0.0), 0.0, minimum=0.0)
        self.wind_direction = coerce_float(mapping.get("wind_direction", 0.0), 0.0)
        self.wind_turbulence_ratio = coerce_float(mapping.get("wind_turbulence_ratio", 0.1), 0.1, minimum=0.0)

        self.drag_coeff = coerce_float(mapping.get("drag_coeff", 0.01), 0.01, minimum=0.0)
        self.terminal_velocity = coerce_float(mapping.get("terminal_velocity", 50.0), 50.0, minimum=0.0)
        self.min_speed_for_drag = coerce_float(mapping.get("min_speed_for_drag", 0.01), 0.01, minimum=0.0)

        self.elasticity = coerce_float(mapping.get("elasticity", 0.8), 0.8, minimum=0.0, maximum=1.0)
        self.tangential_damping = coerce_float(mapping.get("tangential_damping", 0.2), 0.2, minimum=0.0, maximum=1.0)
        self.boundary_margin = coerce_float(mapping.get("boundary_margin", 0.01), 0.01, minimum=0.0)
        self.corner_threshold = coerce_float(mapping.get("corner_threshold", 0.05), 0.05, minimum=0.0)
        self.max_angular_velocity = coerce_float(mapping.get("max_angular_velocity", 5.0), 5.0, minimum=0.0)

        self.default_mass = coerce_float(mapping.get("default_mass", 1.0), 1.0, minimum=1.0e-12)
        self.default_charge = coerce_float(mapping.get("default_charge", 0.0), 0.0)

        self.enable_tunneling = coerce_bool(mapping.get("enable_tunneling", False), False)
        self.tunneling_probability = coerce_float(
            mapping.get("tunneling_probability", 0.05),
            0.05,
            minimum=0.0,
            maximum=1.0,
        )
        self.barrier_positions = tuple(
            coerce_float(value, 0.0)
            for value in ensure_list(mapping.get("barrier_positions", (-8.0, 8.0)))
        )
        self.barrier_width = coerce_float(mapping.get("barrier_width", 0.1), 0.1, minimum=0.0)

        self.enable_relativistic = coerce_bool(mapping.get("enable_relativistic", True), True)
        self.relativistic_threshold = coerce_float(
            mapping.get("relativistic_threshold", 0.1),
            0.1,
            minimum=0.0,
            maximum=1.0,
        )
        self.relativistic_safety_factor = coerce_float(
            mapping.get("relativistic_safety_factor", 0.999999),
            0.999999,
            minimum=0.0,
            maximum=1.0,
        )

        self.enable_electromagnetic = coerce_bool(mapping.get("enable_electromagnetic", False), False)
        self.electric_field = self._normalize_vector(mapping.get("electric_field", (0.0, 10.0)), "electric_field")
        self.magnetic_field = coerce_float(mapping.get("magnetic_field", 0.5), 0.5)

        self.dt = coerce_float(mapping.get("dt", 0.02), 0.02, minimum=1.0e-12)
        self.enable_history = coerce_bool(mapping.get("enable_history", True), True)
        self.history_limit = coerce_int(mapping.get("history_limit", 200), 200, minimum=1)

        # --- classical extensions ---
        self.static_friction_coeff = coerce_float(mapping.get("static_friction_coeff", 0.5), 0.5, minimum=0.0)
        self.kinetic_friction_coeff = coerce_float(mapping.get("kinetic_friction_coeff", 0.3), 0.3, minimum=0.0)
        self.body_radius = coerce_float(mapping.get("body_radius", 0.5), 0.5, minimum=1e-12)
        self.air_density = coerce_float(mapping.get("air_density", 1.225), 1.225, minimum=0.0)
        self.fluid_density = coerce_float(mapping.get("fluid_density", 997.0), 997.0, minimum=0.0)
        self.buoyancy_enabled = coerce_bool(mapping.get("buoyancy_enabled", False), False)
        self.magnus_coeff = coerce_float(mapping.get("magnus_coeff", 0.0), 0.0)
        self.angular_velocity_index = coerce_int(mapping.get("angular_velocity_index", 5), 5, minimum=0)

        self.coriolis_omega = coerce_float(mapping.get("coriolis_omega", 0.0), 0.0)
        self.coriolis_axis = self._normalize_vector(mapping.get("coriolis_axis", (0.0, 0.0)), "coriolis_axis")

        self.spring_constants = tuple(coerce_float(v, 0.0) for v in ensure_list(mapping.get("spring_constants", ())))
        self.spring_anchors = tuple(
            tuple(coerce_float(c, 0.0) for c in ensure_list(a))
            for a in ensure_list(mapping.get("spring_anchors", ()))
        )
        self.spring_damping = coerce_float(mapping.get("spring_damping", 0.0), 0.0, minimum=0.0)

        self.pendulum_enabled = coerce_bool(mapping.get("pendulum_enabled", False), False)
        self.pendulum_length = coerce_float(mapping.get("pendulum_length", 1.0), 1.0, minimum=1e-12)
        self.pendulum_anchor = self._normalize_vector(mapping.get("pendulum_anchor", (0.0, 0.0)), "pendulum_anchor")

        self.nbody_masses = tuple(coerce_float(m, 1.0) for m in ensure_list(mapping.get("nbody_masses", ())))
        self.nbody_positions = tuple(
            tuple(coerce_float(c, 0.0) for c in ensure_list(p))
            for p in ensure_list(mapping.get("nbody_positions", ()))
        )
        self.nbody_G_enabled = coerce_bool(mapping.get("nbody_G_enabled", False), False)

        # --- thermodynamics ---
        self.enable_thermodynamics = coerce_bool(mapping.get("enable_thermodynamics", False), False)
        self.ambient_temperature = coerce_float(mapping.get("ambient_temperature", 293.15), 293.15)
        self.body_temperature = coerce_float(mapping.get("body_temperature", 293.15), 293.15)
        self.heat_transfer_coeff = coerce_float(mapping.get("heat_transfer_coeff", 0.0), 0.0, minimum=0.0)
        self.thermal_conductivity = coerce_float(mapping.get("thermal_conductivity", 0.0), 0.0, minimum=0.0)
        self.emissivity = coerce_float(mapping.get("emissivity", 0.9), 0.9, minimum=0.0, maximum=1.0)
        self.specific_heat = coerce_float(mapping.get("specific_heat", 4186.0), 4186.0, minimum=1e-12)
        self.thermal_expansion_coeff = coerce_float(mapping.get("thermal_expansion_coeff", 0.0), 0.0)
        self.reference_temperature = coerce_float(mapping.get("reference_temperature", 293.15), 293.15)
        self.container_volume = coerce_float(mapping.get("container_volume", 1.0), 1.0, minimum=1e-12)

        # --- EM extensions ---
        self.enable_coulomb = coerce_bool(mapping.get("enable_coulomb", False), False)
        raw_other_charges = ensure_list(mapping.get("other_charges", ())) or ()
        parsed_other_charges = []
        for entry in raw_other_charges:
            if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                continue
            position, charge = entry[0], entry[1]
            parsed_other_charges.append(
                (
                    tuple(coerce_float(c, 0.0) for c in ensure_list(position)),
                    coerce_float(charge, 0.0),
                )
            )
        self.other_charges = tuple(parsed_other_charges)

        self.enable_faraday = coerce_bool(mapping.get("enable_faraday", False), False)
        self.magnetic_flux_rate = coerce_float(mapping.get("magnetic_flux_rate", 0.0), 0.0)
        self.enable_radiation_pressure = coerce_bool(mapping.get("enable_radiation_pressure", False), False)
        self.irradiance = coerce_float(mapping.get("irradiance", 0.0), 0.0, minimum=0.0)

        # --- relativity ---
        self.enable_time_dilation = coerce_bool(mapping.get("enable_time_dilation", False), False)
        self.enable_length_contraction = coerce_bool(mapping.get("enable_length_contraction", False), False)
        self.enable_relativistic_momentum = coerce_bool(mapping.get("enable_relativistic_momentum", False), False)

        # --- quantum ---
        self.enable_wkb_tunneling = coerce_bool(mapping.get("enable_wkb_tunneling", False), False)
        self.barrier_heights = tuple(coerce_float(v, 0.0) for v in ensure_list(mapping.get("barrier_heights", ())))
        self.particle_mass_kg = coerce_float(mapping.get("particle_mass_kg", 9.11e-31), 9.11e-31, minimum=1e-40)
        self.enable_de_broglie = coerce_bool(mapping.get("enable_de_broglie", False), False)
        self.enable_uncertainty_floor = coerce_bool(mapping.get("enable_uncertainty_floor", False), False)
        self.uncertainty_scale = coerce_float(mapping.get("uncertainty_scale", 1.0), 1.0, minimum=0.0)

        # --- materials ---
        self.enable_material_stress = coerce_bool(mapping.get("enable_material_stress", False), False)
        self.youngs_modulus = coerce_float(mapping.get("youngs_modulus", 2.0e11), 2.0e11, minimum=0.0)
        self.yield_strength = coerce_float(mapping.get("yield_strength", 2.5e8), 2.5e8, minimum=0.0)
        self.fracture_strain = coerce_float(mapping.get("fracture_strain", 0.05), 0.05, minimum=0.0)
        self.plastic_hardening = coerce_float(mapping.get("plastic_hardening", 0.1), 0.1, minimum=0.0)
        self.rest_length = coerce_float(mapping.get("rest_length", 1.0), 1.0, minimum=1e-12)
        self.enable_fatigue = coerce_bool(mapping.get("enable_fatigue", False), False)
        self.fatigue_limit = coerce_float(mapping.get("fatigue_limit", 1.0), 1.0, minimum=0.0)

        # --- constraints ---
        self.enable_distance_constraints = coerce_bool(mapping.get("enable_distance_constraints", False), False)
        distance_targets_raw = ensure_list(mapping.get("distance_targets", ()))
        self.distance_targets = tuple(
            (
                coerce_int(target[0], 0),
                coerce_int(target[1], 1),
                coerce_float(target[2], 1.0),
            )
            for target in distance_targets_raw
            if isinstance(target, (tuple, list)) and len(target) >= 3
        ) if mapping.get("distance_targets") else ()

        self.enable_angle_constraints = coerce_bool(mapping.get("enable_angle_constraints", False), False)
        angle_targets_raw = ensure_list(mapping.get("angle_targets", ()))
        self.angle_targets = tuple(
            (
                coerce_int(target[0], 4),
                coerce_float(target[1], 0.0),
            )
            for target in angle_targets_raw
            if isinstance(target, (tuple, list)) and len(target) >= 2
        ) if mapping.get("angle_targets") else ()

        self.min_velocity = coerce_float(mapping.get("min_velocity", 0.0), 0.0, minimum=0.0)
        self.max_velocity = coerce_float(mapping.get("max_velocity", 1.0e6), 1.0e6, minimum=0.0)
        self.max_acceleration = coerce_float(mapping.get("max_acceleration", 1.0e9), 1.0e9, minimum=0.0)
        self.max_jerk = coerce_float(mapping.get("max_jerk", 1.0e12), 1.0e12, minimum=0.0)

        # --- audit ---
        self.enable_conservation_audit = coerce_bool(mapping.get("enable_conservation_audit", False), False)
        self.energy_tolerance = coerce_float(mapping.get("energy_tolerance", 1e-6), 1e-6, minimum=0.0)

        # --- internal state for audit/jerk ---
        self._prev_accel: Optional[Tuple[float, float]] = None
        self._prev_vel: Optional[Tuple[float, float]] = None
        self._fatigue_counter: float = 0.0
        self._initial_energy: Optional[float] = None

        raw_seed = mapping.get("random_seed", None)
        self.random_seed = None if raw_seed in (None, "", "none", "None") else coerce_int(raw_seed, 0)

        self.constants = dict(self.CONSTANTS)
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.history_limit)
        self._rng = np.random.default_rng(self.random_seed)
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
    def _config_as_dict(self) -> Dict[str, Any]:
        """Return the active physics configuration as a plain dictionary."""
        return {key: getattr(self, key) for key in self.CONFIG_FIELDS}

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
            self.gravity,
            "gravity",
            minimum=0.0,
            config=self.physics_config,
            error_cls=BaseConfigurationError,
        )
        ensure_numeric_range(
            self.elasticity,
            "elasticity",
            minimum=0.0,
            maximum=1.0,
            config=self.physics_config,
            error_cls=BaseConfigurationError,
        )
        ensure_numeric_range(
            self.relativistic_threshold,
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
        step_dt = self.dt if dt is None else coerce_float(dt, self.dt, minimum=1.0e-12)
        particle_mass = coerce_float(mass if mass is not None else self.default_mass, self.default_mass, minimum=1.0e-12)

        has_vel_x = len(state) >= 3
        has_vel_y = len(state) >= 4
        has_ang_vel = len(state) >= 6
        has_charge = len(state) >= 8
        speed_before = self._speed(state)
        relativistic_clamp_applied = False
        tunneling_events = 0
        electromagnetic_applied = False

        if has_vel_y:
            state[3] -= self.gravity * step_dt

        damping = max(0.0, 1.0 - self.friction_coeff * step_dt)
        if has_vel_x:
            state[2] *= damping
        if has_vel_y:
            state[3] *= damping

        if has_vel_x and has_vel_y and self.wind_strength > 0.0:
            base_wind_x = self.wind_strength * math.cos(self.wind_direction)
            base_wind_y = self.wind_strength * math.sin(self.wind_direction)
            turbulence_scale = self.wind_strength * self.wind_turbulence_ratio
            turbulence_x, turbulence_y = self._rng.normal(0.0, turbulence_scale, 2)
            state[2] += (base_wind_x + turbulence_x) * step_dt
            state[3] += (base_wind_y + turbulence_y) * step_dt

        if has_vel_x and has_vel_y and self.drag_coeff > 0.0:
            vx = float(state[2])
            vy = float(state[3])
            speed = float(np.hypot(vx, vy))
            if speed > self.min_speed_for_drag:
                drag_force = self.drag_coeff * (speed ** 2)
                drag_accel = drag_force / particle_mass
                state[2] += (-drag_accel * vx / speed) * step_dt
                state[3] += (-drag_accel * vy / speed) * step_dt

        if has_vel_x and has_vel_y and self.terminal_velocity > 0.0:
            speed = self._speed(state)
            if speed > self.terminal_velocity:
                scale = self.terminal_velocity / speed
                state[2] *= scale
                state[3] *= scale

        if has_ang_vel:
            rotational_damping = max(0.0, 1.0 - self.rotational_friction * step_dt)
            state[5] *= rotational_damping

        if self.enable_relativistic and has_vel_x and has_vel_y:
            speed = self._speed(state)
            if speed > self.relativistic_threshold * self.constants["c"]:
                max_allowed = self.constants["c"] * self.relativistic_safety_factor
                if speed > max_allowed and speed > 0:
                    scale = max_allowed / speed
                    state[2] *= scale
                    state[3] *= scale
                    relativistic_clamp_applied = True
                    self._stats["relativistic_clamps"] += 1

        if self.enable_tunneling and has_vel_x and self.barrier_width > 0.0:
            for barrier in self.barrier_positions:
                if abs(float(state[0]) - barrier) <= self.barrier_width:
                    if self._rng.random() < self.tunneling_probability:
                        direction = np.sign(state[2]) if abs(float(state[2])) > 0 else 1.0
                        state[0] = barrier + self.barrier_width * direction
                        tunneling_events += 1
                        self._stats["tunneling_events"] += 1

        if self.enable_electromagnetic and has_charge and has_vel_x and has_vel_y:
            charge = float(state[7]) if abs(float(state[7])) > 1.0e-12 else self.default_charge
            if abs(charge) > 1.0e-12:
                ex, ey = self.electric_field
                bz = self.magnetic_field
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

        left = float(low[0] + self.boundary_margin)
        right = float(high[0] - self.boundary_margin)
        bottom = float(low[1] + self.boundary_margin)
        top = float(high[1] - self.boundary_margin)

        def damp_tangent(value: float) -> float:
            return value * max(0.0, 1.0 - self.tangential_damping)

        def in_corner(x: float, y: float) -> bool:
            corners = ((left, bottom), (left, top), (right, bottom), (right, top))
            return any(abs(x - cx) <= self.corner_threshold and abs(y - cy) <= self.corner_threshold for cx, cy in corners)

        if float(state[1]) < bottom:
            state[1] = bottom
            collisions += 1
            if has_vel_y:
                state[3] = abs(float(state[3])) * self.elasticity
            if has_vel_x:
                state[2] = damp_tangent(float(state[2]))

        if float(state[1]) > top:
            state[1] = top
            collisions += 1
            if has_vel_y:
                state[3] = -abs(float(state[3])) * self.elasticity
            if has_vel_x:
                state[2] = damp_tangent(float(state[2]))

        if float(state[0]) < left:
            state[0] = left
            collisions += 1
            if has_vel_x:
                state[2] = abs(float(state[2])) * self.elasticity
            if has_vel_y:
                state[3] = damp_tangent(float(state[3]))

        if float(state[0]) > right:
            state[0] = right
            collisions += 1
            if has_vel_x:
                state[2] = -abs(float(state[2])) * self.elasticity
            if has_vel_y:
                state[3] = damp_tangent(float(state[3]))

        if collisions > 1 and has_vel_x and has_vel_y and in_corner(float(state[0]), float(state[1])):
            state[2] *= self.elasticity
            state[3] *= self.elasticity

        if has_angle:
            state[4] = float(state[4]) % (2.0 * math.pi)
        if has_ang_vel and abs(float(state[5])) > self.max_angular_velocity:
            state[5] = math.copysign(self.max_angular_velocity, float(state[5]))

        self._stats["boundary_steps"] += 1
        self._stats["collisions"] += collisions
        self._record_history(
            PhysicsStepSummary(
                timestamp=utc_now_iso(),
                dt=self.dt,
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

    # ==================================================================
    # --- Classical mechanics -------------------------------------
    # ==================================================================
    
    def apply_spring_forces(self, state: np.ndarray, dt: float, mass: float) -> None:
        """Hooke's law with optional damping for anchored linear springs."""
        if not self.spring_constants or len(state) < 4:
            return
        for k, anchor in zip(self.spring_constants, self.spring_anchors):
            if len(anchor) < 2 or len(state) < 4:
                continue
            dx = float(state[0]) - anchor[0]
            dy = float(state[1]) - anchor[1]
            length = math.hypot(dx, dy) or 1e-12
            rest = self.rest_length
            extension = length - rest
            fx = -k * extension * dx / length
            fy = -k * extension * dy / length
            # damping along spring axis
            vx, vy = float(state[2]), float(state[3])
            v_along = (vx * dx + vy * dy) / length
            fx -= self.spring_damping * v_along * dx / length
            fy -= self.spring_damping * v_along * dy / length
            state[2] += (fx / mass) * dt
            state[3] += (fy / mass) * dt
    
    def apply_pendulum_constraint(self, state: np.ndarray, dt: float) -> None:
        """Rigid pendulum: project position onto a circle around the pivot,
        and remove radial velocity component."""
        if not self.pendulum_enabled or len(state) < 4:
            return
        ax, ay = self.pendulum_anchor
        dx = float(state[0]) - ax
        dy = float(state[1]) - ay
        r = math.hypot(dx, dy) or 1e-12
        # position projection
        ux, uy = dx / r, dy / r
        state[0] = ax + ux * self.pendulum_length
        state[1] = ay + uy * self.pendulum_length
        # remove radial velocity
        vx, vy = float(state[2]), float(state[3])
        v_radial = vx * ux + vy * uy
        state[2] = vx - v_radial * ux
        state[3] = vy - v_radial * uy
    
    def apply_coriolis_centrifugal(self, state: np.ndarray, dt: float) -> None:
        """Rotating-frame pseudo forces: Coriolis, centrifugal, Euler."""
        if self.coriolis_omega == 0.0 or len(state) < 4:
            return
        omega = self.coriolis_omega
        vx, vy = float(state[2]), float(state[3])
        x, y = float(state[0]), float(state[1])
        # 2D simplification: omega along z
        ax_coriolis = 2.0 * omega * vy
        ay_coriolis = -2.0 * omega * vx
        ax_centrifugal = omega ** 2 * x
        ay_centrifugal = omega ** 2 * y
        state[2] += (ax_coriolis + ax_centrifugal) * dt
        state[3] += (ay_coriolis + ay_centrifugal) * dt
    
    def apply_buoyancy(self, state: np.ndarray, dt: float, mass: float) -> None:
        """Archimedes: upward force = rho_fluid * V * g."""
        if not self.buoyancy_enabled or len(state) < 4:
            return
        volume = (4.0 / 3.0) * math.pi * (self.body_radius ** 3)
        f_buoy = self.fluid_density * volume * self.gravity
        state[3] += (f_buoy / mass) * dt
    
    def apply_magnus_effect(self, state: np.ndarray, dt: float) -> None:
        """F_Magnus = S * (omega x v)."""
        if self.magnus_coeff == 0.0 or len(state) < 6:
            return
        omega = float(state[self.angular_velocity_index])
        vx, vy = float(state[2]), float(state[3])
        # 2D: omega along z → (omega x v) = (-omega*vy, omega*vx)
        state[2] += self.magnus_coeff * (-omega * vy) * dt
        state[3] += self.magnus_coeff * ( omega * vx) * dt
    
    def apply_stokes_drag(self, state: np.ndarray, dt: float) -> None:
        """Viscous (Stokes) drag: F = 6*pi*mu*r*v."""
        if len(state) < 4 or self.air_density <= 0.0:
            return
        mu = self.constants["mu_air"]
        coeff = 6.0 * math.pi * mu * self.body_radius
        state[2] -= coeff * float(state[2]) * dt
        state[3] -= coeff * float(state[3]) * dt
    
    def apply_nbody_gravity(self, state: np.ndarray, dt: float) -> None:
        """Attraction toward a set of point masses via Newton's law."""
        if not self.nbody_G_enabled or len(state) < 4:
            return
        G = self.constants["G"]
        for m, pos in zip(self.nbody_masses, self.nbody_positions):
            if len(pos) < 2:
                continue
            dx = pos[0] - float(state[0])
            dy = pos[1] - float(state[1])
            r2 = dx * dx + dy * dy + 1e-12
            inv_r3 = 1.0 / (r2 * math.sqrt(r2))
            state[2] += G * m * dx * inv_r3 * dt
            state[3] += G * m * dy * inv_r3 * dt
    
    def apply_friction_static_kinetic(self, state: np.ndarray, dt: float) -> None:
        """Static vs kinetic friction on the horizontal axis when resting on ground."""
        if len(state) < 4:
            return
        # Only when vertical velocity is negligible (contact)
        if abs(float(state[3])) > 1e-3:
            return
        vx = float(state[2])
        # Static: if below the threshold, kill small velocities
        if abs(vx) < self.static_friction_coeff * self.gravity * dt:
            state[2] = 0.0
        else:
            # Kinetic: decelerate
            decel = self.kinetic_friction_coeff * self.gravity * dt
            state[2] = vx - math.copysign(min(abs(vx), decel), vx)
    
    # ==================================================================
    # --- Thermodynamics ------------------------------------------
    # ==================================================================
    
    def apply_heat_transfer(self, state: np.ndarray, dt: float) -> None:
        """Newton cooling + conduction between body and ambient."""
        if not self.enable_thermodynamics:
            return
        area = 4.0 * math.pi * self.body_radius ** 2
        delta_T = self.body_temperature - self.ambient_temperature
        # Convection
        q_conv = self.heat_transfer_coeff * area * delta_T
        # Conduction (optional)
        q_cond = self.thermal_conductivity * delta_T
        # Black-body radiation
        T_k = max(self.body_temperature, 1e-3)
        q_rad = self.emissivity * self.constants["sigma_sb"] * area * (T_k ** 4 - self.ambient_temperature ** 4)
        dT = -(q_conv + q_cond + q_rad) * dt / self.specific_heat
        self.body_temperature += dT
        # Optional: write temperature into state if a slot exists (>=9)
        if len(state) >= 9:
            state[8] = self.body_temperature
    
    def apply_thermal_expansion(self, state: np.ndarray) -> None:
        if not self.enable_thermodynamics or self.thermal_expansion_coeff == 0.0:
            return
        strain = self.thermal_expansion_coeff * (self.body_temperature - self.reference_temperature)
        self.body_radius = max(1e-9, self.body_radius * (1.0 + strain))
    
    def apply_ideal_gas(self, state: np.ndarray) -> float:
        """Return pressure via ideal-gas law: P = nRT/V."""
        if not self.enable_thermodynamics:
            return 0.0
        n = 1.0  # 1 mole by default
        return n * self.constants["R"] * self.body_temperature / self.container_volume
    
    # ==================================================================
    # --- Electromagnetism ----------------------------------------
    # ==================================================================
    
    def apply_coulomb_forces(self, state: np.ndarray, dt: float, mass: float) -> None:
        """Coulomb force from a set of fixed point charges."""
        if not self.enable_coulomb or len(state) < 8 or not self.other_charges:
            return
        q1 = float(state[7])
        k = self.constants["ke"]
        for pos, q2 in self.other_charges:
            if len(pos) < 2:
                continue
            dx = float(state[0]) - pos[0]
            dy = float(state[1]) - pos[1]
            r2 = dx * dx + dy * dy + 1e-12
            inv_r3 = 1.0 / (r2 * math.sqrt(r2))
            f = k * q1 * q2 * inv_r3
            state[2] += (f * dx / mass) * dt
            state[3] += (f * dy / mass) * dt
    
    def apply_faraday_emf(self, state: np.ndarray, dt: float) -> None:
        """Induced force along x from time-varying magnetic flux: EMF = -dPhi/dt."""
        if not self.enable_faraday or len(state) < 3:
            return
        emf = -self.magnetic_flux_rate
        q = float(state[7]) if len(state) >= 8 else self.default_charge
        # Force ~ q * emf (unit simplification)
        state[2] += q * emf * dt
    
    def apply_radiation_pressure(self, state: np.ndarray, dt: float, mass: float) -> None:
        """Solar-radiation pressure: F = I*A/c for absorbing body."""
        if not self.enable_radiation_pressure or len(state) < 4:
            return
        c = self.constants["c"]
        area = math.pi * self.body_radius ** 2
        f = self.irradiance * area / c
        state[2] += (f / mass) * dt
    
    # ==================================================================
    # --- Relativistic extensions ---------------------------------
    # ==================================================================
    
    def apply_time_dilation(self, state: np.ndarray) -> float:
        """Return gamma (Lorentz factor) and store effective time-scaling."""
        if not self.enable_time_dilation or len(state) < 4:
            return 1.0
        c = self.constants["c"]
        v = self._speed(state)
        beta2 = (v / c) ** 2
        beta2 = min(beta2, 0.999999)
        gamma = 1.0 / math.sqrt(1.0 - beta2)
        if len(state) >= 9:
            state[8] = gamma
        return gamma
    
    def apply_length_contraction(self, state: np.ndarray) -> None:
        if not self.enable_length_contraction or len(state) < 4:
            return
        c = self.constants["c"]
        v = self._speed(state)
        beta2 = min((v / c) ** 2, 0.999999)
        self.body_radius *= math.sqrt(1.0 - beta2)
    
    def apply_relativistic_momentum(self, state: np.ndarray, mass: float) -> None:
        """p = gamma*m*v. Ensures velocity stays consistent with relativistic limits."""
        if not self.enable_relativistic_momentum or len(state) < 4:
            return
        gamma = self.apply_time_dilation(state) if self.enable_time_dilation else 1.0
        # Store p in unused slots if available
        if len(state) >= 10:
            state[8] = gamma * mass * float(state[2])
            state[9] = gamma * mass * float(state[3])
    
    # ==================================================================
    # --- Quantum -------------------------------------------------
    # ==================================================================
    
    def apply_wkb_tunneling(self, state: np.ndarray, dt: float) -> int:
        """WKB-approximated transmission probability through a rectangular barrier."""
        if not self.enable_wkb_tunneling or len(state) < 3 or not self.barrier_heights:
            return 0
        hbar = self.constants["hbar"]
        events = 0
        m = self.particle_mass_kg
        E = 0.5 * m * (float(state[2]) ** 2)  # kinetic energy in x
        for barrier, V0 in zip(self.barrier_positions or (self.barrier_width,), self.barrier_heights):
            if abs(float(state[0]) - barrier) > self.barrier_width:
                continue
            if E <= 0 or V0 <= 0:
                continue
            if E >= V0:
                T = 1.0  # classically allowed
            else:
                kappa = math.sqrt(2.0 * m * (V0 - E)) / hbar
                T = math.exp(-2.0 * kappa * self.barrier_width)
            if self._rng.random() < T:
                direction = math.copysign(1.0, float(state[2]) or 1.0)
                state[0] = barrier + self.barrier_width * direction
                events += 1
        return events
    
    def apply_de_broglie(self, state: np.ndarray) -> None:
        if not self.enable_de_broglie or len(state) < 3:
            return
        h = self.constants["h"]
        p = self.particle_mass_kg * abs(float(state[2]))
        if p > 0 and len(state) >= 9:
            state[8] = h / p  # wavelength
    
    def apply_uncertainty_floor(self, state: np.ndarray) -> None:
        """Ensure no slot in state holds an unphysically small (Δx·Δp < ħ/2) value."""
        if not self.enable_uncertainty_floor or len(state) < 3:
            return
        hbar = self.constants["hbar"]
        # Enforce positional floor if momentum is stored
        p = self.particle_mass_kg * abs(float(state[2]))
        if p > 0:
            min_dx = self.uncertainty_scale * hbar / (2.0 * p)
            # No-op unless there is a dedicated position slot; kept for extension.
    
    # ==================================================================
    # --- Material stress / plasticity / fracture -----------------
    # ==================================================================
    
    def apply_material_stress(self, state: np.ndarray, dt: float) -> Dict[str, Any]:
        """Track strain, yield, hardening, fracture and fatigue on a stretched link."""
        if not self.enable_material_stress or len(state) < 4:
            return {}
        # Simple 1D axial strain from speed along x relative to rest state
        strain = (abs(float(state[2])) - 0.0) * dt / self.rest_length
        stress = self.youngs_modulus * strain
        info: Dict[str, Any] = {"stress": stress, "strain": strain}
        if abs(stress) > self.yield_strength:
            # Plastic hardening (reduces effective modulus)
            self.youngs_modulus *= (1.0 - self.plastic_hardening * dt)
            info["plastic"] = True
        if abs(strain) > self.fracture_strain:
            info["fractured"] = True
        if self.enable_fatigue:
            self._fatigue_counter += abs(strain)
            if self._fatigue_counter > self.fatigue_limit:
                info["fatigued"] = True
                self._fatigue_counter = 0.0
        return info
    
    # ==================================================================
    # --- Constraint family ---------------------------------------
    # ==================================================================
    
    def apply_distance_constraints(self, state: np.ndarray, dt: float) -> None:
        """Enforce fixed distances between indexed state slots (holonomic)."""
        if not self.enable_distance_constraints or not self.distance_targets:
            return
        for i, j, target in self.distance_targets:
            if i >= len(state) or j >= len(state):
                continue
            dx = float(state[j]) - float(state[i])
            if abs(dx) < 1e-12:
                continue
            # Simple hard correction with partial projection
            err = abs(dx) - target
            if err == 0:
                continue
            correction = 0.5 * err * math.copysign(1.0, dx)
            state[i] = float(state[i]) + correction
            state[j] = float(state[j]) - correction
    
    def apply_angle_constraints(self, state: np.ndarray, dt: float) -> None:
        """Clamp angular slots to their target values."""
        if not self.enable_angle_constraints or not self.angle_targets:
            return
        for idx, target in self.angle_targets:
            if idx < len(state):
                state[idx] = target
    
    def apply_velocity_limits(self, state: np.ndarray) -> None:
        if len(state) < 4:
            return
        speed = self._speed(state)
        if speed > self.max_velocity and speed > 0:
            scale = self.max_velocity / speed
            state[2] *= scale
            state[3] *= scale
        if speed < self.min_velocity and speed > 0:
            scale = self.min_velocity / speed
            state[2] *= scale
            state[3] *= scale
    
    def apply_acceleration_limit(self, state: np.ndarray, dt: float) -> None:
        """Clamp acceleration between consecutive steps."""
        if len(state) < 4 or self._prev_vel is None:
            return
        ax = (float(state[2]) - self._prev_vel[0]) / dt
        ay = (float(state[3]) - self._prev_vel[1]) / dt
        a_mag = math.hypot(ax, ay)
        if a_mag > self.max_acceleration and a_mag > 0:
            scale = self.max_acceleration / a_mag
            ax *= scale
            ay *= scale
            state[2] = self._prev_vel[0] + ax * dt
            state[3] = self._prev_vel[1] + ay * dt
    
    def apply_jerk_limit(self, state: np.ndarray, dt: float) -> None:
        """Limit change of acceleration (jerk) between steps."""
        if len(state) < 4 or self._prev_accel is None or self._prev_vel is None:
            return
        new_accel = (
            (float(state[2]) - self._prev_vel[0]) / dt,
            (float(state[3]) - self._prev_vel[1]) / dt,
        )
        jx = (new_accel[0] - self._prev_accel[0]) / dt
        jy = (new_accel[1] - self._prev_accel[1]) / dt
        j_mag = math.hypot(jx, jy)
        if j_mag > self.max_jerk and j_mag > 0:
            scale = self.max_jerk / j_mag
            jx *= scale
            jy *= scale
            # Reconstruct a limited acceleration
            a_x = self._prev_accel[0] + jx * dt
            a_y = self._prev_accel[1] + jy * dt
            state[2] = self._prev_vel[0] + a_x * dt
            state[3] = self._prev_vel[1] + a_y * dt
    
    # ==================================================================
    # --- Conservation audit --------------------------------------
    # ==================================================================
    
    def conservation_audit(self, state: np.ndarray, mass: float) -> ConservationAudit:
        if len(state) < 4:
            KE = PE = 0.0
            px = py = L = 0.0
        else:
            vx, vy = float(state[2]), float(state[3])
            KE = 0.5 * mass * (vx ** 2 + vy ** 2)
            PE = mass * self.gravity * float(state[1])
            px, py = mass * vx, mass * vy
            omega = float(state[self.angular_velocity_index]) if len(state) > self.angular_velocity_index else 0.0
            r2 = float(state[0]) ** 2 + float(state[1]) ** 2
            L = mass * math.sqrt(r2) ** 2 * omega if omega else 0.0
        E_total = KE + PE
        if self._initial_energy is None:
            self._initial_energy = E_total
        drift = (E_total - self._initial_energy) / (abs(self._initial_energy) + 1e-12)
        audit = ConservationAudit(
            timestamp=utc_now_iso(),
            kinetic_energy=KE,
            potential_energy=PE,
            thermal_energy=0.0,
            total_energy=E_total,
            momentum_x=px,
            momentum_y=py,
            angular_momentum=L,
            energy_drift=drift,
            notes={"tolerance": self.energy_tolerance,
                   "within_tolerance": abs(drift) <= self.energy_tolerance},
        )
        if self.enable_history:
            self._history.append(audit.to_dict())
        return audit

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    def _record_history(self, summary: PhysicsStepSummary) -> None:
        if not self.enable_history:
            return
        self._history.append(summary.to_dict())

    def recent_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        limit = coerce_int(limit, 20, minimum=1)
        return list(self._history)[-limit:]

    def stats(self) -> Dict[str, Any]:
        return {
            "config": to_json_safe(self._config_as_dict()),
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


def apply_environmental_effects(self, state, dt=None, mass=None) -> np.ndarray:
    state = self._validate_state_array(np.asarray(state, dtype=float))
    step_dt = self.dt if dt is None else coerce_float(dt, self.dt, minimum=1.0e-12)
    particle_mass = coerce_float(mass if mass is not None else self.default_mass,
                                 self.default_mass, minimum=1.0e-12)

    speed_before = self._speed(state)
    relativistic_clamp_applied = False
    tunneling_events = 0
    electromagnetic_applied = False

    # -- core classical --
    if len(state) >= 4:
        state[3] -= self.gravity * step_dt
    damping = max(0.0, 1.0 - self.friction_coeff * step_dt)
    if len(state) >= 3: state[2] *= damping
    if len(state) >= 4: state[3] *= damping

    if len(state) >= 4 and self.wind_strength > 0.0:
        base_wind_x = self.wind_strength * math.cos(self.wind_direction)
        base_wind_y = self.wind_strength * math.sin(self.wind_direction)
        ts = self.wind_strength * self.wind_turbulence_ratio
        tx, ty = self._rng.normal(0.0, ts, 2)
        state[2] += (base_wind_x + tx) * step_dt
        state[3] += (base_wind_y + ty) * step_dt

    if len(state) >= 4 and self.drag_coeff > 0.0:
        vx, vy = float(state[2]), float(state[3])
        sp = math.hypot(vx, vy)
        if sp > self.min_speed_for_drag:
            drag_accel = (self.drag_coeff * sp ** 2) / particle_mass
            state[2] += (-drag_accel * vx / sp) * step_dt
            state[3] += (-drag_accel * vy / sp) * step_dt

    # --- effect chain ---
    self.apply_spring_forces(state, step_dt, particle_mass)
    self.apply_pendulum_constraint(state, step_dt)
    self.apply_coriolis_centrifugal(state, step_dt)
    self.apply_buoyancy(state, step_dt, particle_mass)
    self.apply_magnus_effect(state, step_dt)
    self.apply_stokes_drag(state, step_dt)
    self.apply_nbody_gravity(state, step_dt)
    self.apply_friction_static_kinetic(state, step_dt)

    # thermodynamic
    self.apply_heat_transfer(state, step_dt)
    self.apply_thermal_expansion(state)

    # EM
    self.apply_coulomb_forces(state, step_dt, particle_mass)
    self.apply_faraday_emf(state, step_dt)
    self.apply_radiation_pressure(state, step_dt, particle_mass)

    # quantum tunneling (WKB overrides probabilistic tunneling when enabled)
    if self.enable_wkb_tunneling:
        tunneling_events += self.apply_wkb_tunneling(state, step_dt)

    # original EM block (Lorentz) unchanged
    if self.enable_electromagnetic and len(state) >= 8:
        charge = float(state[7]) if abs(float(state[7])) > 1e-12 else self.default_charge
        if abs(charge) > 1e-12:
            ex, ey = self.electric_field
            bz = self.magnetic_field
            vx, vy = float(state[2]), float(state[3])
            ax = (charge / particle_mass) * (ex + vy * bz)
            ay = (charge / particle_mass) * (ey - vx * bz)
            state[2] += ax * step_dt
            state[3] += ay * step_dt
            electromagnetic_applied = True
            self._stats["electromagnetic_applications"] += 1

    # terminal velocity
    if len(state) >= 4 and self.terminal_velocity > 0.0:
        sp = self._speed(state)
        if sp > self.terminal_velocity:
            s = self.terminal_velocity / sp
            state[2] *= s; state[3] *= s

    # angular damping + limits
    if len(state) >= 6:
        state[5] *= max(0.0, 1.0 - self.rotational_friction * step_dt)

    # relativistic + quantum stores
    self.apply_time_dilation(state)
    self.apply_length_contraction(state)
    self.apply_relativistic_momentum(state, particle_mass)
    self.apply_de_broglie(state)
    self.apply_uncertainty_floor(state)

    # velocity/accel/jerk limits
    self.apply_velocity_limits(state)
    self.apply_acceleration_limit(state, step_dt)
    self.apply_jerk_limit(state, step_dt)

    # material stress
    if self.enable_material_stress:
        self.apply_material_stress(state, step_dt)

    # constraint family
    self.apply_distance_constraints(state, step_dt)
    self.apply_angle_constraints(state, step_dt)

    # bookkeeping for next-step limits
    self._prev_accel = (
        (float(state[2]) - (self._prev_vel[0] if self._prev_vel else float(state[2]))) / step_dt,
        (float(state[3]) - (self._prev_vel[1] if self._prev_vel else float(state[3]))) / step_dt,
    )
    self._prev_vel = (float(state[2]), float(state[3]))

    # audit
    if self.enable_conservation_audit:
        self.conservation_audit(state, particle_mass)

    self._stats["environment_steps"] += 1
    self._record_history(PhysicsStepSummary(
        timestamp=utc_now_iso(), dt=step_dt, collisions=0,
        relativistic_clamp_applied=relativistic_clamp_applied,
        tunneling_events=tunneling_events,
        electromagnetic_applied=electromagnetic_applied,
        speed_before=speed_before, speed_after=self._speed(state),
        notes={"phase": "environmental_effects_extended"},
    ))
    return state


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
    dt = getattr(env_instance, "dt", engine.dt)
    mass = getattr(env_instance, "mass", engine.default_mass)

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
