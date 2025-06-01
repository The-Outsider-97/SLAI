"""
Physics constraints and environmental effects for the SLAI Environment.

This module provides functions to simulate physical phenomena such as
gravity, friction, and boundary collisions within the SLAIEnv. These functions
are intended to be called as part of the environment's step simulation
when SLAIEnv is not wrapping a pre-existing Gym environment (i.e., self.env is None).

The state vector is assumed to have components at specific indices:
- state_array[0]: x-position
- state_array[1]: y-position (if state_dim >= 2)
- state_array[2]: x-velocity (if state_dim >= 3)
- state_array[3]: y-velocity (if state_dim >= 4)
"""

import numpy as np

def apply_constants(env_instance):
    """
    Registers universal physical constants in the environment instance for access and simulation logic.
    Includes constants from https://en.wikipedia.org/wiki/List_of_physical_constants
    """
    constants = {
        # Gravitational and related constants
        "G": 6.67430e-11,                     # Gravitational constant (m^3·kg^−1·s^−2)
        "g": 9.80665,                         # Standard gravity (m/s^2)
        "lambda": 1.1056e-52,                 # Cosmological constant (1/m^2)

        # Electromagnetic constants
        "epsilon0": 8.8541878128e-12,         # Vacuum permittivity (F/m)
        "mu0": 1.25663706212e-6,              # Vacuum permeability (N/A^2)
        "ke": 8.9875517923e9,                 # Coulomb constant (N·m^2·C^−2)
        "c": 299792458,                       # Speed of light in vacuum (m/s)
        "e": 1.602176634e-19,                 # Elementary charge (C)
        "alpha": 7.2973525693e-3,             # Fine-structure constant (dimensionless)
        "phi0": 2.067833848e-15,              # Magnetic flux quantum (Wb)

        # Thermodynamic constants
        "kB": 1.380649e-23,                   # Boltzmann constant (J/K)
        "R": 8.31446261815324,                # Ideal gas constant (J/mol·K)
        "NA": 6.02214076e23,                  # Avogadro constant (mol^−1)
        "F": 96485.3321233100184,             # Faraday constant (C/mol)
        "sigma": 5.670374419e-8,              # Stefan–Boltzmann constant (W/m²·K⁴)
        "kW": 2.897771955e-3,                 # Wien displacement constant (m·K)

        # Quantum constants
        "h": 6.62607015e-34,                  # Planck constant (J·s)
        "hbar": 1.054571817e-34,              # Reduced Planck constant (J·s)

        # Particle masses
        "me": 9.1093837015e-31,               # Electron mass (kg)
        "mp": 1.67262192369e-27,              # Proton mass (kg)
        "mn": 1.67492749804e-27,              # Neutron mass (kg)
        "u": 1.66053906660e-27,               # Atomic mass unit (kg)

        # Time and frequency
        "Hz": 1.0,                            # Hertz (s^−1)
        "day": 86400,                         # Day in seconds
        "year": 31557600,                     # Julian year (s)

        # Temperature and pressure
        "T0": 273.15,                         # Standard temperature (K)
        "P0": 101325,                         # Standard atmospheric pressure (Pa)

        # Other constants
        "Z0": 376.730313668,                  # Vacuum impedance (ohm)
        "lP": 1.616255e-35,                   # Planck length (m)
        "tP": 5.391247e-44,                   # Planck time (s)
        "mP": 2.176434e-8,                    # Planck mass (kg)
        "TP": 1.416784e32                     # Planck temperature (K)
    }

    env_instance.constants = constants
    return constants

def apply_environmental_effects(env_instance, state_array):
    """
    Applies comprehensive environmental effects to the given state array.
    Including relativistic corrections and quantum effects.
    """
    dt = env_instance.dt
    c = env_instance.constants["c"]
    hbar = env_instance.constants["hbar"]

    # Gravity effect on y-velocity
    if len(state_array) >= 4:
        state_array[3] -= env_instance.gravity * dt
        
    # Relativistic corrections at high velocities
    if len(state_array) >= 4:
        vx, vy = state_array[2], state_array[3]
        speed = np.sqrt(vx**2 + vy**2)
        
        # Apply Lorentz factor for relativistic speeds (>10% c)
        if speed > 0.1 * c:
            gamma = 1 / np.sqrt(1 - (speed**2)/(c**2))
            state_array[2] *= gamma
            state_array[3] *= gamma
    
    # Quantum tunneling probability for thin barriers
    if len(state_array) >= 2:
        barrier_width = 0.2  # meters
        barrier_height = 10.0  # Joules
        mass = 1.0  # kg (assumed particle mass)
        
        # Calculate tunneling probability
        k = np.sqrt(2 * mass * barrier_height) / hbar
        tunneling_prob = np.exp(-2 * k * barrier_width)
        
        # Randomly apply tunneling effect
        if np.random.random() < tunneling_prob:
            # Teleport to other side of barrier
            state_array[0] += barrier_width * 2 * np.sign(state_array[2])
    
    # Electromagnetic forces if charge is present
    if len(state_array) >= 8:  # If charge component exists
        charge = state_array[7]
        if abs(charge) > 1e-9:  # Significant charge
            E_field = np.array([0, 10])  # V/m (simulated electric field)
            state_array[2:4] += (charge * E_field * dt / mass)

    # Linear friction (viscous damping)
    if len(state_array) >= 3:
        state_array[2] *= (1 - env_instance.friction_coeff)
    if len(state_array) >= 4:
        state_array[3] *= (1 - env_instance.friction_coeff)

    # Wind forces (directional with turbulence)
    if len(state_array) >= 4:
        # Base wind vector + random turbulence
        wind_x = env_instance.wind_strength * np.cos(env_instance.wind_direction)
        wind_y = env_instance.wind_strength * np.sin(env_instance.wind_direction)
        turbulence = np.random.normal(0, 0.1 * env_instance.wind_strength, 2)

        # Apply wind acceleration
        state_array[2] += (wind_x + turbulence[0]) * dt
        state_array[3] += (wind_y + turbulence[1]) * dt

    # Air resistance (quadratic drag)
    if len(state_array) >= 4:
        vx, vy = state_array[2], state_array[3]
        speed = np.sqrt(vx**2 + vy**2)
        if speed > 0.01:  # Avoid division by zero
            drag_magnitude = env_instance.drag_coeff * speed**2
            drag_x = -drag_magnitude * vx / speed
            drag_y = -drag_magnitude * vy / speed
            state_array[2] += drag_x * dt
            state_array[3] += drag_y * dt

    # Terminal velocity limit
    if len(state_array) >= 4:
        vx, vy = state_array[2], state_array[3]
        speed = np.sqrt(vx**2 + vy**2)
        if speed > env_instance.terminal_velocity:
            scale = env_instance.terminal_velocity / speed
            state_array[2] *= scale
            state_array[3] *= scale

    # Rotational effects (if angular components exist)
    if len(state_array) >= 6:
        # Angular damping
        state_array[5] *= (1 - env_instance.rotational_friction)
        
        # Conservation of angular momentum
        if len(state_array) >= 7:  # If moment of inertia is included
            I = state_array[6]
            if I > 0:
                state_array[5] = state_array[5] * I / (I + 0.1)  # Simplified model

    # Friction effect on x-velocity (state_array[2])
    # This is applied if the state vector includes an x-velocity component.
    # `len(state_array) >= 3` means state_array has at least 3 elements (indices 0, 1, 2).
    if len(state_array) >= 3:
        state_array[2] *= (1 - env_instance.friction_coeff)

    # Friction effect on y-velocity (state_array[3])
    # This is applied if the state vector includes a y-velocity component.
    if len(state_array) >= 4:
        state_array[3] *= (1 - env_instance.friction_coeff)

    return state_array

def enforce_physics_constraints(env_instance, state_array):
    """
    Enforces physics constraints such as boundary collisions (walls, ground, ceiling).
    Handles inelastic bouncing by reversing and damping velocity components upon collision.

    NOTE: This function replaces the _enforce_physics_constraints method
    in SLAIEnv. Any changes here should be reflected in both locations,
    though this is the canonical implementation.

    Args:
        env_instance: An instance of the SLAIEnv class, providing access to
                      `observation_space` (for boundaries) and `elasticity`.
        state_array (np.ndarray): The current state vector of the environment.
                                  This array will be modified in-place.

    Returns:
        np.ndarray: The modified state_array with physics constraints enforced.
    """
    obs_space = env_instance.observation_space
    elasticity= env_instance.elasticity
    energy_loss = 1 - elasticity  # Energy lost during collision
    corner_threshold = 0.05  # Distance threshold for corner detection
    c = env_instance.constants["c"]

    # Helper function to detect corner collisions
    def is_in_corner(x, y):
        corners = [
            (obs_space.low[0], obs_space.low[1]),  # Bottom-left
            (obs_space.low[0], obs_space.high[1]), # Top-left
            (obs_space.high[0], obs_space.low[1]), # Bottom-right
            (obs_space.high[0], obs_space.high[1]) # Top-right
        ]
        for cx, cy in corners:
            if abs(x - cx) < corner_threshold and abs(y - cy) < corner_threshold:
                return True
        return False
    
    # Ground collision
    if len(state_array) >= 2 and obs_space.low.shape[0] >= 2:
        ground_level = obs_space.low[1] + 0.01
        if state_array[1] < ground_level:
            # Special handling for corners
            if is_in_corner(state_array[0], state_array[1]):
                # Diagonal bounce
                state_array[0] = max(state_array[0], obs_space.low[0] + 0.01)
                state_array[1] = ground_level
                if len(state_array) >= 4:
                    # Reverse both components with energy loss
                    state_array[2] = -state_array[2] * elasticity
                    state_array[3] = -state_array[3] * elasticity
            else:
                # Standard ground collision
                state_array[1] = ground_level
                if len(state_array) >= 4:
                    # Energy conservation: conserve horizontal velocity
                    state_array[3] = -state_array[3] * elasticity
                    # Apply friction to horizontal movement
                    state_array[2] *= (1 - energy_loss)
    
    # Ceiling collision (similar to ground)
    if len(state_array) >= 2 and obs_space.high.shape[0] >= 2:
        ceiling_level = obs_space.high[1] - 0.01
        if state_array[1] > ceiling_level:
            if is_in_corner(state_array[0], state_array[1]):
                state_array[0] = min(state_array[0], obs_space.high[0] - 0.01)
                state_array[1] = ceiling_level
                if len(state_array) >= 4:
                    state_array[2] = -state_array[2] * elasticity
                    state_array[3] = -state_array[3] * elasticity
            else:
                state_array[1] = ceiling_level
                if len(state_array) >= 4:
                    state_array[3] = -state_array[3] * elasticity
                    state_array[2] *= (1 - energy_loss)
    
    # Left wall collision
    if len(state_array) >= 1 and obs_space.low.shape[0] >= 1:
        left_wall = obs_space.low[0] + 0.01
        if state_array[0] < left_wall:
            if is_in_corner(state_array[0], state_array[1]):
                state_array[0] = left_wall
                state_array[1] = min(max(state_array[1], obs_space.low[1] + 0.01), 
                                     obs_space.high[1] - 0.01)
                if len(state_array) >= 4:
                    state_array[2] = -state_array[2] * elasticity
                    state_array[3] = -state_array[3] * elasticity
            else:
                state_array[0] = left_wall
                if len(state_array) >= 3:
                    state_array[2] = -state_array[2] * elasticity
                    if len(state_array) >= 4:
                        state_array[3] *= (1 - energy_loss)
    
    # Right wall collision (similar to left)
    if len(state_array) >= 1 and obs_space.high.shape[0] >= 1:
        right_wall = obs_space.high[0] - 0.01
        if state_array[0] > right_wall:
            if is_in_corner(state_array[0], state_array[1]):
                state_array[0] = right_wall
                state_array[1] = min(max(state_array[1], obs_space.low[1] + 0.01), 
                                     obs_space.high[1] - 0.01)
                if len(state_array) >= 4:
                    state_array[2] = -state_array[2] * elasticity
                    state_array[3] = -state_array[3] * elasticity
            else:
                state_array[0] = right_wall
                if len(state_array) >= 3:
                    state_array[2] = -state_array[2] * elasticity
                    if len(state_array) >= 4:
                        state_array[3] *= (1 - energy_loss)
    
    # Angular constraints (if angular components exist)
    if len(state_array) >= 5:
        # Normalize angle to [0, 2π)
        state_array[4] = state_array[4] % (2 * np.pi)
        
        # Angular velocity constraints
        if len(state_array) >= 6:
            max_angular_velocity = 5.0  # rad/s
            if abs(state_array[5]) > max_angular_velocity:
                state_array[5] = np.sign(state_array[5]) * max_angular_velocity
    
    # Energy conservation constraint
    if len(state_array) >= 4:
        # Calculate kinetic energy
        kinetic_energy = 0.5 * (state_array[2]**2 + state_array[3]**2)

        # Gradually dissipate energy over time
        state_array[2] *= (1 - 0.001 * energy_loss)
        state_array[3] *= (1 - 0.001 * energy_loss)

    # Relativistic length contraction
    if len(state_array) >= 4:
        vx, vy = state_array[2], state_array[3]
        speed = np.sqrt(vx**2 + vy**2)

        if speed > 0.1 * c:
            gamma = 1 / np.sqrt(1 - (speed**2)/(c**2))
            # Contract dimensions perpendicular to motion
            state_array[0] /= gamma  # Length contraction in x-direction

    # Quantum barrier penetration (tunneling check)
    barrier_positions = [-8.0, 8.0]  # Example barrier positions
    barrier_width = 0.1
    for barrier in barrier_positions:
        if abs(state_array[0] - barrier) < barrier_width:
            # Calculate tunneling probability (simplified)
            tunneling_prob = 0.05
            if np.random.random() < tunneling_prob:
                # Quantum tunnel through barrier
                state_array[0] = barrier + barrier_width * np.sign(state_array[0] - barrier)

    # Electromagnetic containment if charge present
    if len(state_array) >= 8 and abs(state_array[7]) > 1e-9:
        charge = state_array[7]
        # Simple magnetic confinement (circular motion)
        B_field = 0.5  # Tesla (magnetic field strength)
        radius = abs(env_instance.mass * state_array[2] / (charge * B_field))
        state_array[0] = radius * np.cos(state_array[4])  # x-position
        state_array[1] = radius * np.sin(state_array[4])  # y-position

    return state_array
