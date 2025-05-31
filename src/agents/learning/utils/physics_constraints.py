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

def apply_environmental_effects(env_instance, state_array):
    """
    Applies gravity and friction effects to the given state array.

    Note: This function modifies `state_array` in-place.

    Args:
        env_instance: An instance of the SLAIEnv class, providing access to
                      environment parameters like `gravity`, `dt` (time step),
                      `friction_coeff`. The length of the state vector is
                      implicitly defined by `env_instance.state_dim`, but
                      checks here use `len(state_array)`.
        state_array (np.ndarray): The current state vector of the environment.
                                  This array will be modified in-place.

    Returns:
        np.ndarray: The modified state_array with environmental effects applied.
    """
    # Gravity effect on y-velocity (state_array[3])
    # This is applied if the state vector includes a y-velocity component.
    # `len(state_array) >= 4` means state_array has at least 4 elements (indices 0, 1, 2, 3).
    if len(state_array) >= 4:
        state_array[3] -= env_instance.gravity * env_instance.dt
        
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

    Note: This function modifies `state_array` in-place.

    Args:
        env_instance: An instance of the SLAIEnv class, providing access to
                      `observation_space` (for boundaries) and `elasticity`.
        state_array (np.ndarray): The current state vector of the environment.
                                  This array will be modified in-place.

    Returns:
        np.ndarray: The modified state_array with physics constraints enforced.
    """
    obs_space = env_instance.observation_space
    elasticity_coeff = env_instance.elasticity

    # Ground collision
    # Checks if y-position (state_array[1]) exists and is below ground.
    # `len(state_array) >= 2` ensures state_array[1] is a valid index.
    # `obs_space.low.shape[0] >= 2` ensures obs_space.low[1] is valid.
    if len(state_array) >= 2 and obs_space.low.shape[0] >= 2:
        ground_level = obs_space.low[1] + 0.01  # Small offset to prevent sticking
        if state_array[1] < ground_level:
            state_array[1] = ground_level
            # If y-velocity (state_array[3]) exists, reverse and dampen it (bounce).
            # `len(state_array) >= 4` ensures state_array[3] is valid.
            if len(state_array) >= 4:
                state_array[3] = -state_array[3] * elasticity_coeff
                
    # Ceiling collision
    # Checks if y-position (state_array[1]) exists and is above ceiling.
    if len(state_array) >= 2 and obs_space.high.shape[0] >= 2:
        ceiling_level = obs_space.high[1] - 0.01  # Small offset
        if state_array[1] > ceiling_level:
            state_array[1] = ceiling_level
            # If y-velocity (state_array[3]) exists, reverse and dampen it.
            if len(state_array) >= 4:
                state_array[3] = -state_array[3] * elasticity_coeff
                
    # Left Wall collision
    # Checks if x-position (state_array[0]) exists and is beyond left wall.
    # `len(state_array) >= 1` ensures state_array[0] is valid.
    # `obs_space.low.shape[0] >= 1` ensures obs_space.low[0] is valid.
    if len(state_array) >= 1 and obs_space.low.shape[0] >= 1:
        left_wall_pos = obs_space.low[0] + 0.01  # Small offset
        if state_array[0] < left_wall_pos:
            state_array[0] = left_wall_pos
            # If x-velocity (state_array[2]) exists, reverse and dampen it.
            # `len(state_array) >= 3` ensures state_array[2] is valid.
            if len(state_array) >= 3:
                state_array[2] = -state_array[2] * elasticity_coeff
                
    # Right Wall collision
    # Checks if x-position (state_array[0]) exists and is beyond right wall.
    if len(state_array) >= 1 and obs_space.high.shape[0] >= 1:
        right_wall_pos = obs_space.high[0] - 0.01  # Small offset
        if state_array[0] > right_wall_pos:
            state_array[0] = right_wall_pos
            # If x-velocity (state_array[2]) exists, reverse and dampen it.
            if len(state_array) >= 3:
                state_array[2] = -state_array[2] * elasticity_coeff
                
    return state_array
