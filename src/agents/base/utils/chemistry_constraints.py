chemistry_constraints

"""
Chemical constraints and reaction dynamics for the SLAI Environment.

This module provides functions to simulate chemical phenomena such as
reaction kinetics, equilibrium, temperature changes due to reactions,
pH effects, and catalysis. These functions are intended to be called
as part of the environment's step simulation when SLAIEnv is not
wrapping a pre-existing Gym environment (i.e., self.env is None).

The state vector is assumed to represent a chemical system.
A typical example for a reaction A + B <=> C + D (with catalyst Cat):
- state_array[0]: Concentration of A (mol/L)
- state_array[1]: Concentration of B (mol/L)
- state_array[2]: Concentration of C (mol/L)
- state_array[3]: Concentration of D (mol/L)
- state_array[4]: Concentration of Catalyst (mol/L)
- state_array[5]: Temperature (K)
- state_array[6]: Concentration of H+ ions (mol/L) (pH is derived: -log10[H+])
- state_array[7]: Volume (L) (can be constant or variable)
- state_array[8...]: Other species or properties

The `env_instance` is expected to hold parameters for the specific
chemical system being modeled (e.g., rate constants, activation energies,
enthalpies, stoichiometry, solubility limits).
"""

import numpy as np

# Define indices for clarity, assuming a reaction A + B <=> C + D
# These would ideally be dynamically set or passed based on the reaction system
CONC_A_IDX = 0
CONC_B_IDX = 1
CONC_C_IDX = 2
CONC_D_IDX = 3
CONC_B_IDX = 4
CONC_C_IDX = 5
CATALYST_IDX = 6
TEMP_IDX = 7
H_PLUS_IDX = 8
VOLUME_IDX = 9
NUM_CORE_SPECIES = 6


def apply_chemical_constants(env_instance):
    """
    Registers universal chemical and physical constants in the environment instance.
    """
    constants = {
        # Fundamental constants
        "N_A": 6.02214076e23,                 # Avogadro constant (mol^−1)
        "R": 8.31446261815324,                # Ideal gas constant (J/mol·K or L·kPa/mol·K)
        "k_B": 1.380649e-23,                  # Boltzmann constant (J/K)
        "F": 96485.33212,                     # Faraday constant (C/mol)
        "h": 6.62607015e-34,                  # Planck constant (J·s)
        "c": 299792458,                       # Speed of light in vacuum (m/s)

        # Standard conditions
        "T_STD": 298.15,                      # Standard temperature (K, 25 °C)
        "P_STD": 101.325,                     # Standard pressure (kPa, 1 atm)
        "P_STD_PA": 100000,                   # Standard pressure (Pa, 1 bar often used in thermo)


        # Water properties
        "Kw_25C": 1.0e-14,                    # Ion product of water at 25°C (mol^2/L^2)
        "DENSITY_WATER_25C": 997.0,           # Density of water at 25°C (kg/m^3 or g/L)
        "CP_WATER_LIQUID": 4184,              # Specific heat capacity of liquid water (J/kg·K)

        # Conversion factors
        "ATM_TO_PA": 101325,                  # Atmospheres to Pascals
        "BAR_TO_PA": 100000,                  # Bar to Pascals
        "L_TO_M3": 0.001,                     # Liters to cubic meters
    }
    env_instance.constants = constants
    return constants

def _calculate_rate_constant(temp, A, Ea, R_const):
    """Helper to calculate rate constant using Arrhenius equation: k = A * exp(-Ea / (R * T))"""
    if temp <= 0: return 0.0 # No reaction at or below absolute zero
    return A * np.exp(-Ea / (R_const * temp))

def apply_chemical_processes(env_instance, state_array):
    """
    Applies chemical reaction kinetics and related effects.
    Assumes `env_instance` contains reaction-specific parameters.

    Expected `env_instance` attributes:
    - dt: time step
    - reaction_stoichiometry: dict, e.g., {'A': -1, 'B': -1, 'C': 1, 'D': 1, 'H+': 0}
                             (negative for reactants, positive for products)
    - k_fwd_params: {'A0': pre_exp_factor, 'Ea': activation_energy_J_mol} for forward reaction
    - k_rev_params: {'A0': pre_exp_factor, 'Ea': activation_energy_J_mol} for reverse reaction
    - catalyst_params: {'Ea_reduction_factor_fwd': 0.8, 'Ea_reduction_factor_rev': 0.8}
                       (e.g., 0.8 means Ea is 80% of uncatalyzed)
    - delta_H_reaction: Enthalpy change of the forward reaction (J/mol of reaction extent)
                       (Negative for exothermic, positive for endothermic)
    - system_properties: {'density_solution_kg_L': 1.0, 'cp_solution_J_kg_K': 4184}
                         (for calculating temperature changes)
    - species_map: {'A': CONC_A_IDX, 'B': CONC_B_IDX, ... , 'H+': H_PLUS_IDX}
    """
    dt = env_instance.dt
    R_const = env_instance.constants["R"]
    current_temp = state_array[TEMP_IDX]
    volume_L = state_array[VOLUME_IDX]
    catalyst_conc = state_array[CATALYST_IDX] if CATALYST_IDX < len(state_array) else 0

    # --- 1. Calculate current rate constants based on temperature and catalyst ---
    Ea_fwd = env_instance.k_fwd_params['Ea']
    Ea_rev = env_instance.k_rev_params['Ea']

    if catalyst_conc > 1e-9: # If catalyst is present
        Ea_fwd *= env_instance.catalyst_params.get('Ea_reduction_factor_fwd', 1.0)
        Ea_rev *= env_instance.catalyst_params.get('Ea_reduction_factor_rev', 1.0)

    k_fwd = _calculate_rate_constant(current_temp, env_instance.k_fwd_params['A0'], Ea_fwd, R_const)
    k_rev = _calculate_rate_constant(current_temp, env_instance.k_rev_params['A0'], Ea_rev, R_const)

    # --- 2. Calculate reaction rates ---
    # Assuming a simple elementary reaction A + B <=> C + D for rate law
    # More complex rate laws would need specific handling.
    conc_A = state_array[env_instance.species_map['A']]
    conc_B = state_array[env_instance.species_map['B']]
    conc_C = state_array[env_instance.species_map['C']]
    conc_D = state_array[env_instance.species_map.get('D', -1)] # Handle if D is not present

    # Ensure concentrations are non-negative for rate calculation
    conc_A = max(0, conc_A)
    conc_B = max(0, conc_B)
    conc_C = max(0, conc_C)
    if 'D' in env_instance.species_map:
         conc_D = max(0, state_array[env_instance.species_map['D']])


    # Rate expressions (mol/L/s)
    # This example assumes A and B are reactants for forward, C and D for reverse.
    # Adjust based on actual reaction order.
    rate_fwd = k_fwd * conc_A * conc_B # 2nd order forward
    rate_rev = k_rev * conc_C * (conc_D if 'D' in env_instance.species_map else 1.0) # Adjust if D not present or different order

    # Net rate of reaction extent (positive for forward, negative for reverse)
    # This 'xi_dot' represents the rate of change of the extent of reaction per unit volume (mol/L/s)
    rate_of_reaction_extent_vol = rate_fwd - rate_rev

    # --- 3. Update concentrations based on stoichiometry and rate ---
    # d(conc_i)/dt = nu_i * rate_of_reaction_extent_vol
    # delta_conc_i = nu_i * rate_of_reaction_extent_vol * dt
    for species_name, stoich_coeff in env_instance.reaction_stoichiometry.items():
        if species_name in env_instance.species_map:
            idx = env_instance.species_map[species_name]
            delta_conc = stoich_coeff * rate_of_reaction_extent_vol * dt
            state_array[idx] += delta_conc
        # else: # Could handle species not in state_array if needed (e.g. solvent)
            # pass

    # --- 4. Update temperature based on heat of reaction ---
    if abs(env_instance.delta_H_reaction) > 1e-9 and volume_L > 1e-9:
        # Moles of reaction extent that occurred in this dt step
        moles_extent_reacted = rate_of_reaction_extent_vol * volume_L * dt

        heat_generated_J = moles_extent_reacted * (-env_instance.delta_H_reaction) # -ve dH for exothermic

        mass_solution_kg = volume_L * env_instance.system_properties['density_solution_kg_L']
        cp_solution_J_kg_K = env_instance.system_properties['cp_solution_J_kg_K']

        if mass_solution_kg > 1e-9 and cp_solution_J_kg_K > 1e-9:
            delta_temp = heat_generated_J / (mass_solution_kg * cp_solution_J_kg_K)
            state_array[TEMP_IDX] += delta_temp

    # --- 5. Update H+ concentration if affected by reaction (implicitly handled by stoichiometry)
    # The change in [H+] is already covered if 'H+' is in reaction_stoichiometry
    # and species_map. We will enforce pH bounds later.

    # --- 6. Other effects (e.g., diffusion, mixing - more advanced) ---
    # For now, assume a well-mixed reactor.

    return state_array

def enforce_chemical_constraints(env_instance, state_array):
    """
    Enforces chemical constraints such as non-negativity, solubility, pH limits.

    Expected `env_instance` attributes:
    - solubility_limits: dict, e.g., {'C': 0.5 (mol/L), 'D': 0.1 (mol/L)}
    - species_map: As in apply_chemical_processes
    - initial_total_moles: dict, e.g. {'element_X': 1.0, 'element_Y': 2.0} for mass balance check (optional)
    """
    Kw = env_instance.constants.get("Kw_25C", 1e-14) # Get Kw, default to 25C value

    # --- 1. Non-negativity of concentrations ---
    # Apply to all species concentrations and catalyst
    # Assumes species concentrations are the first few elements up to CATALYST_IDX,
    # and H_PLUS_IDX. This needs to be more robust if state_array structure varies.
    for i in range(CATALYST_IDX + 1): # Includes A, B, C, D, Catalyst
         if i < len(state_array):
            state_array[i] = max(0.0, state_array[i])
    if H_PLUS_IDX < len(state_array):
        state_array[H_PLUS_IDX] = max(1e-15, state_array[H_PLUS_IDX]) # Avoid log(0) or negative, keep tiny amount

    # --- 2. Temperature limits ---
    if TEMP_IDX < len(state_array):
        state_array[TEMP_IDX] = max(0.0, state_array[TEMP_IDX]) # Absolute zero
        # Could also add an upper practical limit if needed.

    # --- 3. pH related constraints ([H+]) ---
    # pH is derived, so we constrain [H+].
    # Typically, pH 0-14 means [H+] between 1.0 M and 1e-14 M.
    if H_PLUS_IDX < len(state_array):
        state_array[H_PLUS_IDX] = np.clip(state_array[H_PLUS_IDX], 1e-14, 1.0) # Practical limits
        # Note: True pH can go outside 0-14 in very concentrated acids/bases.

    # --- 4. Solubility limits (Precipitation) ---
    # If a species concentration exceeds its solubility, it precipitates.
    # This is a simplified model; actual precipitation might affect other equilibria.
    if hasattr(env_instance, 'solubility_limits') and hasattr(env_instance, 'species_map'):
        for species_name, limit_mol_L in env_instance.solubility_limits.items():
            if species_name in env_instance.species_map:
                idx = env_instance.species_map[species_name]
                if state_array[idx] > limit_mol_L:
                    # Excess amount could be tracked as 'precipitated_species_name'
                    # For now, just cap the concentration in solution.
                    # precipitated_amount = (state_array[idx] - limit_mol_L) * state_array[VOLUME_IDX]
                    state_array[idx] = limit_mol_L

    # --- 5. Volume constraints ---
    if VOLUME_IDX < len(state_array):
        state_array[VOLUME_IDX] = max(1e-9, state_array[VOLUME_IDX]) # Volume cannot be zero or negative

    # --- 6. Conservation of Mass / Atom Balance (Optional Check) ---
    # This is more complex to enforce directly without fighting kinetics.
    # Stoichiometrically correct kinetics *should* conserve mass.
    # This section could be used for verification or applying small corrections
    # if numerical drift occurs over long simulations.
    # Example: if 'initial_total_moles_A_element' is known:
    # current_moles_A_element = state_array[CONC_A_IDX]*Vol + state_array[CONC_C_IDX]*Vol (if C contains one A)
    # if abs(current_moles_A_element - initial_total_moles_A_element) > tolerance:
    #     # Apply a correction factor (can be tricky)
    #     pass

    # --- 7. Equilibrium Check (Informational, not an enforced constraint here) ---
    # The system naturally moves towards equilibrium via kinetics.
    # Q = ([C]^c * [D]^d) / ([A]^a * [B]^b)
    # K_eq = k_fwd / k_rev (for elementary steps, or from thermodynamics delta_G0 = -RTlnK)
    # If Q approx K_eq, system is near equilibrium.

    return state_array

# --- END OF FILE chemistry_constraints.py ---
