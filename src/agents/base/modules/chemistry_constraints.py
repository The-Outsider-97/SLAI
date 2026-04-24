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

from __future__ import annotations

import math
import numpy as np

from dataclasses import dataclass, field
from collections import deque
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.base_errors import *
from ..utils.base_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Chemistry Constraints")
printer = PrettyPrinter()

CONC_A_IDX = 0
CONC_B_IDX = 1
CONC_C_IDX = 2
CONC_D_IDX = 3
CATALYST_IDX = 4
TEMP_IDX = 5
H_PLUS_IDX = 6
VOLUME_IDX = 7
NUM_CORE_SPECIES = 8


@dataclass
class ChemistryConfig:
    """Configuration parameters for the chemistry engine."""

    dt: float = 0.02
    activity_floor: float = 1e-12
    numeric_tolerance: float = 1e-12
    enable_history: bool = True
    history_limit: int = 200

    species_map: Dict[str, int] = field(
        default_factory=lambda: {
            "A": CONC_A_IDX,
            "B": CONC_B_IDX,
            "C": CONC_C_IDX,
            "D": CONC_D_IDX,
            "Cat": CATALYST_IDX,
            "temperature": TEMP_IDX,
            "H+": H_PLUS_IDX,
            "volume": VOLUME_IDX,
        }
    )
    reaction_stoichiometry: Dict[str, float] = field(
        default_factory=lambda: {"A": -1.0, "B": -1.0, "C": 1.0, "D": 1.0}
    )
    reaction_orders_fwd: Dict[str, float] = field(default_factory=dict)
    reaction_orders_rev: Dict[str, float] = field(default_factory=dict)

    k_fwd_params: Dict[str, float] = field(default_factory=lambda: {"A0": 1.0, "Ea": 40000.0})
    k_rev_params: Dict[str, float] = field(default_factory=lambda: {"A0": 0.1, "Ea": 42000.0})

    enable_catalysis: bool = True
    catalyst_species: str = "Cat"
    catalyst_min_active_conc: float = 1e-9
    catalyst_concentration_factor: float = 0.25
    catalyst_params: Dict[str, float] = field(
        default_factory=lambda: {
            "Ea_reduction_factor_fwd": 0.8,
            "Ea_reduction_factor_rev": 0.8,
        }
    )

    enable_ph_modulation: bool = True
    ph_species: str = "H+"
    ph_optimum: float = 7.0
    ph_sensitivity: float = 0.35
    ph_min: float = 0.0
    ph_max: float = 14.0

    delta_H_reaction: float = 0.0
    system_properties: Dict[str, float] = field(
        default_factory=lambda: {
            "density_solution_kg_L": 1.0,
            "cp_solution_J_kg_K": 4184.0,
        }
    )
    min_temperature_K: float = 0.0
    max_temperature_K: float = 5000.0

    min_volume_L: float = 1e-9
    max_volume_L: float = 1e9

    solubility_limits: Dict[str, float] = field(default_factory=dict)
    clip_negative_concentrations: bool = True
    max_concentration_mol_L: float = 1e6

    enforce_mass_balance: bool = False
    auto_correct_mass_balance: bool = False
    element_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    initial_total_moles: Dict[str, float] = field(default_factory=dict)
    mass_balance_tolerance: float = 1e-6


@dataclass(frozen=True)
class ChemistryStepSummary:
    """Audit-friendly summary of one chemistry update."""

    timestamp: str
    dt: float
    forward_rate: float
    reverse_rate: float
    net_rate: float
    temperature_K: float
    pH: Optional[float]
    equilibrium_constant: Optional[float]
    reaction_quotient: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "dt": self.dt,
            "forward_rate": self.forward_rate,
            "reverse_rate": self.reverse_rate,
            "net_rate": self.net_rate,
            "temperature_K": self.temperature_K,
            "pH": self.pH,
            "equilibrium_constant": self.equilibrium_constant,
            "reaction_quotient": self.reaction_quotient,
        }


class ChemistryEngine:
    """
    Centralized chemistry engine for the SLAI Environment.

    Applies reaction kinetics, catalysis, pH modulation, thermal effects,
    equilibrium diagnostics, and constraint enforcement against a chemical
    state vector.
    """

    CONSTANTS: Dict[str, float] = {
        "N_A": 6.02214076e23,
        "R": 8.31446261815324,
        "k_B": 1.380649e-23,
        "F": 96485.33212,
        "h": 6.62607015e-34,
        "c": 299792458.0,
        "T_STD": 298.15,
        "P_STD": 101.325,
        "P_STD_PA": 100000.0,
        "Kw_25C": 1.0e-14,
        "DENSITY_WATER_25C": 997.0,
        "CP_WATER_LIQUID": 4184.0,
        "ATM_TO_PA": 101325.0,
        "BAR_TO_PA": 100000.0,
        "L_TO_M3": 0.001,
    }

    def __init__(self, config: Optional[Union[ChemistryConfig, Mapping[str, Any]]] = None) -> None:
        self.global_config = load_global_config()
        self.chemistry_config = get_config_section("chemistry_constraints") or {}
        self.constants = dict(self.CONSTANTS)
        self.config = self._load_config(config)
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.config.history_limit)
        logger.info("Chemistry Constraints successfully initialized")

    def _load_config(self, override: Optional[Union[ChemistryConfig, Mapping[str, Any]]]) -> ChemistryConfig:
        if isinstance(override, ChemistryConfig):
            config = override
        else:
            merged = deep_merge_dicts(self.chemistry_config, dict(override or {}))
            config = ChemistryConfig(
                dt=coerce_float(merged.get("dt", 0.02), 0.02, minimum=1e-9),
                activity_floor=coerce_float(merged.get("activity_floor", 1e-12), 1e-12, minimum=1e-18),
                numeric_tolerance=coerce_float(merged.get("numeric_tolerance", 1e-12), 1e-12, minimum=0.0),
                enable_history=coerce_bool(merged.get("enable_history", True), True),
                history_limit=coerce_int(merged.get("history_limit", 200), 200, minimum=1),
                species_map=dict(parse_json_if_needed(merged.get("species_map"), default=None) or merged.get("species_map") or ChemistryConfig().species_map),
                reaction_stoichiometry=dict(parse_json_if_needed(merged.get("reaction_stoichiometry"), default=None) or merged.get("reaction_stoichiometry") or ChemistryConfig().reaction_stoichiometry),
                reaction_orders_fwd=dict(parse_json_if_needed(merged.get("reaction_orders_fwd"), default=None) or merged.get("reaction_orders_fwd") or {}),
                reaction_orders_rev=dict(parse_json_if_needed(merged.get("reaction_orders_rev"), default=None) or merged.get("reaction_orders_rev") or {}),
                k_fwd_params=dict(parse_json_if_needed(merged.get("k_fwd_params"), default=None) or merged.get("k_fwd_params") or ChemistryConfig().k_fwd_params),
                k_rev_params=dict(parse_json_if_needed(merged.get("k_rev_params"), default=None) or merged.get("k_rev_params") or ChemistryConfig().k_rev_params),
                enable_catalysis=coerce_bool(merged.get("enable_catalysis", True), True),
                catalyst_species=ensure_non_empty_string(merged.get("catalyst_species", "Cat"), "catalyst_species", config=merged, error_cls=BaseConfigurationError),
                catalyst_min_active_conc=coerce_float(merged.get("catalyst_min_active_conc", 1e-9), 1e-9, minimum=0.0),
                catalyst_concentration_factor=coerce_float(merged.get("catalyst_concentration_factor", 0.25), 0.25, minimum=0.0),
                catalyst_params=dict(parse_json_if_needed(merged.get("catalyst_params"), default=None) or merged.get("catalyst_params") or ChemistryConfig().catalyst_params),
                enable_ph_modulation=coerce_bool(merged.get("enable_ph_modulation", True), True),
                ph_species=ensure_non_empty_string(merged.get("ph_species", "H+"), "ph_species", config=merged, error_cls=BaseConfigurationError),
                ph_optimum=coerce_float(merged.get("ph_optimum", 7.0), 7.0),
                ph_sensitivity=coerce_float(merged.get("ph_sensitivity", 0.35), 0.35, minimum=0.0),
                ph_min=coerce_float(merged.get("ph_min", 0.0), 0.0),
                ph_max=coerce_float(merged.get("ph_max", 14.0), 14.0),
                delta_H_reaction=coerce_float(merged.get("delta_H_reaction", 0.0), 0.0),
                system_properties=dict(parse_json_if_needed(merged.get("system_properties"), default=None) or merged.get("system_properties") or ChemistryConfig().system_properties),
                min_temperature_K=coerce_float(merged.get("min_temperature_K", 0.0), 0.0, minimum=0.0),
                max_temperature_K=coerce_float(merged.get("max_temperature_K", 5000.0), 5000.0, minimum=0.0),
                min_volume_L=coerce_float(merged.get("min_volume_L", 1e-9), 1e-9, minimum=1e-18),
                max_volume_L=coerce_float(merged.get("max_volume_L", 1e9), 1e9, minimum=1e-9),
                solubility_limits=dict(parse_json_if_needed(merged.get("solubility_limits"), default=None) or merged.get("solubility_limits") or {}),
                clip_negative_concentrations=coerce_bool(merged.get("clip_negative_concentrations", True), True),
                max_concentration_mol_L=coerce_float(merged.get("max_concentration_mol_L", 1e6), 1e6, minimum=1e-12),
                enforce_mass_balance=coerce_bool(merged.get("enforce_mass_balance", False), False),
                auto_correct_mass_balance=coerce_bool(merged.get("auto_correct_mass_balance", False), False),
                element_matrix=dict(parse_json_if_needed(merged.get("element_matrix"), default=None) or merged.get("element_matrix") or {}),
                initial_total_moles=dict(parse_json_if_needed(merged.get("initial_total_moles"), default=None) or merged.get("initial_total_moles") or {}),
                mass_balance_tolerance=coerce_float(merged.get("mass_balance_tolerance", 1e-6), 1e-6, minimum=0.0),
            )

        self._validate_config(config)
        return config

    def _validate_config(self, config: ChemistryConfig) -> None:
        ensure_mapping(config.species_map, "species_map", config=self.chemistry_config, error_cls=BaseConfigurationError)
        ensure_mapping(config.reaction_stoichiometry, "reaction_stoichiometry", config=self.chemistry_config, error_cls=BaseConfigurationError)
        ensure_mapping(config.k_fwd_params, "k_fwd_params", config=self.chemistry_config, error_cls=BaseConfigurationError)
        ensure_mapping(config.k_rev_params, "k_rev_params", config=self.chemistry_config, error_cls=BaseConfigurationError)
        ensure_keys(config.k_fwd_params, ["A0", "Ea"], "k_fwd_params", config=self.chemistry_config, error_cls=BaseConfigurationError)
        ensure_keys(config.k_rev_params, ["A0", "Ea"], "k_rev_params", config=self.chemistry_config, error_cls=BaseConfigurationError)
        ensure_condition(config.ph_max >= config.ph_min, "'ph_max' must be >= 'ph_min'.", config=self.chemistry_config, error_cls=BaseConfigurationError)
        ensure_condition(config.max_temperature_K >= config.min_temperature_K, "'max_temperature_K' must be >= 'min_temperature_K'.", config=self.chemistry_config, error_cls=BaseConfigurationError)
        ensure_condition(config.max_volume_L >= config.min_volume_L, "'max_volume_L' must be >= 'min_volume_L'.", config=self.chemistry_config, error_cls=BaseConfigurationError)

    def _record_history(self, summary: ChemistryStepSummary) -> None:
        if self.config.enable_history:
            self._history.append(summary.to_dict())

    def recent_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(self._history)[-coerce_int(limit, 20, minimum=1):]

    def _require_index(self, name: str) -> int:
        if name not in self.config.species_map:
            raise BaseConfigurationError(
                f"Species '{name}' is not present in species_map.",
                self.chemistry_config,
                component="ChemistryEngine",
                operation="species_lookup",
                context={"species": name, "species_map": self.config.species_map},
            )
        index = coerce_int(self.config.species_map[name], -1)
        if index < 0:
            raise BaseConfigurationError(
                f"Species '{name}' has an invalid negative index.",
                self.chemistry_config,
                component="ChemistryEngine",
                operation="species_lookup",
                context={"species": name, "index": index},
            )
        return index

    def _get_index(self, name: str) -> Optional[int]:
        if name not in self.config.species_map:
            return None
        index = coerce_int(self.config.species_map[name], -1)
        return index if index >= 0 else None

    def _state_view(self, state_array: np.ndarray) -> np.ndarray:
        if not isinstance(state_array, np.ndarray):
            raise BaseValidationError(
                "'state_array' must be a numpy.ndarray.",
                self.chemistry_config,
                component="ChemistryEngine",
                operation="state_validation",
                context={"received_type": type(state_array).__name__},
            )
        if state_array.ndim != 1:
            raise BaseValidationError(
                "'state_array' must be one-dimensional.",
                self.chemistry_config,
                component="ChemistryEngine",
                operation="state_validation",
                context={"shape": tuple(state_array.shape)},
            )
        return state_array.astype(float, copy=False)

    def apply_constants(self, env_instance: Any) -> Dict[str, float]:
        env_instance.constants = dict(self.constants)
        return env_instance.constants

    def calculate_rate_constant(self, temperature_K: float, params: Mapping[str, Any], catalyst_multiplier: float = 1.0) -> float:
        ensure_mapping(params, "rate_params", config=self.chemistry_config, error_cls=BaseValidationError)
        pre_exponential = coerce_float(params.get("A0", 0.0), 0.0, minimum=0.0)
        activation_energy = coerce_float(params.get("Ea", 0.0), 0.0)
        if temperature_K <= 0.0:
            return 0.0
        adjusted_Ea = activation_energy * max(catalyst_multiplier, 0.0)
        try:
            return pre_exponential * math.exp(-adjusted_Ea / (self.constants["R"] * temperature_K))
        except OverflowError:
            return 0.0

    def calculate_ph(self, h_plus_concentration: float) -> Optional[float]:
        if h_plus_concentration <= 0.0:
            return None
        return -math.log10(max(h_plus_concentration, self.config.activity_floor))

    def _rate_orders(self, reverse: bool = False) -> Dict[str, float]:
        configured = self.config.reaction_orders_rev if reverse else self.config.reaction_orders_fwd
        if configured:
            return {name: max(0.0, float(order)) for name, order in configured.items()}
        orders: Dict[str, float] = {}
        for species, coeff in self.config.reaction_stoichiometry.items():
            if reverse and coeff > 0:
                orders[species] = abs(float(coeff))
            elif not reverse and coeff < 0:
                orders[species] = abs(float(coeff))
        return orders

    def _reaction_rate(self, state: np.ndarray, rate_constant: float, orders: Mapping[str, float]) -> float:
        rate = rate_constant
        for species, order in orders.items():
            idx = self._get_index(species)
            if idx is None or idx >= len(state):
                raise BaseConfigurationError(
                    f"Species '{species}' is missing from the state vector required for the rate law.",
                    self.chemistry_config,
                    component="ChemistryEngine",
                    operation="reaction_rate",
                    context={"species": species, "index": idx, "state_length": len(state)},
                )
            concentration = max(self.config.activity_floor, float(state[idx]))
            rate *= concentration ** float(order)
        return max(0.0, float(rate))

    def calculate_equilibrium_constant(self, temperature_K: float) -> Optional[float]:
        if temperature_K <= 0.0:
            return None
        k_fwd = self.calculate_rate_constant(temperature_K, self.config.k_fwd_params)
        k_rev = self.calculate_rate_constant(temperature_K, self.config.k_rev_params)
        if k_rev <= self.config.numeric_tolerance:
            return None
        return k_fwd / k_rev

    def calculate_reaction_quotient(self, state: np.ndarray) -> Optional[float]:
        numerator = 1.0
        denominator = 1.0
        for species, coeff in self.config.reaction_stoichiometry.items():
            idx = self._get_index(species)
            if idx is None or idx >= len(state):
                continue
            activity = max(self.config.activity_floor, float(state[idx]))
            if coeff > 0:
                numerator *= activity ** coeff
            elif coeff < 0:
                denominator *= activity ** abs(coeff)
        if denominator <= self.config.numeric_tolerance:
            return None
        return numerator / denominator

    def _ph_multiplier(self, state: np.ndarray) -> float:
        if not self.config.enable_ph_modulation:
            return 1.0
        idx = self._get_index(self.config.ph_species)
        if idx is None or idx >= len(state):
            return 1.0
        ph = self.calculate_ph(float(state[idx]))
        if ph is None:
            return 1.0
        distance = abs(ph - self.config.ph_optimum)
        return max(self.config.activity_floor, math.exp(-self.config.ph_sensitivity * distance))

    def _catalyst_multiplier(self, state: np.ndarray) -> Tuple[float, float]:
        if not self.config.enable_catalysis:
            return 1.0, 0.0
        idx = self._get_index(self.config.catalyst_species)
        if idx is None or idx >= len(state):
            return 1.0, 0.0
        catalyst_conc = max(0.0, float(state[idx]))
        if catalyst_conc < self.config.catalyst_min_active_conc:
            return 1.0, catalyst_conc
        enhancement = 1.0 + (catalyst_conc * self.config.catalyst_concentration_factor)
        return enhancement, catalyst_conc

    def apply_chemical_processes(self, state_array: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        state = self._state_view(state_array)
        dt_value = coerce_float(dt if dt is not None else self.config.dt, self.config.dt, minimum=0.0)
        if dt_value <= 0.0:
            return state

        temp_idx = self._require_index("temperature")
        volume_idx = self._require_index("volume")
        if temp_idx >= len(state) or volume_idx >= len(state):
            raise BaseValidationError(
                "State vector is shorter than required temperature/volume indices.",
                self.chemistry_config,
                component="ChemistryEngine",
                operation="apply_chemical_processes",
                context={"state_length": len(state), "temp_idx": temp_idx, "volume_idx": volume_idx},
            )

        temperature_K = max(self.config.min_temperature_K, float(state[temp_idx]))
        volume_L = max(self.config.min_volume_L, float(state[volume_idx]))
        catalyst_multiplier, catalyst_conc = self._catalyst_multiplier(state)

        fwd_ea_factor = coerce_float(self.config.catalyst_params.get("Ea_reduction_factor_fwd", 1.0), 1.0, minimum=0.0)
        rev_ea_factor = coerce_float(self.config.catalyst_params.get("Ea_reduction_factor_rev", 1.0), 1.0, minimum=0.0)
        fwd_rate_constant = self.calculate_rate_constant(
            temperature_K,
            self.config.k_fwd_params,
            catalyst_multiplier=fwd_ea_factor if catalyst_conc >= self.config.catalyst_min_active_conc else 1.0,
        ) * catalyst_multiplier
        rev_rate_constant = self.calculate_rate_constant(
            temperature_K,
            self.config.k_rev_params,
            catalyst_multiplier=rev_ea_factor if catalyst_conc >= self.config.catalyst_min_active_conc else 1.0,
        ) * catalyst_multiplier

        ph_multiplier = self._ph_multiplier(state)
        forward_rate = self._reaction_rate(state, fwd_rate_constant * ph_multiplier, self._rate_orders(reverse=False))
        reverse_rate = self._reaction_rate(state, rev_rate_constant * ph_multiplier, self._rate_orders(reverse=True))
        net_rate = forward_rate - reverse_rate

        for species_name, stoich_coeff in self.config.reaction_stoichiometry.items():
            idx = self._get_index(species_name)
            if idx is None or idx >= len(state):
                continue
            state[idx] += float(stoich_coeff) * net_rate * dt_value

        if abs(self.config.delta_H_reaction) > self.config.numeric_tolerance and volume_L > self.config.numeric_tolerance:
            extent_moles = net_rate * volume_L * dt_value
            heat_J = extent_moles * (-self.config.delta_H_reaction)
            density = coerce_float(self.config.system_properties.get("density_solution_kg_L", 1.0), 1.0, minimum=self.config.activity_floor)
            cp = coerce_float(self.config.system_properties.get("cp_solution_J_kg_K", 4184.0), 4184.0, minimum=self.config.activity_floor)
            solution_mass = max(self.config.activity_floor, density * volume_L)
            state[temp_idx] += heat_J / (solution_mass * cp)

        ph_idx = self._get_index(self.config.ph_species)
        ph_value = None
        if ph_idx is not None and ph_idx < len(state):
            ph_value = self.calculate_ph(float(state[ph_idx]))

        summary = ChemistryStepSummary(
            timestamp=utc_now_iso(),
            dt=dt_value,
            forward_rate=forward_rate,
            reverse_rate=reverse_rate,
            net_rate=net_rate,
            temperature_K=float(state[temp_idx]),
            pH=ph_value,
            equilibrium_constant=self.calculate_equilibrium_constant(float(state[temp_idx])),
            reaction_quotient=self.calculate_reaction_quotient(state),
        )
        self._record_history(summary)
        return state

    def _apply_mass_balance(self, state: np.ndarray) -> np.ndarray:
        if not self.config.enforce_mass_balance or not self.config.element_matrix or not self.config.initial_total_moles:
            return state

        volume_idx = self._get_index("volume")
        volume_L = float(state[volume_idx]) if volume_idx is not None and volume_idx < len(state) else 1.0
        volume_L = max(volume_L, self.config.min_volume_L)

        for element_name, species_weights in self.config.element_matrix.items():
            target_total = self.config.initial_total_moles.get(element_name)
            if target_total is None:
                continue
            current_total = 0.0
            participating_indices: List[Tuple[int, float]] = []
            for species_name, coefficient in species_weights.items():
                idx = self._get_index(species_name)
                if idx is None or idx >= len(state):
                    continue
                participating_indices.append((idx, float(coefficient)))
                current_total += max(0.0, float(state[idx])) * volume_L * float(coefficient)

            drift = current_total - float(target_total)
            if abs(drift) <= self.config.mass_balance_tolerance:
                continue
            if not self.config.auto_correct_mass_balance:
                raise BaseStateError(
                    f"Mass balance drift exceeded tolerance for element '{element_name}'.",
                    self.chemistry_config,
                    component="ChemistryEngine",
                    operation="mass_balance",
                    context={
                        "element": element_name,
                        "target_total": target_total,
                        "current_total": current_total,
                        "drift": drift,
                        "tolerance": self.config.mass_balance_tolerance,
                    },
                )

            participating_moles = sum(max(0.0, float(state[idx])) * volume_L * max(weight, 0.0) for idx, weight in participating_indices)
            if participating_moles <= self.config.numeric_tolerance:
                continue
            scale = max(0.0, float(target_total) / participating_moles)
            for idx, _ in participating_indices:
                state[idx] = max(0.0, float(state[idx]) * scale)

        return state

    def enforce_chemical_constraints(self, state_array: np.ndarray) -> np.ndarray:
        state = self._state_view(state_array)

        if self.config.clip_negative_concentrations:
            protected_indices = {
                idx for idx in [self._get_index("temperature"), self._get_index("volume")]
                if idx is not None
            }
            for index in range(len(state)):
                if index in protected_indices:
                    continue
                state[index] = max(0.0, float(state[index]))

        temp_idx = self._get_index("temperature")
        if temp_idx is not None and temp_idx < len(state):
            state[temp_idx] = float(np.clip(state[temp_idx], self.config.min_temperature_K, self.config.max_temperature_K))

        volume_idx = self._get_index("volume")
        if volume_idx is not None and volume_idx < len(state):
            state[volume_idx] = float(np.clip(state[volume_idx], self.config.min_volume_L, self.config.max_volume_L))

        h_idx = self._get_index(self.config.ph_species)
        if h_idx is not None and h_idx < len(state):
            h_min = 10 ** (-self.config.ph_max)
            h_max = 10 ** (-self.config.ph_min)
            state[h_idx] = float(np.clip(state[h_idx], h_min, h_max))

        for species_name, limit_mol_L in self.config.solubility_limits.items():
            idx = self._get_index(species_name)
            if idx is None or idx >= len(state):
                continue
            state[idx] = min(float(state[idx]), float(limit_mol_L))

        for species_name, idx in self.config.species_map.items():
            if species_name in {"temperature", "volume"} or idx >= len(state):
                continue
            state[idx] = float(np.clip(state[idx], 0.0, self.config.max_concentration_mol_L))

        self._apply_mass_balance(state)
        return state

    def apply_all(self, state_array: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        state = self.apply_chemical_processes(state_array, dt=dt)
        return self.enforce_chemical_constraints(state)

    def diagnostics(self, state_array: np.ndarray) -> Dict[str, Any]:
        state = self._state_view(state_array)
        temp_idx = self._get_index("temperature")
        h_idx = self._get_index(self.config.ph_species)
        diagnostics = {
            "temperature_K": float(state[temp_idx]) if temp_idx is not None and temp_idx < len(state) else None,
            "pH": self.calculate_ph(float(state[h_idx])) if h_idx is not None and h_idx < len(state) else None,
            "equilibrium_constant": self.calculate_equilibrium_constant(float(state[temp_idx])) if temp_idx is not None and temp_idx < len(state) else None,
            "reaction_quotient": self.calculate_reaction_quotient(state),
            "history_length": len(self._history),
        }
        return drop_none_values(to_json_safe(diagnostics), recursive=True, drop_empty=False)


_engine: Optional[ChemistryEngine] = None


def _get_engine() -> ChemistryEngine:
    global _engine
    if _engine is None:
        _engine = ChemistryEngine()
    return _engine


def apply_chemical_constants(env_instance: Any) -> Dict[str, float]:
    return _get_engine().apply_constants(env_instance)


def apply_chemical_processes(env_instance: Any, state_array: np.ndarray) -> np.ndarray:
    engine = _get_engine()
    if hasattr(env_instance, "dt"):
        engine.config.dt = coerce_float(getattr(env_instance, "dt"), engine.config.dt, minimum=0.0)
    if hasattr(env_instance, "reaction_stoichiometry") and isinstance(env_instance.reaction_stoichiometry, Mapping):
        engine.config.reaction_stoichiometry = dict(env_instance.reaction_stoichiometry)
    if hasattr(env_instance, "k_fwd_params") and isinstance(env_instance.k_fwd_params, Mapping):
        engine.config.k_fwd_params = dict(env_instance.k_fwd_params)
    if hasattr(env_instance, "k_rev_params") and isinstance(env_instance.k_rev_params, Mapping):
        engine.config.k_rev_params = dict(env_instance.k_rev_params)
    if hasattr(env_instance, "catalyst_params") and isinstance(env_instance.catalyst_params, Mapping):
        engine.config.catalyst_params = dict(env_instance.catalyst_params)
    if hasattr(env_instance, "system_properties") and isinstance(env_instance.system_properties, Mapping):
        engine.config.system_properties = dict(env_instance.system_properties)
    if hasattr(env_instance, "delta_H_reaction"):
        engine.config.delta_H_reaction = coerce_float(getattr(env_instance, "delta_H_reaction"), engine.config.delta_H_reaction)
    if hasattr(env_instance, "species_map") and isinstance(env_instance.species_map, Mapping):
        engine.config.species_map = dict(env_instance.species_map)
    if hasattr(env_instance, "solubility_limits") and isinstance(env_instance.solubility_limits, Mapping):
        engine.config.solubility_limits = dict(env_instance.solubility_limits)
    if hasattr(env_instance, "initial_total_moles") and isinstance(env_instance.initial_total_moles, Mapping):
        engine.config.initial_total_moles = dict(env_instance.initial_total_moles)
    if hasattr(env_instance, "element_matrix") and isinstance(env_instance.element_matrix, Mapping):
        engine.config.element_matrix = dict(env_instance.element_matrix)
    if not hasattr(env_instance, "constants") or not isinstance(getattr(env_instance, "constants"), Mapping):
        engine.apply_constants(env_instance)
    return engine.apply_chemical_processes(state_array, dt=engine.config.dt)


def enforce_chemical_constraints(env_instance: Any, state_array: np.ndarray) -> np.ndarray:
    engine = _get_engine()
    if hasattr(env_instance, "solubility_limits") and isinstance(env_instance.solubility_limits, Mapping):
        engine.config.solubility_limits = dict(env_instance.solubility_limits)
    if hasattr(env_instance, "species_map") and isinstance(env_instance.species_map, Mapping):
        engine.config.species_map = dict(env_instance.species_map)
    if hasattr(env_instance, "initial_total_moles") and isinstance(env_instance.initial_total_moles, Mapping):
        engine.config.initial_total_moles = dict(env_instance.initial_total_moles)
    if hasattr(env_instance, "element_matrix") and isinstance(env_instance.element_matrix, Mapping):
        engine.config.element_matrix = dict(env_instance.element_matrix)
    return engine.enforce_chemical_constraints(state_array)


def apply_all_chemical_constraints(env_instance: Any, state_array: np.ndarray) -> np.ndarray:
    engine = _get_engine()
    apply_chemical_processes(env_instance, state_array)
    return engine.enforce_chemical_constraints(state_array)


if __name__ == "__main__":
    print("\n=== Running Chemistry Constraints ===\n")
    printer.status("TEST", "Chemistry Constraints initialized", "info")

    engine = ChemistryEngine(
        {
            "dt": 0.05,
            "delta_H_reaction": -25000.0,
            "solubility_limits": {"C": 2.0, "D": 2.0},
        }
    )

    state = np.array([
        1.0,
        1.2,
        0.1,
        0.0,
        0.05,
        298.15,
        1e-7,
        1.0,
        0.0,
        0.0,
    ], dtype=float)

    printer.pretty("INITIAL_STATE", state.tolist(), "info")

    for step in range(5):
        engine.apply_all(state, dt=0.05)
        printer.pretty(
            f"STEP_{step + 1}",
            {
                "state": state.tolist(),
                "diagnostics": engine.diagnostics(state),
            },
            "success",
        )

    printer.pretty("RECENT_HISTORY", engine.recent_history(), "success")

    print("\n=== Test ran successfully ===\n")
