"""
Biology constraints and ecosystem dynamics for the SLAI Environment.

This module provides functions to simulate biological processes such as
population growth, nutrient limitation, predation, toxicity, temperature
response, organism health, stress accumulation, and adaptation. The functions
are intended to be called as part of the environment's step simulation when the
SLAI environment is managing its own system state.

The state vector is assumed to represent a biological system. A typical default
layout is:
- state_array[0]: primary population / biomass density
- state_array[1]: predator or competing population
- state_array[2]: nutrient / food resource availability
- state_array[3]: toxin / waste burden
- state_array[4]: temperature (K)
- state_array[5]: health / vitality score in [0, 1]
- state_array[6]: stress score in [0, 1]
- state_array[7]: adaptation / resilience score in [0, 1]
- state_array[8]: effective carrying capacity modifier or reserve slot

The exact layout can be overridden through configuration or by setting
attributes on the environment instance passed into the legacy wrapper
functions.
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field
from collections import deque
from collections.abc import Mapping
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.base_errors import *
from ..utils.base_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Biology Constraints")
printer = PrettyPrinter()


@dataclass
class BiologyConfig:
    """Configuration parameters for the biology engine."""

    dt: float = 0.02
    numeric_tolerance: float = 1e-12
    enable_history: bool = True
    history_limit: int = 200

    species_map: Dict[str, int] = field(
        default_factory=lambda: {
            "population": 0,
            "predator_population": 1,
            "nutrient": 2,
            "toxin": 3,
            "temperature": 4,
            "health": 5,
            "stress": 6,
            "adaptation": 7,
            "capacity_modifier": 8,
        }
    )

    intrinsic_growth_rate: float = 0.8
    carrying_capacity: float = 100.0
    enable_allee_effect: bool = False
    allee_threshold: float = 1.0
    min_viable_population: float = 0.0

    enable_predation: bool = False
    predation_rate: float = 0.01
    predator_efficiency: float = 0.2
    predator_decay_rate: float = 0.05
    competition_coefficient: float = 0.0

    enable_nutrient_cycling: bool = True
    nutrient_regen_rate: float = 0.5
    nutrient_consumption_rate: float = 0.02
    nutrient_half_saturation: float = 5.0
    nutrient_yield: float = 1.0

    enable_toxin_dynamics: bool = True
    toxin_generation_rate: float = 0.001
    toxin_decay_rate: float = 0.05
    toxin_damage_rate: float = 0.1
    toxin_tolerance: float = 0.25

    enable_temperature_response: bool = True
    optimal_temperature: float = 310.15
    temperature_tolerance: float = 10.0
    temperature_damage_rate: float = 0.05
    environment_relaxation_rate: float = 0.01
    ambient_temperature: float = 310.15

    enable_health_dynamics: bool = True
    health_recovery_rate: float = 0.05
    stress_recovery_rate: float = 0.05
    stress_accumulation_rate: float = 0.1

    enable_adaptation: bool = True
    adaptation_rate: float = 0.02
    mutation_noise_scale: float = 0.0
    adaptation_protection_factor: float = 0.5

    min_population: float = 0.0
    max_population: float = 1e9
    min_nutrient: float = 0.0
    max_nutrient: float = 1e9
    min_toxin: float = 0.0
    max_toxin: float = 1e9
    min_temperature: float = 0.0
    max_temperature: float = 1e4
    min_health: float = 0.0
    max_health: float = 1.0
    min_stress: float = 0.0
    max_stress: float = 1.0
    min_adaptation: float = 0.0
    max_adaptation: float = 1.0

    enforce_bounds: bool = True
    snapshot_pretty: bool = True


@dataclass(frozen=True)
class BiologyStepSummary:
    """Summary of one biology simulation step."""

    timestamp: str
    dt: float
    population_before: float
    population_after: float
    predator_before: Optional[float]
    predator_after: Optional[float]
    nutrient_before: Optional[float]
    nutrient_after: Optional[float]
    toxin_before: Optional[float]
    toxin_after: Optional[float]
    temperature_before: Optional[float]
    temperature_after: Optional[float]
    health_before: Optional[float]
    health_after: Optional[float]
    stress_before: Optional[float]
    stress_after: Optional[float]
    adaptation_before: Optional[float]
    adaptation_after: Optional[float]
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "dt": self.dt,
            "population_before": self.population_before,
            "population_after": self.population_after,
            "predator_before": self.predator_before,
            "predator_after": self.predator_after,
            "nutrient_before": self.nutrient_before,
            "nutrient_after": self.nutrient_after,
            "toxin_before": self.toxin_before,
            "toxin_after": self.toxin_after,
            "temperature_before": self.temperature_before,
            "temperature_after": self.temperature_after,
            "health_before": self.health_before,
            "health_after": self.health_after,
            "stress_before": self.stress_before,
            "stress_after": self.stress_after,
            "adaptation_before": self.adaptation_before,
            "adaptation_after": self.adaptation_after,
            "notes": list(self.notes),
        }


class BiologyEngine:
    """
    Centralized biology engine for ecosystem and organism-level constraints.

    The engine applies population growth, environmental limitation, predation,
    toxicity, temperature response, health/stress adaptation, and hard bounds.
    """

    CONSTANTS = {
        "N_A": 6.02214076e23,
        "k_B": 1.380649e-23,
        "R": 8.31446261815324,
        "BODY_TEMP_HUMAN_K": 310.15,
        "PHYSIOLOGICAL_PH": 7.4,
        "SECONDS_PER_DAY": 86400.0,
        "CAL_TO_J": 4.184,
    }

    def __init__(self, config: Optional[Union[BiologyConfig, Mapping[str, Any]]] = None) -> None:
        if config is None:
            loaded = get_config_section("biology_constraints") or {}
            self.config = self._config_from_mapping(loaded)
        elif isinstance(config, BiologyConfig):
            self.config = config
        elif isinstance(config, Mapping):
            self.config = self._config_from_mapping(config)
        else:
            raise BaseValidationError(
                "BiologyEngine config must be None, a mapping, or BiologyConfig.",
                None,
                component="BiologyEngine",
                operation="__init__",
                context={"received_type": type(config).__name__},
            )

        self.constants = dict(self.CONSTANTS)
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self.config.history_limit)
        self._validate_config()
        logger.info("Biology Constraints successfully initialized")

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _config_from_mapping(self, raw: Mapping[str, Any]) -> BiologyConfig:
        ensure_mapping(raw, "biology_constraints", error_cls=BaseConfigurationError)
        species_map = dict(raw.get("species_map") or BiologyConfig().species_map)
        return BiologyConfig(
            dt=coerce_float(raw.get("dt", 0.02), 0.02, minimum=0.0),
            numeric_tolerance=coerce_float(raw.get("numeric_tolerance", 1e-12), 1e-12, minimum=0.0),
            enable_history=coerce_bool(raw.get("enable_history", True), True),
            history_limit=coerce_int(raw.get("history_limit", 200), 200, minimum=1),
            species_map=species_map,
            intrinsic_growth_rate=coerce_float(raw.get("intrinsic_growth_rate", 0.8), 0.8),
            carrying_capacity=coerce_float(raw.get("carrying_capacity", 100.0), 100.0, minimum=1e-12),
            enable_allee_effect=coerce_bool(raw.get("enable_allee_effect", False), False),
            allee_threshold=coerce_float(raw.get("allee_threshold", 1.0), 1.0, minimum=0.0),
            min_viable_population=coerce_float(raw.get("min_viable_population", 0.0), 0.0, minimum=0.0),
            enable_predation=coerce_bool(raw.get("enable_predation", False), False),
            predation_rate=coerce_float(raw.get("predation_rate", 0.01), 0.01, minimum=0.0),
            predator_efficiency=coerce_float(raw.get("predator_efficiency", 0.2), 0.2, minimum=0.0),
            predator_decay_rate=coerce_float(raw.get("predator_decay_rate", 0.05), 0.05, minimum=0.0),
            competition_coefficient=coerce_float(raw.get("competition_coefficient", 0.0), 0.0, minimum=0.0),
            enable_nutrient_cycling=coerce_bool(raw.get("enable_nutrient_cycling", True), True),
            nutrient_regen_rate=coerce_float(raw.get("nutrient_regen_rate", 0.5), 0.5, minimum=0.0),
            nutrient_consumption_rate=coerce_float(raw.get("nutrient_consumption_rate", 0.02), 0.02, minimum=0.0),
            nutrient_half_saturation=coerce_float(raw.get("nutrient_half_saturation", 5.0), 5.0, minimum=1e-12),
            nutrient_yield=coerce_float(raw.get("nutrient_yield", 1.0), 1.0, minimum=0.0),
            enable_toxin_dynamics=coerce_bool(raw.get("enable_toxin_dynamics", True), True),
            toxin_generation_rate=coerce_float(raw.get("toxin_generation_rate", 0.001), 0.001, minimum=0.0),
            toxin_decay_rate=coerce_float(raw.get("toxin_decay_rate", 0.05), 0.05, minimum=0.0),
            toxin_damage_rate=coerce_float(raw.get("toxin_damage_rate", 0.1), 0.1, minimum=0.0),
            toxin_tolerance=coerce_float(raw.get("toxin_tolerance", 0.25), 0.25, minimum=0.0),
            enable_temperature_response=coerce_bool(raw.get("enable_temperature_response", True), True),
            optimal_temperature=coerce_float(raw.get("optimal_temperature", 310.15), 310.15, minimum=0.0),
            temperature_tolerance=coerce_float(raw.get("temperature_tolerance", 10.0), 10.0, minimum=1e-12),
            temperature_damage_rate=coerce_float(raw.get("temperature_damage_rate", 0.05), 0.05, minimum=0.0),
            environment_relaxation_rate=coerce_float(raw.get("environment_relaxation_rate", 0.01), 0.01, minimum=0.0),
            ambient_temperature=coerce_float(raw.get("ambient_temperature", 310.15), 310.15, minimum=0.0),
            enable_health_dynamics=coerce_bool(raw.get("enable_health_dynamics", True), True),
            health_recovery_rate=coerce_float(raw.get("health_recovery_rate", 0.05), 0.05, minimum=0.0),
            stress_recovery_rate=coerce_float(raw.get("stress_recovery_rate", 0.05), 0.05, minimum=0.0),
            stress_accumulation_rate=coerce_float(raw.get("stress_accumulation_rate", 0.1), 0.1, minimum=0.0),
            enable_adaptation=coerce_bool(raw.get("enable_adaptation", True), True),
            adaptation_rate=coerce_float(raw.get("adaptation_rate", 0.02), 0.02, minimum=0.0),
            mutation_noise_scale=coerce_float(raw.get("mutation_noise_scale", 0.0), 0.0, minimum=0.0),
            adaptation_protection_factor=coerce_float(raw.get("adaptation_protection_factor", 0.5), 0.5, minimum=0.0),
            min_population=coerce_float(raw.get("min_population", 0.0), 0.0),
            max_population=coerce_float(raw.get("max_population", 1e9), 1e9, minimum=1e-12),
            min_nutrient=coerce_float(raw.get("min_nutrient", 0.0), 0.0),
            max_nutrient=coerce_float(raw.get("max_nutrient", 1e9), 1e9, minimum=1e-12),
            min_toxin=coerce_float(raw.get("min_toxin", 0.0), 0.0),
            max_toxin=coerce_float(raw.get("max_toxin", 1e9), 1e9, minimum=1e-12),
            min_temperature=coerce_float(raw.get("min_temperature", 0.0), 0.0),
            max_temperature=coerce_float(raw.get("max_temperature", 1e4), 1e4, minimum=1e-12),
            min_health=coerce_float(raw.get("min_health", 0.0), 0.0),
            max_health=coerce_float(raw.get("max_health", 1.0), 1.0, minimum=0.0),
            min_stress=coerce_float(raw.get("min_stress", 0.0), 0.0),
            max_stress=coerce_float(raw.get("max_stress", 1.0), 1.0, minimum=0.0),
            min_adaptation=coerce_float(raw.get("min_adaptation", 0.0), 0.0),
            max_adaptation=coerce_float(raw.get("max_adaptation", 1.0), 1.0, minimum=0.0),
            enforce_bounds=coerce_bool(raw.get("enforce_bounds", True), True),
            snapshot_pretty=coerce_bool(raw.get("snapshot_pretty", True), True),
        )

    def _validate_config(self) -> None:
        cfg = self.config
        ensure_mapping(cfg.species_map, "species_map", error_cls=BaseConfigurationError)
        ensure_condition(
            "population" in cfg.species_map,
            "species_map must define a 'population' index.",
            error_cls=BaseConfigurationError,
            context={"species_map": cfg.species_map},
        )
        for key in ("dt", "carrying_capacity", "history_limit", "temperature_tolerance"):
            ensure_condition(
                getattr(cfg, key) >= 0,
                f"Configuration value '{key}' must be non-negative.",
                error_cls=BaseConfigurationError,
                context={"key": key, "value": getattr(cfg, key)},
            )
        ensure_condition(
            cfg.min_population <= cfg.max_population,
            "Population bounds are invalid.",
            error_cls=BaseConfigurationError,
            context={"min_population": cfg.min_population, "max_population": cfg.max_population},
        )
        ensure_condition(
            cfg.min_health <= cfg.max_health and cfg.min_stress <= cfg.max_stress,
            "Health or stress bounds are invalid.",
            error_cls=BaseConfigurationError,
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _idx(self, name: str) -> Optional[int]:
        value = self.config.species_map.get(name)
        return int(value) if value is not None else None

    def _has(self, state: np.ndarray, name: str) -> bool:
        idx = self._idx(name)
        return idx is not None and 0 <= idx < len(state)

    def _get(self, state: np.ndarray, name: str, default: float = 0.0) -> float:
        idx = self._idx(name)
        if idx is None or idx >= len(state):
            return float(default)
        return float(state[idx])

    def _set(self, state: np.ndarray, name: str, value: float) -> None:
        idx = self._idx(name)
        if idx is not None and idx < len(state):
            state[idx] = value

    def _record_history(self, summary: BiologyStepSummary) -> None:
        if self.config.enable_history:
            self._history.append(summary.to_dict())

    def recent_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        count = coerce_int(limit, 20, minimum=1)
        return list(self._history)[-count:]

    def stats(self) -> Dict[str, Any]:
        return {
            "history_length": len(self._history),
            "species_map": dict(self.config.species_map),
            "carrying_capacity": self.config.carrying_capacity,
            "enable_predation": self.config.enable_predation,
            "enable_nutrient_cycling": self.config.enable_nutrient_cycling,
            "enable_toxin_dynamics": self.config.enable_toxin_dynamics,
            "enable_temperature_response": self.config.enable_temperature_response,
            "enable_adaptation": self.config.enable_adaptation,
        }

    # ------------------------------------------------------------------
    # Core biology logic
    # ------------------------------------------------------------------
    def apply_biological_processes(self, state: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        ensure_condition(
            isinstance(state, np.ndarray),
            "state must be a numpy ndarray.",
            error_cls=BaseValidationError,
            context={"received_type": type(state).__name__},
        )
        if state.ndim != 1:
            raise BaseValidationError(
                "Biology state array must be one-dimensional.",
                None,
                component="BiologyEngine",
                operation="apply_biological_processes",
                context={"shape": tuple(state.shape)},
            )

        step_dt = self.config.dt if dt is None else coerce_float(dt, self.config.dt, minimum=0.0)
        if step_dt <= 0:
            return state

        pop_before = self._get(state, "population")
        pred_before = self._get(state, "predator_population") if self._has(state, "predator_population") else None
        nutrient_before = self._get(state, "nutrient") if self._has(state, "nutrient") else None
        toxin_before = self._get(state, "toxin") if self._has(state, "toxin") else None
        temp_before = self._get(state, "temperature") if self._has(state, "temperature") else None
        health_before = self._get(state, "health", 1.0) if self._has(state, "health") else None
        stress_before = self._get(state, "stress", 0.0) if self._has(state, "stress") else None
        adapt_before = self._get(state, "adaptation", 0.0) if self._has(state, "adaptation") else None

        notes: List[str] = []

        population = max(self.config.min_population, pop_before)
        predator = max(0.0, pred_before if pred_before is not None else 0.0)
        nutrient = max(0.0, nutrient_before if nutrient_before is not None else self.config.max_nutrient)
        toxin = max(0.0, toxin_before if toxin_before is not None else 0.0)
        temperature = temp_before if temp_before is not None else self.config.optimal_temperature
        health = health_before if health_before is not None else 1.0
        stress = stress_before if stress_before is not None else 0.0
        adaptation = adapt_before if adapt_before is not None else 0.0

        # Carrying capacity can optionally be modulated by a state slot.
        capacity_modifier = self._get(state, "capacity_modifier", 1.0) if self._has(state, "capacity_modifier") else 1.0
        effective_capacity = max(self.config.numeric_tolerance, self.config.carrying_capacity * max(self.config.numeric_tolerance, capacity_modifier))

        nutrient_factor = 1.0
        if self.config.enable_nutrient_cycling and self._has(state, "nutrient"):
            nutrient_factor = nutrient / (self.config.nutrient_half_saturation + max(nutrient, self.config.numeric_tolerance))

        if self.config.enable_temperature_response:
            temp_delta = temperature - self.config.optimal_temperature
            temp_penalty = np.exp(-((temp_delta / self.config.temperature_tolerance) ** 2))
        else:
            temp_penalty = 1.0

        adaptation_guard = 1.0 - min(1.0, adaptation) * self.config.adaptation_protection_factor
        toxin_excess = max(0.0, toxin - self.config.toxin_tolerance)
        toxin_penalty = max(0.0, 1.0 - toxin_excess * self.config.toxin_damage_rate * adaptation_guard)
        effective_health = np.clip(health * toxin_penalty * temp_penalty, self.config.min_health, self.config.max_health)

        logistic_term = self.config.intrinsic_growth_rate * population * (1.0 - (population / effective_capacity))
        growth_term = logistic_term * nutrient_factor * effective_health

        if self.config.enable_allee_effect and population < self.config.allee_threshold:
            growth_term *= (population / max(self.config.allee_threshold, self.config.numeric_tolerance))
            notes.append("allee_effect")

        competition_loss = self.config.competition_coefficient * population * predator if predator > 0 else 0.0

        predation_loss = 0.0
        predator_gain = 0.0
        if self.config.enable_predation and self._has(state, "predator_population"):
            predation_loss = self.config.predation_rate * population * predator
            predator_gain = self.config.predator_efficiency * predation_loss
            predator = predator + (predator_gain - self.config.predator_decay_rate * predator) * step_dt
            notes.append("predation")

        nutrient_consumption = 0.0
        if self.config.enable_nutrient_cycling and self._has(state, "nutrient"):
            nutrient_consumption = self.config.nutrient_consumption_rate * population * step_dt
            nutrient_regen = self.config.nutrient_regen_rate * step_dt
            nutrient = nutrient + nutrient_regen - nutrient_consumption
            notes.append("nutrient_cycling")

        if self.config.enable_toxin_dynamics and self._has(state, "toxin"):
            toxin = toxin + (self.config.toxin_generation_rate * population - self.config.toxin_decay_rate * toxin) * step_dt
            notes.append("toxin_dynamics")

        # Health/stress dynamics
        if self.config.enable_health_dynamics:
            environmental_damage = max(0.0, 1.0 - temp_penalty) * self.config.temperature_damage_rate
            toxin_damage = toxin_excess * self.config.toxin_damage_rate
            stress_input = (environmental_damage + toxin_damage) * self.config.stress_accumulation_rate
            stress = stress + stress_input * step_dt - self.config.stress_recovery_rate * stress * step_dt
            health = health + self.config.health_recovery_rate * (1.0 - health) * step_dt - (environmental_damage + toxin_damage + stress) * step_dt
            notes.append("health_dynamics")

        if self.config.enable_adaptation and self._has(state, "adaptation"):
            challenge = abs(temperature - self.config.optimal_temperature) / max(self.config.temperature_tolerance, self.config.numeric_tolerance)
            challenge += toxin_excess
            adaptation = adaptation + self.config.adaptation_rate * challenge * step_dt * (1.0 - adaptation)
            if self.config.mutation_noise_scale > 0:
                adaptation += float(np.random.normal(0.0, self.config.mutation_noise_scale)) * step_dt
            notes.append("adaptation")

        if self.config.enable_temperature_response and self._has(state, "temperature"):
            population_heat = 0.001 * population
            temperature += (self.config.ambient_temperature - temperature) * self.config.environment_relaxation_rate * step_dt
            temperature += population_heat * step_dt
            notes.append("temperature_response")

        population_delta = (growth_term - predation_loss - competition_loss) * step_dt
        population = population + population_delta

        if self.config.min_viable_population > 0 and population < self.config.min_viable_population:
            population = 0.0
            notes.append("min_viable_population_collapse")

        self._set(state, "population", population)
        if self._has(state, "predator_population"):
            self._set(state, "predator_population", predator)
        if self._has(state, "nutrient"):
            self._set(state, "nutrient", nutrient)
        if self._has(state, "toxin"):
            self._set(state, "toxin", toxin)
        if self._has(state, "temperature"):
            self._set(state, "temperature", temperature)
        if self._has(state, "health"):
            self._set(state, "health", health)
        if self._has(state, "stress"):
            self._set(state, "stress", stress)
        if self._has(state, "adaptation"):
            self._set(state, "adaptation", adaptation)

        self.enforce_biological_constraints(state)

        summary = BiologyStepSummary(
            timestamp=utc_now_iso(),
            dt=step_dt,
            population_before=pop_before,
            population_after=self._get(state, "population"),
            predator_before=pred_before,
            predator_after=self._get(state, "predator_population") if self._has(state, "predator_population") else None,
            nutrient_before=nutrient_before,
            nutrient_after=self._get(state, "nutrient") if self._has(state, "nutrient") else None,
            toxin_before=toxin_before,
            toxin_after=self._get(state, "toxin") if self._has(state, "toxin") else None,
            temperature_before=temp_before,
            temperature_after=self._get(state, "temperature") if self._has(state, "temperature") else None,
            health_before=health_before,
            health_after=self._get(state, "health") if self._has(state, "health") else None,
            stress_before=stress_before,
            stress_after=self._get(state, "stress") if self._has(state, "stress") else None,
            adaptation_before=adapt_before,
            adaptation_after=self._get(state, "adaptation") if self._has(state, "adaptation") else None,
            notes=tuple(notes),
        )
        self._record_history(summary)
        return state

    def enforce_biological_constraints(self, state: np.ndarray) -> np.ndarray:
        ensure_condition(
            isinstance(state, np.ndarray),
            "state must be a numpy ndarray.",
            error_cls=BaseValidationError,
            context={"received_type": type(state).__name__},
        )
        if not self.config.enforce_bounds:
            return state

        if self._has(state, "population"):
            self._set(state, "population", float(np.clip(self._get(state, "population"), self.config.min_population, self.config.max_population)))
        if self._has(state, "predator_population"):
            self._set(state, "predator_population", float(np.clip(self._get(state, "predator_population"), self.config.min_population, self.config.max_population)))
        if self._has(state, "nutrient"):
            self._set(state, "nutrient", float(np.clip(self._get(state, "nutrient"), self.config.min_nutrient, self.config.max_nutrient)))
        if self._has(state, "toxin"):
            self._set(state, "toxin", float(np.clip(self._get(state, "toxin"), self.config.min_toxin, self.config.max_toxin)))
        if self._has(state, "temperature"):
            self._set(state, "temperature", float(np.clip(self._get(state, "temperature"), self.config.min_temperature, self.config.max_temperature)))
        if self._has(state, "health"):
            self._set(state, "health", float(np.clip(self._get(state, "health"), self.config.min_health, self.config.max_health)))
        if self._has(state, "stress"):
            self._set(state, "stress", float(np.clip(self._get(state, "stress"), self.config.min_stress, self.config.max_stress)))
        if self._has(state, "adaptation"):
            self._set(state, "adaptation", float(np.clip(self._get(state, "adaptation"), self.config.min_adaptation, self.config.max_adaptation)))

        return state

    def apply_all(self, state: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        self.apply_biological_processes(state, dt=dt)
        self.enforce_biological_constraints(state)
        return state


# ----------------------------------------------------------------------
# Legacy compatibility API
# ----------------------------------------------------------------------
_engine: Optional[BiologyEngine] = None


def _get_engine() -> BiologyEngine:
    global _engine
    if _engine is None:
        _engine = BiologyEngine()
    return _engine


def apply_biological_constants(env_instance: Any) -> Dict[str, float]:
    engine = _get_engine()
    env_instance.constants = dict(engine.constants)
    return env_instance.constants


def _sync_engine_from_env(engine: BiologyEngine, env_instance: Any) -> None:
    cfg = engine.config
    sync_fields = [
        "dt",
        "intrinsic_growth_rate",
        "carrying_capacity",
        "enable_predation",
        "predation_rate",
        "predator_efficiency",
        "predator_decay_rate",
        "competition_coefficient",
        "enable_nutrient_cycling",
        "nutrient_regen_rate",
        "nutrient_consumption_rate",
        "nutrient_half_saturation",
        "nutrient_yield",
        "enable_toxin_dynamics",
        "toxin_generation_rate",
        "toxin_decay_rate",
        "toxin_damage_rate",
        "toxin_tolerance",
        "enable_temperature_response",
        "optimal_temperature",
        "temperature_tolerance",
        "temperature_damage_rate",
        "environment_relaxation_rate",
        "ambient_temperature",
        "enable_health_dynamics",
        "health_recovery_rate",
        "stress_recovery_rate",
        "stress_accumulation_rate",
        "enable_adaptation",
        "adaptation_rate",
        "mutation_noise_scale",
        "adaptation_protection_factor",
    ]
    for field_name in sync_fields:
        if hasattr(env_instance, field_name):
            setattr(cfg, field_name, getattr(env_instance, field_name))
    if hasattr(env_instance, "species_map") and isinstance(getattr(env_instance, "species_map"), Mapping):
        cfg.species_map = dict(getattr(env_instance, "species_map"))


def apply_biological_processes(env_instance: Any, state_array: np.ndarray) -> np.ndarray:
    engine = _get_engine()
    _sync_engine_from_env(engine, env_instance)
    dt = getattr(env_instance, "dt", engine.config.dt)
    return engine.apply_biological_processes(state_array, dt=dt)


def enforce_biological_constraints(env_instance: Any, state_array: np.ndarray) -> np.ndarray:
    engine = _get_engine()
    _sync_engine_from_env(engine, env_instance)
    return engine.enforce_biological_constraints(state_array)


def apply_all_biological_constraints(env_instance: Any, state_array: np.ndarray) -> np.ndarray:
    engine = _get_engine()
    _sync_engine_from_env(engine, env_instance)
    dt = getattr(env_instance, "dt", engine.config.dt)
    return engine.apply_all(state_array, dt=dt)


if __name__ == "__main__":
    print("\n=== Running Biology Constraints ===\n")
    printer.status("TEST", "Biology Constraints initialized", "info")

    engine = BiologyEngine()

    state = np.array([
        25.0,   # population
        4.0,    # predator population
        80.0,   # nutrient
        0.05,   # toxin
        310.15, # temperature
        0.9,    # health
        0.1,    # stress
        0.2,    # adaptation
        1.0,    # capacity modifier
    ], dtype=float)

    print(f"Initial state: {state}")
    for step in range(10):
        state = engine.apply_all(state, dt=0.05)
        print(
            f"Step {step + 1}: "
            f"pop={state[0]:7.3f}, pred={state[1]:7.3f}, nutrient={state[2]:7.3f}, "
            f"toxin={state[3]:7.3f}, temp={state[4]:7.3f}, health={state[5]:6.3f}, "
            f"stress={state[6]:6.3f}, adapt={state[7]:6.3f}"
        )

    printer.pretty("BIOLOGY_STATS", engine.stats(), "success")
    printer.pretty("RECENT_HISTORY", engine.recent_history(), "success")

    print("\n=== Test ran successfully ===\n")
