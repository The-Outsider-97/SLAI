"""
Engineering calculations and design checks for the SLAI Environment.

This module provides a production-ready, config-driven engineering layer that
owns five domains behind one validated engine:

- Medical (biomedical) engineering: anthropometrics, hemodynamics, gas
  exchange, ECG/biosignal helpers, one-compartment kinetics, radiation dose,
  and device-safety limits (leakage current, vital-sign alarm ranges).
- Civil engineering: load combinations, beam response, column buckling,
  reinforced-concrete flexure, bearing capacity, seismic base shear,
  hydrology/hydraulics, and traffic flow.
- Computer engineering: CPU performance, parallel speedup, memory hierarchy,
  cache addressing, power, pipelines, roofline, queueing, reliability, RAID.
- Electronics / communications engineering: circuit fundamentals, filters and
  resonance, op-amps, ADC metrics, noise, thermal limits, link budgets, channel
  capacity, bit-error rates, and error-control sizing.
- Mechanical engineering: multiaxial stress, shafts, gears, fasteners,
  bearings, fatigue, pressure vessels, pipe flow, heat exchangers, vibration,
  and tolerance stacks.

Conventions
-----------
- All quantities are SI unless the parameter name carries an explicit unit
  suffix (``_mmhg``, ``_ml``, ``_mm``, ``_db``, ``_rpm``, ``_mah``...).
- Every public calculation validates its inputs through the shared base error
  and helper layers, then returns either a float or a plain dictionary.
- Every public calculation is recorded in a bounded audit history; every
  design check produces an ``EngineeringCheck`` with a utilization ratio.
- Physical constants are shared with ``PhysicsEngine.CONSTANTS`` instead of
  being redeclared here; this module only owns engineering-specific defaults,
  which live in ``base_config.yaml`` under ``base_engineering``.

Scope note: the medical routines are engineering computations (device and
signal design, physiological modelling). They are not clinical decision
support and never recommend doses or treatments.
"""

from __future__ import annotations

import math
import statistics

from dataclasses import dataclass, field
from collections import deque
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Deque, Dict, List, Optional, Sequence, Tuple

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.base_errors import *
from ..utils.base_helpers import *
from .physics_constraints import PhysicsEngine
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Base Engineering")
printer = PrettyPrinter()

# ----------------------------------------------------------------------
# Module constants (conversion factors and sentinels only; physical
# constants come from PhysicsEngine.CONSTANTS)
# ----------------------------------------------------------------------
MMHG_TO_PA = 133.322387415
_UTILIZATION_CAP = 1.0e6
_LIMIT_MODES: Tuple[str, ...] = ("max", "min", "range")
_MATERIAL_KEYS: Tuple[str, ...] = (
    "youngs_modulus",
    "yield_strength",
    "ultimate_strength",
    "density",
    "poissons_ratio",
)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


# ----------------------------------------------------------------------
# Audit records
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class EngineeringRecord:
    """Structured audit record for one engineering calculation."""

    timestamp: str
    domain: str
    operation: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_json_safe({
            "record_type": "calculation",
            "timestamp": self.timestamp,
            "domain": self.domain,
            "operation": self.operation,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "notes": self.notes,
        })


@dataclass(frozen=True)
class EngineeringCheck:
    """Design-limit check result with a normalized utilization ratio."""

    timestamp: str
    domain: str
    name: str
    mode: str
    demand: float
    limit: Any
    utilization: float
    passed: bool
    severity: str
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_json_safe({
            "record_type": "check",
            "timestamp": self.timestamp,
            "domain": self.domain,
            "name": self.name,
            "mode": self.mode,
            "demand": self.demand,
            "limit": self.limit,
            "utilization": self.utilization,
            "passed": self.passed,
            "severity": self.severity,
            "notes": self.notes,
        })


class _AuditRecorder:
    """Bounded history and counters shared by every engineering domain."""

    def __init__(self, enabled: bool, limit: int):
        self.enabled = enabled
        self.history: Deque[Dict[str, Any]] = deque(maxlen=limit)
        self.stats: Dict[str, Any] = {
            "calculations": 0,
            "checks": 0,
            "checks_passed": 0,
            "checks_failed": 0,
            "by_domain": {},
        }

    def _count_domain(self, domain: str) -> None:
        by_domain = self.stats["by_domain"]
        by_domain[domain] = by_domain.get(domain, 0) + 1

    def record_calculation(self, record: EngineeringRecord) -> None:
        self.stats["calculations"] += 1
        self._count_domain(record.domain)
        if self.enabled:
            self.history.append(record.to_dict())

    def record_check(self, check: EngineeringCheck) -> None:
        self.stats["checks"] += 1
        self.stats["checks_passed" if check.passed else "checks_failed"] += 1
        if self.enabled:
            self.history.append(check.to_dict())


@dataclass(frozen=True)
class _SharedContext:
    """Immutable settings and libraries shared by all domains."""

    constants: Mapping[str, float]
    materials: Mapping[str, Mapping[str, float]]
    default_material: str
    default_safety_factor: float
    numeric_tolerance: float
    strict_checks: bool
    warn_utilization: float
    critical_utilization: float
    sampling_oversample_factor: float


# ----------------------------------------------------------------------
# Shared domain infrastructure
# ----------------------------------------------------------------------
class _DomainBase:
    """
    Shared validation, limit-checking, and audit plumbing for all domains.

    Subclasses implement ``_configure`` (parse their config section into
    attributes) and ``_validate_config``; everything else is inherited so
    no domain re-implements validation, history, or check logic.
    """

    DOMAIN = "base"
    CONFIG_FIELDS: Tuple[str, ...] = ()

    def __init__(self, section: Mapping[str, Any], recorder: _AuditRecorder, shared: _SharedContext):
        ensure_mapping(
            section,
            self.DOMAIN,
            config=dict(section) if isinstance(section, Mapping) else None,
            error_cls=BaseConfigurationError,
            component=self.component,
            operation="__init__",
        )
        self.config: Dict[str, Any] = dict(section)
        self._recorder = recorder
        self.shared = shared
        self.constants = shared.constants
        self._configure(self.config)
        self._validate_config()

    @property
    def component(self) -> str:
        return f"EngineeringEngine.{self.DOMAIN}"

    # -- configuration -------------------------------------------------
    def _configure(self, mapping: Mapping[str, Any]) -> None:  # pragma: no cover - overridden
        return None

    def _validate_config(self) -> None:  # pragma: no cover - overridden
        return None

    def _config_as_dict(self) -> Dict[str, Any]:
        return {key: getattr(self, key) for key in self.CONFIG_FIELDS}

    def _config_choice(self, value: Any, options: Sequence[str], name: str) -> str:
        text = str(value).strip().lower()
        ensure_one_of(
            text, tuple(options), name,
            config=self.config, error_cls=BaseConfigurationError,
            component=self.component, operation="configuration",
        )
        return text

    def _float_map(self, raw: Any, name: str, *, minimum: Optional[float] = 0.0) -> Dict[str, float]:
        ensure_mapping(
            raw if raw is not None else {}, name,
            config=self.config, error_cls=BaseConfigurationError,
            component=self.component, operation="configuration",
        )
        return {
            str(key).strip().lower(): coerce_float(value, 0.0, minimum=minimum)
            for key, value in (raw or {}).items()
        }

    def _range_map(self, raw: Any, name: str) -> Dict[str, Tuple[float, float]]:
        ensure_mapping(
            raw if raw is not None else {}, name,
            config=self.config, error_cls=BaseConfigurationError,
            component=self.component, operation="configuration",
        )
        ranges: Dict[str, Tuple[float, float]] = {}
        for key, value in (raw or {}).items():
            pair = ensure_list(value)
            ensure_condition(
                len(pair) == 2 and coerce_float(pair[0]) < coerce_float(pair[1]),
                f"'{name}.{key}' must be a [low, high] pair with low < high.",
                config=self.config, error_cls=BaseConfigurationError,
                component=self.component, operation="configuration",
                context={"field": f"{name}.{key}", "received": to_json_safe(value)},
            )
            ranges[str(key).strip().lower()] = (coerce_float(pair[0]), coerce_float(pair[1]))
        return ranges

    # -- input validation ----------------------------------------------
    def _num(
        self,
        value: Any,
        name: str,
        *,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        positive: bool = False,
        allow_zero: bool = False,
    ) -> float:
        """Coerce to a finite float and enforce optional range constraints."""
        number = coerce_float(value, math.nan)
        ensure_condition(
            math.isfinite(number),
            f"'{name}' must be a finite number.",
            config=self.config,
            error_cls=BaseValidationError,
            component=self.component,
            operation="input_validation",
            context={"field": name, "received": to_json_safe(value)},
        )
        if positive:
            ensure_positive(
                number,
                name,
                allow_zero=allow_zero,
                config=self.config,
                error_cls=BaseValidationError,
                component=self.component,
                operation="input_validation",
            )
        if minimum is not None or maximum is not None:
            ensure_numeric_range(
                number,
                name,
                minimum=minimum,
                maximum=maximum,
                config=self.config,
                error_cls=BaseValidationError,
                component=self.component,
                operation="input_validation",
            )
        return number

    def _count(self, value: Any, name: str, *, minimum: int = 1) -> int:
        number = self._num(value, name, minimum=float(minimum))
        ensure_condition(
            float(number).is_integer(),
            f"'{name}' must be a whole number.",
            config=self.config, error_cls=BaseValidationError,
            component=self.component, operation="input_validation",
            context={"field": name, "received": number},
        )
        return int(number)

    def _values(self, values: Any, name: str, *, min_len: int = 1, positive: bool = False) -> List[float]:
        items = ensure_list(values)
        ensure_condition(
            len(items) >= min_len,
            f"'{name}' must contain at least {min_len} value(s).",
            config=self.config, error_cls=BaseValidationError,
            component=self.component, operation="input_validation",
            context={"field": name, "length": len(items)},
        )
        return [self._num(item, f"{name}[{index}]", positive=positive) for index, item in enumerate(items)]

    def _choice(self, value: Any, options: Sequence[str], name: str) -> str:
        text = str(value).strip().lower()
        ensure_one_of(
            text, tuple(options), name,
            config=self.config, error_cls=BaseValidationError,
            component=self.component, operation="input_validation",
        )
        return text

    def _require(
        self,
        condition: bool,
        message: str,
        *,
        error_cls: type[BaseError] = BaseValidationError,
        **context: Any,
    ) -> None:
        ensure_condition(
            condition, message,
            config=self.config, error_cls=error_cls,
            component=self.component, operation="calculation",
            context=context,
        )

    def _sampling_rate(self, max_frequency_hz: float) -> float:
        """Minimum practical sampling rate (shared by medical and electronics)."""
        return self.shared.sampling_oversample_factor * max_frequency_hz

    # -- audit ---------------------------------------------------------
    def _finish(self, operation: str, outputs: Any, *, notes: Optional[Any] = None, **inputs: Any) -> Any:
        payload = dict(outputs) if isinstance(outputs, Mapping) else {"value": outputs}
        safe_notes: Dict[str, Any]
        if notes is None:
            safe_notes = {}
        elif isinstance(notes, Mapping):
            safe_notes = dict(notes)
        else:
            safe_notes = {"value": notes}
        self._recorder.record_calculation(
            EngineeringRecord(
                timestamp=utc_now_iso(),
                domain=self.DOMAIN,
                operation=operation,
                inputs=to_json_safe(inputs),
                outputs=to_json_safe(payload),
                notes=to_json_safe(safe_notes),
            )
        )
        return outputs

    def evaluate_limit(self, name: str, demand: float, limit: Any, *, mode: str = "max", **notes: Any) -> EngineeringCheck:
        """
        Compare a demand against a limit and return a recorded check.

        Modes:
            max   - pass when demand <= limit; utilization = demand / limit
            min   - pass when demand >= limit; utilization = limit / demand
            range - ``limit`` is (low, high); utilization = distance from the
                    midpoint as a fraction of the half-width
        """
        mode = self._choice(mode, _LIMIT_MODES, "mode")
        value = self._num(demand, f"{name}.demand")
        tol = self.shared.numeric_tolerance

        if mode == "range":
            bounds = self._values(limit, f"{name}.limit", min_len=2)
            self._require(len(bounds) == 2 and bounds[0] < bounds[1],
                          f"'{name}.limit' must be (low, high) with low < high.", field=name)
            low, high = bounds
            utilization = abs(value - (low + high) / 2.0) / ((high - low) / 2.0)
            passed = low <= value <= high
            limit_value: Any = [low, high]
        else:
            bound = self._num(limit, f"{name}.limit")
            limit_value = bound
            if mode == "max":
                passed = value <= bound
                utilization = max(0.0, value / bound) if bound > tol else (0.0 if passed else _UTILIZATION_CAP)
            else:
                passed = value >= bound
                utilization = max(0.0, bound / value) if value > tol else (0.0 if passed else _UTILIZATION_CAP)

        utilization = min(_UTILIZATION_CAP, utilization)
        if not passed:
            severity = "critical" if utilization >= self.shared.critical_utilization else "high"
        else:
            severity = "medium" if utilization >= self.shared.warn_utilization else "low"

        check = EngineeringCheck(
            timestamp=utc_now_iso(),
            domain=self.DOMAIN,
            name=name,
            mode=mode,
            demand=value,
            limit=limit_value,
            utilization=utilization,
            passed=passed,
            severity=severity,
            notes=dict(notes),
        )
        self._recorder.record_check(check)

        if not passed:
            logger.warning(f"[{self.DOMAIN}] check '{name}' failed (utilization={utilization:.3f})")
            if self.shared.strict_checks:
                raise BaseRuntimeError(
                    f"Engineering check '{name}' failed (utilization={utilization:.3f}).",
                    self.config,
                    component=self.component,
                    operation="evaluate_limit",
                    severity=severity,
                    context=check.to_dict(),
                    resolution_hint="Revise the design inputs or relax the configured limit.",
                )
        return check


class _StructuralMixin:
    """
    Section, material, and safety-factor helpers shared by civil and mechanical.

    This mixin is designed to be combined with :class:`_DomainBase` in the
    concrete engineering domains (``CivilEngineering``, ``MechanicalEngineering``).
    It relies on the following members being provided by that base class:

    * ``self.shared``          -> :class:`_SharedContext` (materials, defaults,
                                  tolerances, sub-methods such as ``.materials``,
                                  ``.default_material``, ``.default_safety_factor``,
                                  ``.numeric_tolerance``).
    * ``self.config``          -> raw configuration mapping for error context.
    * ``self.component``       -> fully-qualified name for diagnostics.
    * ``self._num()``          -> finite-float coercion with range checks.
    * ``self._choice()``       -> validation against a closed set of options.
    * ``self._require()``      -> condition guard with structured error context.
    * ``self._finish()``       -> audit recording + transparent output echo.
    * ``self.evaluate_limit()``-> design check with utilization ratio.

    The declarations inside ``if TYPE_CHECKING:`` make those dependencies
    explicit for static analysis.  That block is erased at import time, so
    runtime behaviour is unchanged and the real implementations resolve
    through the MRO as before.
    """

    SECTION_SHAPES: Tuple[str, ...] = ("rectangle", "circle", "hollow_circle")

    if TYPE_CHECKING:
        # ----- Provided by _DomainBase in the concrete MRO -----
        DOMAIN: ClassVar[str]
        shared: "_SharedContext"
        config: Dict[str, Any]

        @property
        def component(self) -> str: ...

        def _num(
            self,
            value: Any,
            name: str,
            *,
            minimum: Optional[float] = None,
            maximum: Optional[float] = None,
            positive: bool = False,
            allow_zero: bool = False,
        ) -> float: ...

        def _choice(self, value: Any, options: Sequence[str], name: str) -> str: ...

        def _require(
            self,
            condition: bool,
            message: str,
            *,
            error_cls: type = BaseValidationError,
            **context: Any,
        ) -> None: ...

        def _finish(
            self,
            operation: str,
            outputs: Any,
            *,
            notes: Optional[Any] = None,
            **inputs: Any,
        ) -> Any: ...

        def evaluate_limit(
            self,
            name: str,
            demand: float,
            limit: Any,
            *,
            mode: str = "max",
            **notes: Any,
        ) -> "EngineeringCheck": ...

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------
    def material(self, name: Optional[str] = None) -> Dict[str, float]:
        """Return a copy of a configured material property set."""
        key = self._choice(
            name or self.shared.default_material,
            tuple(self.shared.materials),
            "material",
        )
        return dict(self.shared.materials[key])

    def section_properties(
        self,
        shape: str,
        *,
        width_m: Optional[float] = None,
        height_m: Optional[float] = None,
        diameter_m: Optional[float] = None,
        inner_diameter_m: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Area, second moment, section modulus, and (for circular shapes) polar moment."""
        shape = self._choice(shape, self.SECTION_SHAPES, "shape")
        polar: Optional[float] = None
        if shape == "rectangle":
            b = self._num(width_m, "width_m", positive=True)
            h = self._num(height_m, "height_m", positive=True)
            area, inertia, extreme = b * h, b * h ** 3 / 12.0, h / 2.0
        else:
            d_out = self._num(diameter_m, "diameter_m", positive=True)
            d_in = 0.0
            if shape == "hollow_circle":
                d_in = self._num(inner_diameter_m, "inner_diameter_m", positive=True)
                self._require(
                    d_in < d_out,
                    "'inner_diameter_m' must be smaller than 'diameter_m'.",
                    inner_diameter_m=d_in,
                    diameter_m=d_out,
                )
            area = math.pi * (d_out ** 2 - d_in ** 2) / 4.0
            inertia = math.pi * (d_out ** 4 - d_in ** 4) / 64.0
            polar = 2.0 * inertia
            extreme = d_out / 2.0
        result = {
            "area_m2": area,
            "second_moment_m4": inertia,
            "extreme_fiber_m": extreme,
            "section_modulus_m3": inertia / extreme,
            "radius_of_gyration_m": math.sqrt(inertia / area),
            "polar_moment_m4": polar,
        }
        return self._finish("section_properties", result, shape=shape)

    def bending_stress(
        self,
        moment_nm: float,
        second_moment_m4: float,
        extreme_fiber_m: float,
    ) -> float:
        """Flexure formula: sigma = M c / I."""
        m = self._num(moment_nm, "moment_nm")
        inertia = self._num(second_moment_m4, "second_moment_m4", positive=True)
        c = self._num(extreme_fiber_m, "extreme_fiber_m", positive=True)
        return self._finish(
            "bending_stress",
            m * c / inertia,
            moment_nm=m,
            second_moment_m4=inertia,
            extreme_fiber_m=c,
        )

    def factor_of_safety(
        self,
        capacity: float,
        demand: float,
        *,
        required: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Capacity/demand ratio checked against the required (default) safety factor."""
        cap = self._num(capacity, "capacity", positive=True)
        dem = self._num(demand, "demand", minimum=0.0)
        needed = self._num(
            required if required is not None else self.shared.default_safety_factor,
            "required",
            positive=True,
        )
        fos = cap / dem if dem > self.shared.numeric_tolerance else _UTILIZATION_CAP
        check = self.evaluate_limit(
            "factor_of_safety",
            min(fos, _UTILIZATION_CAP),
            needed,
            mode="min",
        )
        result = {
            "factor_of_safety": check.demand,
            "required": needed,
            "passed": check.passed,
            "check": check.to_dict(),
        }
        return self._finish("factor_of_safety", result, capacity=cap, demand=dem)


# ======================================================================
# Medical engineering
# ======================================================================
class MedicalEngineering(_DomainBase):
    """Biomedical engineering calculations (not clinical decision support)."""

    DOMAIN = "medical"
    DISCLAIMER = "Engineering computation only; not for clinical decision-making."

    BSA_FORMULAS: Dict[str, Callable[[float, float], float]] = {
        "mosteller": lambda w, h: math.sqrt(w * h / 3600.0),
        "dubois": lambda w, h: 0.007184 * w ** 0.425 * h ** 0.725,
        "haycock": lambda w, h: 0.024265 * w ** 0.5378 * h ** 0.3964,
    }
    # QT in milliseconds, RR in seconds.
    QT_FORMULAS: Dict[str, Callable[[float, float], float]] = {
        "bazett": lambda qt, rr: qt / math.sqrt(rr),
        "fridericia": lambda qt, rr: qt / rr ** (1.0 / 3.0),
        "framingham": lambda qt, rr: qt + 154.0 * (1.0 - rr),
        "hodges": lambda qt, rr: qt + 1.75 * (60.0 / rr - 60.0),
    }

    CONFIG_FIELDS = (
        "bsa_formula", "qt_formula", "hemoglobin_binding_capacity_ml_per_g",
        "oxygen_solubility_ml_per_dl_per_mmhg", "water_vapor_pressure_mmhg",
        "respiratory_quotient", "atmospheric_pressure_mmhg", "spo2_ratio_intercept",
        "spo2_ratio_slope", "blood_viscosity_pa_s", "radiation_weighting_factors",
        "tissue_weighting_factors", "leakage_current_limits_ua", "vital_sign_ranges",
        "biosignal_bands_hz",
    )

    def _configure(self, mapping: Mapping[str, Any]) -> None:
        self.bsa_formula = self._config_choice(mapping.get("bsa_formula", "mosteller"), tuple(self.BSA_FORMULAS), "bsa_formula")
        self.qt_formula = self._config_choice(mapping.get("qt_formula", "bazett"), tuple(self.QT_FORMULAS), "qt_formula")
        self.hemoglobin_binding_capacity_ml_per_g = coerce_float(mapping.get("hemoglobin_binding_capacity_ml_per_g", 1.34), 1.34, minimum=0.0)
        self.oxygen_solubility_ml_per_dl_per_mmhg = coerce_float(mapping.get("oxygen_solubility_ml_per_dl_per_mmhg", 0.003), 0.003, minimum=0.0)
        self.water_vapor_pressure_mmhg = coerce_float(mapping.get("water_vapor_pressure_mmhg", 47.0), 47.0, minimum=0.0)
        self.respiratory_quotient = coerce_float(mapping.get("respiratory_quotient", 0.8), 0.8, minimum=1.0e-6)
        self.atmospheric_pressure_mmhg = coerce_float(mapping.get("atmospheric_pressure_mmhg", 760.0), 760.0, minimum=1.0)
        self.spo2_ratio_intercept = coerce_float(mapping.get("spo2_ratio_intercept", 110.0), 110.0)
        self.spo2_ratio_slope = coerce_float(mapping.get("spo2_ratio_slope", 25.0), 25.0)
        self.blood_viscosity_pa_s = coerce_float(mapping.get("blood_viscosity_pa_s", 0.0035), 0.0035, minimum=1.0e-9)
        self.radiation_weighting_factors = self._float_map(mapping.get("radiation_weighting_factors"), "radiation_weighting_factors")
        self.tissue_weighting_factors = self._float_map(mapping.get("tissue_weighting_factors"), "tissue_weighting_factors")
        raw_leakage = mapping.get("leakage_current_limits_ua") or {}
        ensure_mapping(raw_leakage, "leakage_current_limits_ua", config=self.config,
                       error_cls=BaseConfigurationError, component=self.component, operation="configuration")
        self.leakage_current_limits_ua = {
            str(part).strip().lower(): self._float_map(conditions, f"leakage_current_limits_ua.{part}")
            for part, conditions in raw_leakage.items()
        }
        self.vital_sign_ranges = self._range_map(mapping.get("vital_sign_ranges"), "vital_sign_ranges")
        self.biosignal_bands_hz = self._range_map(mapping.get("biosignal_bands_hz"), "biosignal_bands_hz")

    def _validate_config(self) -> None:
        total = sum(self.tissue_weighting_factors.values())
        ensure_condition(
            not self.tissue_weighting_factors or abs(total - 1.0) <= 1.0e-6,
            "tissue_weighting_factors must sum to 1.0.",
            config=self.config, error_cls=BaseConfigurationError,
            component=self.component, operation="configuration",
            context={"sum": total},
        )
        ensure_condition(
            bool(self.radiation_weighting_factors),
            "radiation_weighting_factors must not be empty.",
            config=self.config, error_cls=BaseConfigurationError,
            component=self.component, operation="configuration",
        )

    # -- anthropometrics -----------------------------------------------
    def body_surface_area(self, weight_kg: float, height_cm: float, formula: Optional[str] = None) -> float:
        """Body surface area in m^2."""
        w = self._num(weight_kg, "weight_kg", positive=True)
        h = self._num(height_cm, "height_cm", positive=True)
        name = self._choice(formula or self.bsa_formula, tuple(self.BSA_FORMULAS), "formula")
        return self._finish("body_surface_area", self.BSA_FORMULAS[name](w, h),
                            weight_kg=w, height_cm=h, formula=name)

    def body_mass_index(self, weight_kg: float, height_m: float) -> float:
        w = self._num(weight_kg, "weight_kg", positive=True)
        h = self._num(height_m, "height_m", positive=True)
        return self._finish("body_mass_index", w / h ** 2, weight_kg=w, height_m=h)

    # -- hemodynamics ----------------------------------------------------
    def mean_arterial_pressure(self, systolic_mmhg: float, diastolic_mmhg: float) -> float:
        """MAP = DBP + (SBP - DBP) / 3."""
        sbp = self._num(systolic_mmhg, "systolic_mmhg", positive=True)
        dbp = self._num(diastolic_mmhg, "diastolic_mmhg", positive=True)
        self._require(sbp >= dbp, "Systolic pressure must be >= diastolic pressure.",
                      systolic_mmhg=sbp, diastolic_mmhg=dbp)
        return self._finish("mean_arterial_pressure", dbp + (sbp - dbp) / 3.0,
                            systolic_mmhg=sbp, diastolic_mmhg=dbp)

    def cardiac_output(self, heart_rate_bpm: float, stroke_volume_ml: float,
                       body_surface_area_m2: Optional[float] = None) -> Dict[str, Any]:
        """Cardiac output (L/min) and, when BSA is given, cardiac index (L/min/m^2)."""
        hr = self._num(heart_rate_bpm, "heart_rate_bpm", positive=True)
        sv = self._num(stroke_volume_ml, "stroke_volume_ml", positive=True)
        co = hr * sv / 1000.0
        index = None
        if body_surface_area_m2 is not None:
            index = co / self._num(body_surface_area_m2, "body_surface_area_m2", positive=True)
        return self._finish("cardiac_output", {"cardiac_output_l_min": co, "cardiac_index_l_min_m2": index},
                            heart_rate_bpm=hr, stroke_volume_ml=sv, body_surface_area_m2=body_surface_area_m2)

    def systemic_vascular_resistance(self, map_mmhg: float, cvp_mmhg: float, cardiac_output_l_min: float) -> float:
        """SVR in dyn*s/cm^5: 80 * (MAP - CVP) / CO."""
        mean_p = self._num(map_mmhg, "map_mmhg", positive=True)
        venous = self._num(cvp_mmhg, "cvp_mmhg", minimum=0.0)
        co = self._num(cardiac_output_l_min, "cardiac_output_l_min", positive=True)
        self._require(mean_p >= venous, "MAP must be >= central venous pressure.", map_mmhg=mean_p, cvp_mmhg=venous)
        return self._finish("systemic_vascular_resistance", 80.0 * (mean_p - venous) / co,
                            map_mmhg=mean_p, cvp_mmhg=venous, cardiac_output_l_min=co)

    def poiseuille_flow(self, radius_m: float, length_m: float, pressure_drop_pa: float,
                        viscosity_pa_s: Optional[float] = None) -> Dict[str, float]:
        """Laminar Newtonian flow through a rigid tube (Hagen-Poiseuille)."""
        r = self._num(radius_m, "radius_m", positive=True)
        length = self._num(length_m, "length_m", positive=True)
        dp = self._num(pressure_drop_pa, "pressure_drop_pa", positive=True)
        mu = self._num(viscosity_pa_s if viscosity_pa_s is not None else self.blood_viscosity_pa_s,
                       "viscosity_pa_s", positive=True)
        flow = math.pi * r ** 4 * dp / (8.0 * mu * length)
        result = {
            "flow_m3_s": flow,
            "flow_ml_min": flow * 6.0e7,
            "mean_velocity_m_s": flow / (math.pi * r ** 2),
            "resistance_pa_s_m3": 8.0 * mu * length / (math.pi * r ** 4),
            "wall_shear_stress_pa": 4.0 * mu * flow / (math.pi * r ** 3),
        }
        return self._finish("poiseuille_flow", result, radius_m=r, length_m=length,
                            pressure_drop_pa=dp, viscosity_pa_s=mu)

    def windkessel_step(self, pressure_pa: float, inflow_m3_s: float, dt_s: float,
                        peripheral_resistance_pa_s_m3: float, compliance_m3_pa: float) -> float:
        """One explicit-Euler step of the two-element Windkessel model."""
        p = self._num(pressure_pa, "pressure_pa", minimum=0.0)
        q = self._num(inflow_m3_s, "inflow_m3_s", minimum=0.0)
        dt = self._num(dt_s, "dt_s", positive=True)
        r = self._num(peripheral_resistance_pa_s_m3, "peripheral_resistance_pa_s_m3", positive=True)
        c = self._num(compliance_m3_pa, "compliance_m3_pa", positive=True)
        self._require(dt < r * c, "dt_s must be smaller than R*C for a stable explicit step.",
                      dt_s=dt, time_constant_s=r * c)
        return self._finish("windkessel_step", max(0.0, p + dt / c * (q - p / r)),
                            pressure_pa=p, inflow_m3_s=q, dt_s=dt)

    # -- cardiac signals -------------------------------------------------
    def heart_rate_variability(self, rr_intervals_s: Sequence[float]) -> Dict[str, float]:
        """Mean heart rate (bpm), SDNN (ms), and RMSSD (ms) from RR intervals."""
        rr = self._values(rr_intervals_s, "rr_intervals_s", min_len=2, positive=True)
        diffs = [b - a for a, b in zip(rr, rr[1:])]
        result = {
            "heart_rate_bpm": 60.0 / statistics.fmean(rr),
            "sdnn_ms": statistics.stdev(rr) * 1000.0,
            "rmssd_ms": math.sqrt(statistics.fmean(d * d for d in diffs)) * 1000.0,
        }
        return self._finish("heart_rate_variability", result, interval_count=len(rr))

    def corrected_qt(self, qt_ms: float, rr_s: float, formula: Optional[str] = None) -> float:
        """Rate-corrected QT interval (ms)."""
        qt = self._num(qt_ms, "qt_ms", positive=True)
        rr = self._num(rr_s, "rr_s", positive=True)
        name = self._choice(formula or self.qt_formula, tuple(self.QT_FORMULAS), "formula")
        return self._finish("corrected_qt", self.QT_FORMULAS[name](qt, rr), qt_ms=qt, rr_s=rr, formula=name)

    # -- oxygenation and gas exchange -------------------------------------
    def arterial_oxygen_content(self, hemoglobin_g_dl: float, sao2_fraction: float, pao2_mmhg: float = 0.0) -> float:
        """CaO2 (mL O2/dL) = 1.34*Hb*SaO2 + 0.003*PaO2."""
        hb = self._num(hemoglobin_g_dl, "hemoglobin_g_dl", positive=True)
        sat = self._num(sao2_fraction, "sao2_fraction", minimum=0.0, maximum=1.0)
        pao2 = self._num(pao2_mmhg, "pao2_mmhg", minimum=0.0)
        content = (self.hemoglobin_binding_capacity_ml_per_g * hb * sat
                   + self.oxygen_solubility_ml_per_dl_per_mmhg * pao2)
        return self._finish("arterial_oxygen_content", content,
                            hemoglobin_g_dl=hb, sao2_fraction=sat, pao2_mmhg=pao2)

    def oxygen_delivery(self, cardiac_output_l_min: float, cao2_ml_dl: float) -> float:
        """DO2 (mL O2/min) = CO * CaO2 * 10."""
        co = self._num(cardiac_output_l_min, "cardiac_output_l_min", positive=True)
        cao2 = self._num(cao2_ml_dl, "cao2_ml_dl", minimum=0.0)
        return self._finish("oxygen_delivery", co * cao2 * 10.0, cardiac_output_l_min=co, cao2_ml_dl=cao2)

    def alveolar_gas(self, fio2: float, paco2_mmhg: float, pao2_mmhg: Optional[float] = None,
                     atmospheric_pressure_mmhg: Optional[float] = None) -> Dict[str, Optional[float]]:
        """Alveolar gas equation and optional A-a gradient."""
        fi = self._num(fio2, "fio2", minimum=0.0, maximum=1.0)
        pco2 = self._num(paco2_mmhg, "paco2_mmhg", minimum=0.0)
        patm = self._num(atmospheric_pressure_mmhg if atmospheric_pressure_mmhg is not None
                         else self.atmospheric_pressure_mmhg, "atmospheric_pressure_mmhg", positive=True)
        alveolar = fi * (patm - self.water_vapor_pressure_mmhg) - pco2 / self.respiratory_quotient
        gradient = None
        if pao2_mmhg is not None:
            gradient = alveolar - self._num(pao2_mmhg, "pao2_mmhg", minimum=0.0)
        return self._finish("alveolar_gas", {"alveolar_po2_mmhg": alveolar, "a_a_gradient_mmhg": gradient},
                            fio2=fi, paco2_mmhg=pco2, atmospheric_pressure_mmhg=patm)

    def spo2_from_ratio(self, ratio_of_ratios: float) -> float:
        """Empirical pulse-oximetry calibration, clamped to 0-100 %."""
        r = self._num(ratio_of_ratios, "ratio_of_ratios", minimum=0.0)
        spo2 = min(100.0, max(0.0, self.spo2_ratio_intercept - self.spo2_ratio_slope * r))
        return self._finish("spo2_from_ratio", spo2, ratio_of_ratios=r)

    # -- infusion and kinetics ---------------------------------------------
    def infusion_rate(self, volume_ml: float, duration_h: float,
                      drop_factor_gtt_per_ml: Optional[float] = None) -> Dict[str, Optional[float]]:
        """Pump rate (mL/h) and optional gravity drip rate (drops/min) for a set volume and duration."""
        volume = self._num(volume_ml, "volume_ml", positive=True)
        hours = self._num(duration_h, "duration_h", positive=True)
        drops = None
        if drop_factor_gtt_per_ml is not None:
            drops = volume * self._num(drop_factor_gtt_per_ml, "drop_factor_gtt_per_ml", positive=True) / (hours * 60.0)
        return self._finish("infusion_rate", {"rate_ml_h": volume / hours, "drip_rate_gtt_min": drops},
                            notes={"disclaimer": self.DISCLAIMER}, volume_ml=volume, duration_h=hours)

    def pk_one_compartment(self, dose_mg: float, volume_l: float, elimination_rate_per_h: float,
                           time_h: float = 0.0) -> Dict[str, float]:
        """IV-bolus one-compartment kinetics: C(t) = (D/V) exp(-k t)."""
        dose = self._num(dose_mg, "dose_mg", positive=True)
        vd = self._num(volume_l, "volume_l", positive=True)
        ke = self._num(elimination_rate_per_h, "elimination_rate_per_h", positive=True)
        t = self._num(time_h, "time_h", minimum=0.0)
        c0 = dose / vd
        result = {
            "initial_concentration_mg_l": c0,
            "concentration_mg_l": c0 * math.exp(-ke * t),
            "half_life_h": math.log(2.0) / ke,
            "clearance_l_h": ke * vd,
        }
        return self._finish("pk_one_compartment", result, notes={"disclaimer": self.DISCLAIMER},
                            dose_mg=dose, volume_l=vd, elimination_rate_per_h=ke, time_h=t)

    def steady_state_concentration(self, infusion_rate_mg_h: float, clearance_l_h: float) -> float:
        """Css = R0 / CL for a constant-rate infusion."""
        rate = self._num(infusion_rate_mg_h, "infusion_rate_mg_h", positive=True)
        cl = self._num(clearance_l_h, "clearance_l_h", positive=True)
        return self._finish("steady_state_concentration", rate / cl,
                            notes={"disclaimer": self.DISCLAIMER}, infusion_rate_mg_h=rate, clearance_l_h=cl)

    # -- radiation ---------------------------------------------------------
    def equivalent_dose(self, absorbed_dose_gy: float, radiation_type: Optional[str] = None,
                        weighting_factor: Optional[float] = None) -> float:
        """Equivalent dose (Sv) = w_R * absorbed dose; pass ``weighting_factor`` for energy-dependent types."""
        dose = self._num(absorbed_dose_gy, "absorbed_dose_gy", minimum=0.0)
        if weighting_factor is not None:
            w_r = self._num(weighting_factor, "weighting_factor", positive=True)
            label = "custom"
        else:
            self._require(radiation_type is not None, "Provide 'radiation_type' or 'weighting_factor'.")
            label = self._choice(radiation_type, tuple(self.radiation_weighting_factors), "radiation_type")
            w_r = self.radiation_weighting_factors[label]
        return self._finish("equivalent_dose", w_r * dose, absorbed_dose_gy=dose, radiation=label, weighting_factor=w_r)

    def effective_dose(self, equivalent_doses_sv: Mapping[str, float]) -> Dict[str, float]:
        """Effective dose (Sv) = sum(w_T * H_T) over the supplied tissues."""
        ensure_mapping(equivalent_doses_sv, "equivalent_doses_sv", config=self.config,
                       error_cls=BaseValidationError, component=self.component, operation="input_validation")
        total = covered = 0.0
        for tissue, dose in equivalent_doses_sv.items():
            key = self._choice(tissue, tuple(self.tissue_weighting_factors), "tissue")
            total += self.tissue_weighting_factors[key] * self._num(dose, f"equivalent_doses_sv.{tissue}", minimum=0.0)
            covered += self.tissue_weighting_factors[key]
        return self._finish("effective_dose", {"effective_dose_sv": total, "covered_weight_fraction": covered},
                            tissues=list(equivalent_doses_sv))

    # -- device safety -------------------------------------------------------
    def check_leakage_current(self, current_ua: float, applied_part: str, condition: str = "normal") -> EngineeringCheck:
        """Patient leakage current against the configured IEC 60601-1 style limit."""
        part = self._choice(applied_part, tuple(self.leakage_current_limits_ua), "applied_part")
        cond = self._choice(condition, tuple(self.leakage_current_limits_ua[part]), "condition")
        return self.evaluate_limit(f"leakage_current_{part}_{cond}", current_ua,
                                   self.leakage_current_limits_ua[part][cond], mode="max", unit="uA")

    def check_vital_sign(self, name: str, value: float) -> EngineeringCheck:
        """Reading against the configured adult resting reference range."""
        key = self._choice(name, tuple(self.vital_sign_ranges), "vital_sign")
        return self.evaluate_limit(f"vital_sign_{key}", value, self.vital_sign_ranges[key], mode="range")

    def biosignal_sampling_rate(self, signal: str) -> Dict[str, Any]:
        """Minimum practical sampling rate for a configured biosignal band."""
        key = self._choice(signal, tuple(self.biosignal_bands_hz), "signal")
        band = self.biosignal_bands_hz[key]
        return self._finish("biosignal_sampling_rate",
                            {"band_hz": list(band), "minimum_sampling_rate_hz": self._sampling_rate(band[1])},
                            signal=key)


# ======================================================================
# Civil engineering
# ======================================================================
class CivilEngineering(_StructuralMixin, _DomainBase):
    """Structural, geotechnical, hydraulic, and transportation calculations."""

    DOMAIN = "civil"

    # (support, load_type) -> (max moment, max shear, max deflection)
    BEAM_CASES: Dict[Tuple[str, str], Tuple[Callable[..., float], Callable[..., float], Callable[..., float]]] = {
        ("simply_supported", "udl"): (
            lambda w, L: w * L ** 2 / 8.0, lambda w, L: w * L / 2.0,
            lambda w, L, ei: 5.0 * w * L ** 4 / (384.0 * ei)),
        ("simply_supported", "point_center"): (
            lambda p, L: p * L / 4.0, lambda p, L: p / 2.0,
            lambda p, L, ei: p * L ** 3 / (48.0 * ei)),
        ("cantilever", "udl"): (
            lambda w, L: w * L ** 2 / 2.0, lambda w, L: w * L,
            lambda w, L, ei: w * L ** 4 / (8.0 * ei)),
        ("cantilever", "point_end"): (
            lambda p, L: p * L, lambda p, L: p,
            lambda p, L, ei: p * L ** 3 / (3.0 * ei)),
        ("fixed_fixed", "udl"): (
            lambda w, L: w * L ** 2 / 12.0, lambda w, L: w * L / 2.0,
            lambda w, L, ei: w * L ** 4 / (384.0 * ei)),
    }

    CONFIG_FIELDS = (
        "dead_load_factor", "live_load_factor", "deflection_limit_ratio",
        "concrete_modulus_coefficient", "flexural_resistance_factor", "rebar_yield_strength_pa",
        "concrete_stress_block_factor", "tension_controlled_c_over_d", "bearing_capacity_safety_factor",
        "soil_unit_weight_n_m3", "seismic_importance_factor", "seismic_response_modification",
        "seismic_min_cs_sds_coefficient", "seismic_min_cs", "runoff_coefficient",
        "manning_roughness", "effective_length_factor",
    )

    def _configure(self, mapping: Mapping[str, Any]) -> None:
        self.dead_load_factor = coerce_float(mapping.get("dead_load_factor", 1.2), 1.2, minimum=1.0)
        self.live_load_factor = coerce_float(mapping.get("live_load_factor", 1.6), 1.6, minimum=1.0)
        self.deflection_limit_ratio = coerce_float(mapping.get("deflection_limit_ratio", 360.0), 360.0, minimum=1.0)
        self.concrete_modulus_coefficient = coerce_float(mapping.get("concrete_modulus_coefficient", 4700.0), 4700.0, minimum=1.0)
        self.flexural_resistance_factor = coerce_float(mapping.get("flexural_resistance_factor", 0.9), 0.9, minimum=0.1, maximum=1.0)
        self.rebar_yield_strength_pa = coerce_float(mapping.get("rebar_yield_strength_pa", 4.2e8), 4.2e8, minimum=1.0)
        self.concrete_stress_block_factor = coerce_float(mapping.get("concrete_stress_block_factor", 0.85), 0.85, minimum=0.1, maximum=1.0)
        self.tension_controlled_c_over_d = coerce_float(mapping.get("tension_controlled_c_over_d", 0.375), 0.375, minimum=0.01, maximum=1.0)
        self.bearing_capacity_safety_factor = coerce_float(mapping.get("bearing_capacity_safety_factor", 3.0), 3.0, minimum=1.0)
        self.soil_unit_weight_n_m3 = coerce_float(mapping.get("soil_unit_weight_n_m3", 18000.0), 18000.0, minimum=0.0)
        self.seismic_importance_factor = coerce_float(mapping.get("seismic_importance_factor", 1.0), 1.0, minimum=0.5)
        self.seismic_response_modification = coerce_float(mapping.get("seismic_response_modification", 3.0), 3.0, minimum=1.0)
        self.seismic_min_cs_sds_coefficient = coerce_float(mapping.get("seismic_min_cs_sds_coefficient", 0.044), 0.044, minimum=0.0)
        self.seismic_min_cs = coerce_float(mapping.get("seismic_min_cs", 0.01), 0.01, minimum=0.0)
        self.runoff_coefficient = coerce_float(mapping.get("runoff_coefficient", 0.7), 0.7, minimum=0.0, maximum=1.0)
        self.manning_roughness = coerce_float(mapping.get("manning_roughness", 0.013), 0.013, minimum=1.0e-6)
        self.effective_length_factor = coerce_float(mapping.get("effective_length_factor", 1.0), 1.0, minimum=0.5)

    def _validate_config(self) -> None:
        ensure_numeric_range(self.flexural_resistance_factor, "flexural_resistance_factor",
                             minimum=0.1, maximum=1.0, config=self.config, error_cls=BaseConfigurationError)

    # -- loads -------------------------------------------------------------
    def factored_load(self, dead_load: float, live_load: float,
                      dead_factor: Optional[float] = None, live_factor: Optional[float] = None) -> float:
        """LRFD combination: gamma_D * D + gamma_L * L."""
        d = self._num(dead_load, "dead_load", minimum=0.0)
        l = self._num(live_load, "live_load", minimum=0.0)
        gd = self._num(dead_factor if dead_factor is not None else self.dead_load_factor, "dead_factor", positive=True)
        gl = self._num(live_factor if live_factor is not None else self.live_load_factor, "live_factor", positive=True)
        return self._finish("factored_load", gd * d + gl * l, dead_load=d, live_load=l, dead_factor=gd, live_factor=gl)

    def seismic_base_shear(self, seismic_weight_n: float, sds_g: float,
                           response_modification: Optional[float] = None,
                           importance_factor: Optional[float] = None) -> Dict[str, float]:
        """Equivalent-lateral-force base shear V = Cs * W with the minimum-Cs floors."""
        w = self._num(seismic_weight_n, "seismic_weight_n", positive=True)
        sds = self._num(sds_g, "sds_g", positive=True)
        r = self._num(response_modification if response_modification is not None
                      else self.seismic_response_modification, "response_modification", minimum=1.0)
        ie = self._num(importance_factor if importance_factor is not None
                       else self.seismic_importance_factor, "importance_factor", positive=True)
        cs_floor = max(self.seismic_min_cs_sds_coefficient * sds * ie, self.seismic_min_cs)
        cs = max(sds / (r / ie), cs_floor)
        return self._finish("seismic_base_shear", {"cs": cs, "base_shear_n": cs * w},
                            seismic_weight_n=w, sds_g=sds, response_modification=r, importance_factor=ie)

    # -- members -------------------------------------------------------------
    def beam_response(self, span_m: float, load: float, second_moment_m4: float, *,
                      support: str = "simply_supported", load_type: str = "udl",
                      youngs_modulus_pa: Optional[float] = None, material: Optional[str] = None) -> Dict[str, Any]:
        """Peak moment, shear, and deflection; ``load`` is N/m for 'udl' and N for point loads."""
        length = self._num(span_m, "span_m", positive=True)
        p = self._num(load, "load", positive=True)
        inertia = self._num(second_moment_m4, "second_moment_m4", positive=True)
        modulus = self._num(youngs_modulus_pa if youngs_modulus_pa is not None
                            else self.material(material)["youngs_modulus"], "youngs_modulus_pa", positive=True)
        supports = tuple(dict.fromkeys(s for s, _ in self.BEAM_CASES))
        load_types = tuple(dict.fromkeys(t for _, t in self.BEAM_CASES))
        case = (self._choice(support, supports, "support"),
                self._choice(load_type, load_types, "load_type"))
        self._require(case in self.BEAM_CASES, "Unsupported support/load_type combination.",
                      allowed=[f"{s}:{t}" for s, t in self.BEAM_CASES], received=f"{case[0]}:{case[1]}")
        moment_fn, shear_fn, deflection_fn = self.BEAM_CASES[case]
        deflection = deflection_fn(p, length, modulus * inertia)
        check = self.evaluate_limit("beam_deflection", deflection, length / self.deflection_limit_ratio, mode="max")
        result = {
            "max_moment_nm": moment_fn(p, length),
            "max_shear_n": shear_fn(p, length),
            "max_deflection_m": deflection,
            "deflection_limit_m": length / self.deflection_limit_ratio,
            "deflection_check": check.to_dict(),
        }
        return self._finish("beam_response", result, span_m=length, load=p, support=case[0], load_type=case[1])

    def euler_buckling(self, second_moment_m4: float, length_m: float, *, youngs_modulus_pa: Optional[float] = None,
                       material: Optional[str] = None, effective_length_factor: Optional[float] = None,
                       area_m2: Optional[float] = None) -> Dict[str, Optional[float]]:
        """Critical column load P_cr = pi^2 E I / (K L)^2, plus slenderness if area is supplied."""
        inertia = self._num(second_moment_m4, "second_moment_m4", positive=True)
        length = self._num(length_m, "length_m", positive=True)
        modulus = self._num(youngs_modulus_pa if youngs_modulus_pa is not None
                            else self.material(material)["youngs_modulus"], "youngs_modulus_pa", positive=True)
        k = self._num(effective_length_factor if effective_length_factor is not None
                      else self.effective_length_factor, "effective_length_factor", positive=True)
        pcr = math.pi ** 2 * modulus * inertia / (k * length) ** 2
        slenderness = critical_stress = None
        if area_m2 is not None:
            area = self._num(area_m2, "area_m2", positive=True)
            slenderness = k * length / math.sqrt(inertia / area)
            critical_stress = pcr / area
        return self._finish("euler_buckling",
                            {"critical_load_n": pcr, "slenderness_ratio": slenderness, "critical_stress_pa": critical_stress},
                            second_moment_m4=inertia, length_m=length, effective_length_factor=k)

    def concrete_modulus(self, concrete_strength_pa: float) -> float:
        """Ec = coefficient * sqrt(f'c [MPa]) MPa, returned in Pa."""
        fc = self._num(concrete_strength_pa, "concrete_strength_pa", positive=True)
        return self._finish("concrete_modulus", self.concrete_modulus_coefficient * math.sqrt(fc / 1.0e6) * 1.0e6,
                            concrete_strength_pa=fc)

    def rc_beam_flexural_capacity(self, width_m: float, effective_depth_m: float, steel_area_m2: float,
                                  concrete_strength_pa: float, steel_yield_pa: Optional[float] = None) -> Dict[str, Any]:
        """Singly reinforced rectangular section (rectangular stress block, steel assumed yielding)."""
        b = self._num(width_m, "width_m", positive=True)
        d = self._num(effective_depth_m, "effective_depth_m", positive=True)
        a_s = self._num(steel_area_m2, "steel_area_m2", positive=True)
        fc = self._num(concrete_strength_pa, "concrete_strength_pa", positive=True)
        fy = self._num(steel_yield_pa if steel_yield_pa is not None else self.rebar_yield_strength_pa,
                       "steel_yield_pa", positive=True)
        depth = a_s * fy / (self.concrete_stress_block_factor * fc * b)
        self._require(depth < d, "Stress-block depth exceeds the effective depth; section is over-reinforced.",
                      stress_block_depth_m=depth, effective_depth_m=d)
        fc_mpa = fc / 1.0e6
        beta1 = 0.85 if fc_mpa <= 28.0 else max(0.65, 0.85 - 0.05 * (fc_mpa - 28.0) / 7.0)
        nominal = a_s * fy * (d - depth / 2.0)
        c_over_d = depth / beta1 / d
        result = {
            "stress_block_depth_m": depth,
            "nominal_moment_nm": nominal,
            "design_moment_nm": self.flexural_resistance_factor * nominal,
            "reinforcement_ratio": a_s / (b * d),
            "c_over_d": c_over_d,
            "tension_controlled": c_over_d <= self.tension_controlled_c_over_d,
        }
        return self._finish("rc_beam_flexural_capacity", result, width_m=b, effective_depth_m=d,
                            steel_area_m2=a_s, concrete_strength_pa=fc, steel_yield_pa=fy)

    # -- geotechnical ---------------------------------------------------------
    def bearing_capacity(self, width_m: float, embedment_depth_m: float, cohesion_pa: float,
                         friction_angle_deg: float, unit_weight_n_m3: Optional[float] = None,
                         safety_factor: Optional[float] = None) -> Dict[str, float]:
        """Strip-footing general bearing-capacity equation (Meyerhof N-factors, no shape/depth factors)."""
        b = self._num(width_m, "width_m", positive=True)
        df = self._num(embedment_depth_m, "embedment_depth_m", minimum=0.0)
        c = self._num(cohesion_pa, "cohesion_pa", minimum=0.0)
        phi = math.radians(self._num(friction_angle_deg, "friction_angle_deg", minimum=0.0, maximum=45.0))
        gamma = self._num(unit_weight_n_m3 if unit_weight_n_m3 is not None else self.soil_unit_weight_n_m3,
                          "unit_weight_n_m3", positive=True)
        fs = self._num(safety_factor if safety_factor is not None else self.bearing_capacity_safety_factor,
                       "safety_factor", minimum=1.0)
        nq = math.exp(math.pi * math.tan(phi)) * math.tan(math.pi / 4.0 + phi / 2.0) ** 2
        nc = (nq - 1.0) / math.tan(phi) if phi > 0.0 else math.pi + 2.0
        n_gamma = (nq - 1.0) * math.tan(1.4 * phi)
        q_ult = c * nc + gamma * df * nq + 0.5 * gamma * b * n_gamma
        result = {"nc": nc, "nq": nq, "n_gamma": n_gamma, "ultimate_pa": q_ult, "allowable_pa": q_ult / fs}
        return self._finish("bearing_capacity", result, width_m=b, embedment_depth_m=df, cohesion_pa=c,
                            friction_angle_deg=math.degrees(phi), unit_weight_n_m3=gamma, safety_factor=fs)

    # -- hydrology / hydraulics -------------------------------------------------
    def rational_method_peak_flow(self, rainfall_intensity_mm_h: float, catchment_area_ha: float,
                                  runoff_coefficient: Optional[float] = None) -> float:
        """Peak runoff Q (m^3/s) = C * i * A / 360."""
        i = self._num(rainfall_intensity_mm_h, "rainfall_intensity_mm_h", positive=True)
        area = self._num(catchment_area_ha, "catchment_area_ha", positive=True)
        c = self._num(runoff_coefficient if runoff_coefficient is not None else self.runoff_coefficient,
                      "runoff_coefficient", minimum=0.0, maximum=1.0)
        return self._finish("rational_method_peak_flow", c * i * area / 360.0,
                            rainfall_intensity_mm_h=i, catchment_area_ha=area, runoff_coefficient=c)

    def manning_flow(self, flow_area_m2: float, wetted_perimeter_m: float, slope: float,
                     roughness: Optional[float] = None) -> Dict[str, float]:
        """Uniform open-channel or part-full pipe flow via Manning's equation."""
        area = self._num(flow_area_m2, "flow_area_m2", positive=True)
        perimeter = self._num(wetted_perimeter_m, "wetted_perimeter_m", positive=True)
        s = self._num(slope, "slope", positive=True)
        n = self._num(roughness if roughness is not None else self.manning_roughness, "roughness", positive=True)
        radius = area / perimeter
        velocity = radius ** (2.0 / 3.0) * math.sqrt(s) / n
        return self._finish("manning_flow",
                            {"hydraulic_radius_m": radius, "velocity_m_s": velocity, "discharge_m3_s": velocity * area},
                            flow_area_m2=area, wetted_perimeter_m=perimeter, slope=s, roughness=n)

    # -- transportation -----------------------------------------------------------
    def greenshields_flow(self, density_veh_km: float, free_flow_speed_kmh: float,
                          jam_density_veh_km: float) -> Dict[str, float]:
        """Greenshields linear speed-density model."""
        k = self._num(density_veh_km, "density_veh_km", minimum=0.0)
        vf = self._num(free_flow_speed_kmh, "free_flow_speed_kmh", positive=True)
        kj = self._num(jam_density_veh_km, "jam_density_veh_km", positive=True)
        self._require(k <= kj, "Density cannot exceed jam density.", density_veh_km=k, jam_density_veh_km=kj)
        speed = vf * (1.0 - k / kj)
        return self._finish("greenshields_flow",
                            {"speed_kmh": speed, "flow_veh_h": k * speed, "capacity_veh_h": vf * kj / 4.0,
                             "optimal_density_veh_km": kj / 2.0},
                            density_veh_km=k, free_flow_speed_kmh=vf, jam_density_veh_km=kj)


# ======================================================================
# Computer engineering
# ======================================================================
class ComputerEngineering(_DomainBase):
    """Architecture, performance, power, queueing, and reliability calculations."""

    DOMAIN = "computer"

    # level -> (minimum disks, usable-capacity fn(n, c), guaranteed fault tolerance fn(n))
    RAID_LEVELS: Dict[str, Tuple[int, Callable[[int, float], float], Callable[[int], int]]] = {
        "0": (2, lambda n, c: n * c, lambda n: 0),
        "1": (2, lambda n, c: c, lambda n: n - 1),
        "5": (3, lambda n, c: (n - 1) * c, lambda n: 1),
        "6": (4, lambda n, c: (n - 2) * c, lambda n: 2),
        "10": (4, lambda n, c: (n // 2) * c, lambda n: 1),
    }

    CONFIG_FIELDS = ("max_cpu_utilization", "availability_target", "dynamic_power_activity_factor")

    def _configure(self, mapping: Mapping[str, Any]) -> None:
        self.max_cpu_utilization = coerce_float(mapping.get("max_cpu_utilization", 0.8), 0.8, minimum=0.01, maximum=1.0)
        self.availability_target = coerce_float(mapping.get("availability_target", 0.999), 0.999, minimum=0.0, maximum=1.0)
        self.dynamic_power_activity_factor = coerce_float(mapping.get("dynamic_power_activity_factor", 0.1), 0.1, minimum=0.0, maximum=1.0)

    def cpu_performance(self, instruction_count: float, cpi: float, clock_hz: float) -> Dict[str, float]:
        """CPU time = IC * CPI / f, with MIPS."""
        ic = self._num(instruction_count, "instruction_count", positive=True)
        cycles = self._num(cpi, "cpi", positive=True)
        f = self._num(clock_hz, "clock_hz", positive=True)
        t = ic * cycles / f
        return self._finish("cpu_performance", {"cpu_time_s": t, "mips": ic / (t * 1.0e6)},
                            instruction_count=ic, cpi=cycles, clock_hz=f)

    def parallel_speedup(self, parallel_fraction: float, processors: int) -> Dict[str, float]:
        """Amdahl (fixed-size) and Gustafson (scaled) speedup with parallel efficiency."""
        p = self._num(parallel_fraction, "parallel_fraction", minimum=0.0, maximum=1.0)
        n = self._count(processors, "processors")
        amdahl = 1.0 / ((1.0 - p) + p / n)
        result = {
            "amdahl_speedup": amdahl,
            "amdahl_limit": None if p >= 1.0 else 1.0 / (1.0 - p),  # None = unbounded
            "gustafson_speedup": (1.0 - p) + p * n,
            "parallel_efficiency": amdahl / n,
        }
        return self._finish("parallel_speedup", result, parallel_fraction=p, processors=n)

    def amat(self, cache_levels: Sequence[Sequence[float]], memory_access_time_s: float) -> float:
        """Average memory access time; ``cache_levels`` is [(hit_time_s, miss_rate), ...] from L1 outward."""
        levels = ensure_list(cache_levels)
        self._require(len(levels) >= 1, "At least one cache level is required.")
        time = self._num(memory_access_time_s, "memory_access_time_s", positive=True)
        parsed: List[Tuple[float, float]] = []
        for index, level in enumerate(levels):
            pair = ensure_list(level)
            self._require(len(pair) == 2, f"cache_levels[{index}] must be (hit_time_s, miss_rate).", index=index)
            parsed.append((self._num(pair[0], f"cache_levels[{index}].hit_time_s", positive=True),
                           self._num(pair[1], f"cache_levels[{index}].miss_rate", minimum=0.0, maximum=1.0)))
        for hit_time, miss_rate in reversed(parsed):
            time = hit_time + miss_rate * time
        return self._finish("amat", time, levels=parsed, memory_access_time_s=memory_access_time_s)

    def cache_address_breakdown(self, address_bits: int, cache_size_bytes: int, block_size_bytes: int,
                                associativity: int = 1) -> Dict[str, int]:
        """Tag/index/offset split for a set-associative cache (sizes must be powers of two)."""
        bits = self._count(address_bits, "address_bits")
        size = self._count(cache_size_bytes, "cache_size_bytes")
        block = self._count(block_size_bytes, "block_size_bytes")
        ways = self._count(associativity, "associativity")
        self._require(all(_is_power_of_two(v) for v in (size, block, ways)),
                      "cache_size_bytes, block_size_bytes and associativity must be powers of two.",
                      cache_size_bytes=size, block_size_bytes=block, associativity=ways)
        sets = size // (block * ways)
        self._require(sets >= 1, "Cache is too small for the requested block size and associativity.", sets=sets)
        offset_bits, index_bits = int(math.log2(block)), int(math.log2(sets))
        tag_bits = bits - index_bits - offset_bits
        self._require(tag_bits > 0, "address_bits is too small for this cache geometry.", tag_bits=tag_bits)
        return self._finish("cache_address_breakdown",
                            {"sets": sets, "offset_bits": offset_bits, "index_bits": index_bits, "tag_bits": tag_bits},
                            address_bits=bits, cache_size_bytes=size, block_size_bytes=block, associativity=ways)

    def dynamic_power(self, capacitance_f: float, voltage_v: float, frequency_hz: float,
                      activity_factor: Optional[float] = None) -> float:
        """CMOS switching power P = alpha * C * V^2 * f."""
        alpha = self._num(activity_factor if activity_factor is not None else self.dynamic_power_activity_factor,
                          "activity_factor", minimum=0.0, maximum=1.0)
        c = self._num(capacitance_f, "capacitance_f", positive=True)
        v = self._num(voltage_v, "voltage_v", positive=True)
        f = self._num(frequency_hz, "frequency_hz", positive=True)
        return self._finish("dynamic_power", alpha * c * v ** 2 * f,
                            capacitance_f=c, voltage_v=v, frequency_hz=f, activity_factor=alpha)

    def pipeline_speedup(self, stages: int, stall_cycles_per_instruction: float = 0.0) -> Dict[str, float]:
        """Ideal pipeline speedup over an unpipelined design, reduced by stalls."""
        depth = self._count(stages, "stages")
        stalls = self._num(stall_cycles_per_instruction, "stall_cycles_per_instruction", minimum=0.0)
        cpi = 1.0 + stalls
        return self._finish("pipeline_speedup", {"speedup": depth / cpi, "effective_cpi": cpi},
                            stages=depth, stall_cycles_per_instruction=stalls)

    def roofline(self, peak_flops: float, memory_bandwidth_bytes_s: float,
                 arithmetic_intensity_flop_per_byte: float) -> Dict[str, Any]:
        """Roofline-attainable performance and the limiting resource."""
        peak = self._num(peak_flops, "peak_flops", positive=True)
        bandwidth = self._num(memory_bandwidth_bytes_s, "memory_bandwidth_bytes_s", positive=True)
        intensity = self._num(arithmetic_intensity_flop_per_byte, "arithmetic_intensity_flop_per_byte", positive=True)
        ridge = peak / bandwidth
        return self._finish("roofline",
                            {"attainable_flops": min(peak, bandwidth * intensity), "ridge_point_flop_per_byte": ridge,
                             "bound": "memory" if intensity < ridge else "compute"},
                            peak_flops=peak, memory_bandwidth_bytes_s=bandwidth, arithmetic_intensity=intensity)

    def queue_mm1(self, arrival_rate: float, service_rate: float) -> Dict[str, Any]:
        """M/M/1 steady-state metrics; utilization is checked against ``max_cpu_utilization``."""
        lam = self._num(arrival_rate, "arrival_rate", positive=True)
        mu = self._num(service_rate, "service_rate", positive=True)
        rho = lam / mu
        self._require(rho < 1.0, "Queue is unstable: arrival_rate must be < service_rate.",
                      error_cls=BaseStateError, utilization=rho)
        check = self.evaluate_limit("queue_utilization", rho, self.max_cpu_utilization, mode="max")
        result = {
            "utilization": rho,
            "mean_in_system": rho / (1.0 - rho),
            "mean_in_queue": rho ** 2 / (1.0 - rho),
            "mean_time_in_system_s": 1.0 / (mu - lam),
            "mean_wait_s": rho / (mu - lam),
            "utilization_check": check.to_dict(),
        }
        return self._finish("queue_mm1", result, arrival_rate=lam, service_rate=mu)

    def availability(self, mtbf_h: float, mttr_h: float) -> Dict[str, Any]:
        """Steady-state availability A = MTBF / (MTBF + MTTR) checked against the target."""
        mtbf = self._num(mtbf_h, "mtbf_h", positive=True)
        mttr = self._num(mttr_h, "mttr_h", minimum=0.0)
        value = mtbf / (mtbf + mttr)
        check = self.evaluate_limit("availability", value, self.availability_target, mode="min")
        hours_per_year = self.constants["year"] / 3600.0
        return self._finish("availability",
                            {"availability": value, "downtime_h_per_year": (1.0 - value) * hours_per_year,
                             "availability_check": check.to_dict()},
                            mtbf_h=mtbf, mttr_h=mttr)

    def system_reliability(self, component_reliabilities: Sequence[float], topology: str = "series") -> float:
        """Series or parallel combination of component reliabilities (or availabilities)."""
        values = self._values(component_reliabilities, "component_reliabilities")
        for index, value in enumerate(values):
            self._require(0.0 <= value <= 1.0, f"component_reliabilities[{index}] must be within [0, 1].", index=index, value=value)
        mode = self._choice(topology, ("series", "parallel"), "topology")
        if mode == "series":
            combined = math.prod(values)
        else:
            combined = 1.0 - math.prod(1.0 - v for v in values)
        return self._finish("system_reliability", combined, count=len(values), topology=mode)

    def raid_profile(self, level: Any, disks: int, disk_capacity_bytes: float) -> Dict[str, Any]:
        """Usable capacity and guaranteed fault tolerance for a RAID level."""
        key = self._choice(str(level).lower().replace("raid", ""), tuple(self.RAID_LEVELS), "level")
        n = self._count(disks, "disks")
        capacity = self._num(disk_capacity_bytes, "disk_capacity_bytes", positive=True)
        minimum, usable_fn, tolerance_fn = self.RAID_LEVELS[key]
        self._require(n >= minimum, f"RAID {key} requires at least {minimum} disks.", disks=n, minimum=minimum)
        self._require(key != "10" or n % 2 == 0, "RAID 10 requires an even number of disks.", disks=n)
        usable = usable_fn(n, capacity)
        return self._finish("raid_profile",
                            {"usable_bytes": usable, "efficiency": usable / (n * capacity),
                             "guaranteed_fault_tolerance": tolerance_fn(n)},
                            level=key, disks=n, disk_capacity_bytes=capacity)


# ======================================================================
# Electronics and communications engineering
# ======================================================================
class ElectronicsCommsEngineering(_DomainBase):
    """Circuit, signal-chain, and radio/link calculations."""

    DOMAIN = "electronics_comms"

    CONFIG_FIELDS = (
        "resistor_power_derating", "max_junction_temperature_c", "battery_derating_factor",
        "copper_resistivity_ohm_m", "copper_temp_coefficient_per_c", "reference_conductor_temperature_c",
        "noise_temperature_k", "antenna_efficiency", "required_link_margin_db",
    )

    def _configure(self, mapping: Mapping[str, Any]) -> None:
        self.resistor_power_derating = coerce_float(mapping.get("resistor_power_derating", 0.5), 0.5, minimum=0.01, maximum=1.0)
        self.max_junction_temperature_c = coerce_float(mapping.get("max_junction_temperature_c", 125.0), 125.0)
        self.battery_derating_factor = coerce_float(mapping.get("battery_derating_factor", 0.8), 0.8, minimum=0.01, maximum=1.0)
        self.copper_resistivity_ohm_m = coerce_float(mapping.get("copper_resistivity_ohm_m", 1.68e-8), 1.68e-8, minimum=1.0e-12)
        self.copper_temp_coefficient_per_c = coerce_float(mapping.get("copper_temp_coefficient_per_c", 0.00393), 0.00393)
        self.reference_conductor_temperature_c = coerce_float(mapping.get("reference_conductor_temperature_c", 20.0), 20.0)
        self.noise_temperature_k = coerce_float(mapping.get("noise_temperature_k", 290.0), 290.0, minimum=1.0)
        self.antenna_efficiency = coerce_float(mapping.get("antenna_efficiency", 0.55), 0.55, minimum=0.01, maximum=1.0)
        self.required_link_margin_db = coerce_float(mapping.get("required_link_margin_db", 6.0), 6.0)

    # -- pure helpers shared by several public calculations -------------------
    @staticmethod
    def _db(ratio: float, kind: str) -> float:
        return (10.0 if kind == "power" else 20.0) * math.log10(ratio)

    def _fspl_db(self, distance_m: float, frequency_hz: float) -> float:
        return self._db(4.0 * math.pi * distance_m * frequency_hz / self.constants["c"], "voltage")

    def _noise_floor_dbm(self, bandwidth_hz: float, noise_figure_db: float, temperature_k: float) -> float:
        return self._db(self.constants["kB"] * temperature_k * bandwidth_hz / 1.0e-3, "power") + noise_figure_db

    # ======================= electronics ======================================
    def ohms_law(self, voltage_v: Optional[float] = None, current_a: Optional[float] = None,
                 resistance_ohm: Optional[float] = None) -> Dict[str, float]:
        """Solve V = I R from exactly two known quantities and report power."""
        known = {"voltage_v": voltage_v, "current_a": current_a, "resistance_ohm": resistance_ohm}
        given = {k: v for k, v in known.items() if v is not None}
        self._require(len(given) == 2, "Provide exactly two of voltage_v, current_a, resistance_ohm.", provided=list(given))
        v, i, r = (self._num(known[k], k) if known[k] is not None else None for k in known)
        if r is not None:
            self._require(r > 0.0, "'resistance_ohm' must be > 0.", resistance_ohm=r)
        if v is None:
            assert i is not None and r is not None
            v = i * r
        elif i is None:
            assert v is not None and r is not None
            i = v / r
        else:
            self._require(abs(i) > self.shared.numeric_tolerance, "'current_a' must be non-zero.", current_a=i)
            r = v / i
        return self._finish("ohms_law", {"voltage_v": v, "current_a": i, "resistance_ohm": r, "power_w": v * i}, **given)

    def combine_elements(self, values: Sequence[float], topology: str, kind: str = "resistor") -> float:
        """Equivalent R, L, or C of a series/parallel network."""
        items = self._values(values, "values", positive=True)
        mode = self._choice(topology, ("series", "parallel"), "topology")
        element = self._choice(kind, ("resistor", "inductor", "capacitor"), "kind")
        additive = (mode == "series") != (element == "capacitor")
        total = sum(items) if additive else 1.0 / sum(1.0 / v for v in items)
        return self._finish("combine_elements", total, count=len(items), topology=mode, kind=element)

    def voltage_divider(self, vin_v: float, r_top_ohm: float, r_bottom_ohm: float,
                        r_load_ohm: Optional[float] = None) -> float:
        """Divider output, optionally loaded in parallel with the bottom resistor."""
        vin = self._num(vin_v, "vin_v")
        top = self._num(r_top_ohm, "r_top_ohm", positive=True)
        bottom = self._num(r_bottom_ohm, "r_bottom_ohm", positive=True)
        if r_load_ohm is not None:
            load = self._num(r_load_ohm, "r_load_ohm", positive=True)
            bottom = bottom * load / (bottom + load)
        return self._finish("voltage_divider", vin * bottom / (top + bottom), vin_v=vin, r_top_ohm=top,
                            r_bottom_ohm=r_bottom_ohm, r_load_ohm=r_load_ohm)

    def rc_response(self, resistance_ohm: float, capacitance_f: float, time_s: Optional[float] = None,
                    v_initial: float = 0.0, v_final: float = 1.0) -> Dict[str, Optional[float]]:
        """RC time constant, cutoff, 1 % settling time, and step response at ``time_s``."""
        r = self._num(resistance_ohm, "resistance_ohm", positive=True)
        c = self._num(capacitance_f, "capacitance_f", positive=True)
        tau = r * c
        v0, v1 = self._num(v_initial, "v_initial"), self._num(v_final, "v_final")
        voltage = None
        if time_s is not None:
            voltage = v1 + (v0 - v1) * math.exp(-self._num(time_s, "time_s", minimum=0.0) / tau)
        return self._finish("rc_response",
                            {"tau_s": tau, "cutoff_hz": 1.0 / (2.0 * math.pi * tau),
                             "settling_time_1pct_s": tau * math.log(100.0), "voltage_v": voltage},
                            resistance_ohm=r, capacitance_f=c, time_s=time_s)

    def resonance(self, inductance_h: float, capacitance_f: float,
                  series_resistance_ohm: Optional[float] = None) -> Dict[str, Optional[float]]:
        """LC resonance, characteristic impedance, and series-RLC Q and bandwidth."""
        l = self._num(inductance_h, "inductance_h", positive=True)
        c = self._num(capacitance_f, "capacitance_f", positive=True)
        f0 = 1.0 / (2.0 * math.pi * math.sqrt(l * c))
        z0 = math.sqrt(l / c)
        q = bandwidth = None
        if series_resistance_ohm is not None:
            q = z0 / self._num(series_resistance_ohm, "series_resistance_ohm", positive=True)
            bandwidth = f0 / q
        return self._finish("resonance", {"resonant_hz": f0, "characteristic_impedance_ohm": z0,
                                          "quality_factor": q, "bandwidth_hz": bandwidth},
                            inductance_h=l, capacitance_f=c)

    def ac_power(self, v_rms: float, i_rms: float, power_factor: float) -> Dict[str, float]:
        """Real, reactive, and apparent power."""
        v = self._num(v_rms, "v_rms", minimum=0.0)
        i = self._num(i_rms, "i_rms", minimum=0.0)
        pf = self._num(power_factor, "power_factor", minimum=0.0, maximum=1.0)
        s = v * i
        return self._finish("ac_power", {"real_w": s * pf, "reactive_var": s * math.sqrt(1.0 - pf ** 2), "apparent_va": s},
                            v_rms=v, i_rms=i, power_factor=pf)

    def stored_energy(self, kind: str, value: float, level: float) -> float:
        """Energy in a capacitor (C, V) or inductor (L, I): 0.5 * X * level^2."""
        element = self._choice(kind, ("capacitor", "inductor"), "kind")
        x = self._num(value, "value", positive=True)
        lvl = self._num(level, "level")
        return self._finish("stored_energy", 0.5 * x * lvl ** 2, kind=element, value=x, level=lvl)

    def led_resistor(self, supply_v: float, forward_v: float, forward_current_a: float) -> Dict[str, float]:
        """Series resistor value and derated power rating for an LED."""
        vs = self._num(supply_v, "supply_v", positive=True)
        vf = self._num(forward_v, "forward_v", positive=True)
        i = self._num(forward_current_a, "forward_current_a", positive=True)
        self._require(vs > vf, "'supply_v' must exceed 'forward_v'.", supply_v=vs, forward_v=vf)
        dissipation = (vs - vf) * i
        return self._finish("led_resistor",
                            {"resistance_ohm": (vs - vf) / i, "dissipation_w": dissipation,
                             "minimum_power_rating_w": dissipation / self.resistor_power_derating},
                            supply_v=vs, forward_v=vf, forward_current_a=i)

    def opamp_gain(self, topology: str, feedback_resistance_ohm: float, input_resistance_ohm: float,
                   gain_bandwidth_hz: Optional[float] = None) -> Dict[str, Optional[float]]:
        """Ideal closed-loop gain and, with GBW, closed-loop bandwidth (GBW / noise gain)."""
        mode = self._choice(topology, ("inverting", "non_inverting"), "topology")
        rf = self._num(feedback_resistance_ohm, "feedback_resistance_ohm", positive=True)
        rin = self._num(input_resistance_ohm, "input_resistance_ohm", positive=True)
        gain = -rf / rin if mode == "inverting" else 1.0 + rf / rin
        noise_gain = 1.0 + rf / rin
        bandwidth = None
        if gain_bandwidth_hz is not None:
            bandwidth = self._num(gain_bandwidth_hz, "gain_bandwidth_hz", positive=True) / noise_gain
        return self._finish("opamp_gain", {"gain": gain, "gain_db": self._db(abs(gain), "voltage"),
                                           "closed_loop_bandwidth_hz": bandwidth},
                            topology=mode, feedback_resistance_ohm=rf, input_resistance_ohm=rin)

    def to_db(self, ratio: float, kind: str = "power") -> float:
        """Convert a linear ratio to decibels ('power' = 10 log, 'voltage' = 20 log)."""
        value = self._num(ratio, "ratio", positive=True)
        mode = self._choice(kind, ("power", "voltage"), "kind")
        return self._finish("to_db", self._db(value, mode), ratio=value, kind=mode)

    def from_db(self, decibels: float, kind: str = "power") -> float:
        """Convert decibels to a linear ratio."""
        db = self._num(decibels, "decibels")
        mode = self._choice(kind, ("power", "voltage"), "kind")
        return self._finish("from_db", 10.0 ** (db / (10.0 if mode == "power" else 20.0)), decibels=db, kind=mode)

    def adc_metrics(self, bits: int, reference_voltage_v: float, sinad_db: Optional[float] = None) -> Dict[str, Optional[float]]:
        """LSB size, ideal SNR, and ENOB from measured SINAD."""
        n = self._count(bits, "bits")
        vref = self._num(reference_voltage_v, "reference_voltage_v", positive=True)
        enob = None if sinad_db is None else (self._num(sinad_db, "sinad_db") - 1.76) / 6.02
        return self._finish("adc_metrics", {"lsb_v": vref / 2 ** n, "levels": 2 ** n,
                                            "ideal_snr_db": 6.02 * n + 1.76, "enob_bits": enob},
                            bits=n, reference_voltage_v=vref)

    def required_sampling_rate(self, max_frequency_hz: float) -> float:
        """Practical minimum sampling rate (Nyquist times the configured oversampling factor)."""
        f = self._num(max_frequency_hz, "max_frequency_hz", positive=True)
        return self._finish("required_sampling_rate", self._sampling_rate(f), max_frequency_hz=f)

    def thermal_noise(self, bandwidth_hz: float, temperature_k: Optional[float] = None,
                      resistance_ohm: Optional[float] = None) -> Dict[str, Optional[float]]:
        """Johnson-Nyquist noise power (kTB) and, for a resistor, RMS noise voltage."""
        b = self._num(bandwidth_hz, "bandwidth_hz", positive=True)
        t = self._num(temperature_k if temperature_k is not None else self.noise_temperature_k, "temperature_k", positive=True)
        power = self.constants["kB"] * t * b
        vrms = None
        if resistance_ohm is not None:
            vrms = math.sqrt(4.0 * power * self._num(resistance_ohm, "resistance_ohm", positive=True))
        return self._finish("thermal_noise", {"noise_power_w": power, "noise_power_dbm": self._db(power / 1.0e-3, "power"),
                                              "noise_voltage_rms_v": vrms},
                            bandwidth_hz=b, temperature_k=t, resistance_ohm=resistance_ohm)

    def junction_temperature(self, ambient_c: float, power_dissipation_w: float, theta_ja_c_per_w: float,
                             max_junction_c: Optional[float] = None) -> Dict[str, Any]:
        """Tj = Ta + P * theta_JA, checked against the junction limit."""
        ta = self._num(ambient_c, "ambient_c")
        p = self._num(power_dissipation_w, "power_dissipation_w", minimum=0.0)
        theta = self._num(theta_ja_c_per_w, "theta_ja_c_per_w", positive=True)
        limit = self._num(max_junction_c if max_junction_c is not None else self.max_junction_temperature_c, "max_junction_c")
        tj = ta + p * theta
        check = self.evaluate_limit("junction_temperature", tj, limit, mode="max", unit="C")
        return self._finish("junction_temperature", {"junction_c": tj, "margin_c": limit - tj, "check": check.to_dict()},
                            ambient_c=ta, power_dissipation_w=p, theta_ja_c_per_w=theta)

    def battery_runtime(self, capacity_mah: float, load_ma: float, derating: Optional[float] = None) -> float:
        """Runtime in hours for a constant load."""
        cap = self._num(capacity_mah, "capacity_mah", positive=True)
        load = self._num(load_ma, "load_ma", positive=True)
        k = self._num(derating if derating is not None else self.battery_derating_factor, "derating", positive=True, maximum=1.0)
        return self._finish("battery_runtime", cap * k / load, capacity_mah=cap, load_ma=load, derating=k)

    def conductor_resistance(self, length_m: float, cross_section_m2: float, temperature_c: Optional[float] = None,
                             resistivity_ohm_m: Optional[float] = None, temp_coefficient_per_c: Optional[float] = None) -> float:
        """DC resistance R = rho(T) L / A (copper defaults)."""
        length = self._num(length_m, "length_m", positive=True)
        area = self._num(cross_section_m2, "cross_section_m2", positive=True)
        rho = self._num(resistivity_ohm_m if resistivity_ohm_m is not None else self.copper_resistivity_ohm_m,
                        "resistivity_ohm_m", positive=True)
        alpha = self._num(temp_coefficient_per_c if temp_coefficient_per_c is not None else self.copper_temp_coefficient_per_c,
                          "temp_coefficient_per_c")
        temp = self._num(temperature_c if temperature_c is not None else self.reference_conductor_temperature_c, "temperature_c")
        rho_t = rho * (1.0 + alpha * (temp - self.reference_conductor_temperature_c))
        return self._finish("conductor_resistance", rho_t * length / area, length_m=length,
                            cross_section_m2=area, temperature_c=temp)

    # ======================= communications ===================================
    def wavelength(self, frequency_hz: float) -> float:
        f = self._num(frequency_hz, "frequency_hz", positive=True)
        return self._finish("wavelength", self.constants["c"] / f, frequency_hz=f)

    def free_space_path_loss_db(self, distance_m: float, frequency_hz: float) -> float:
        d = self._num(distance_m, "distance_m", positive=True)
        f = self._num(frequency_hz, "frequency_hz", positive=True)
        return self._finish("free_space_path_loss_db", self._fspl_db(d, f), distance_m=d, frequency_hz=f)

    def link_budget(self, tx_power_dbm: float, tx_gain_dbi: float, rx_gain_dbi: float, distance_m: float,
                    frequency_hz: float, bandwidth_hz: float, noise_figure_db: float, *,
                    misc_losses_db: float = 0.0, required_snr_db: Optional[float] = None,
                    required_margin_db: Optional[float] = None) -> Dict[str, Any]:
        """Free-space link budget; with ``required_snr_db`` the margin is checked."""
        tx = self._num(tx_power_dbm, "tx_power_dbm")
        gt = self._num(tx_gain_dbi, "tx_gain_dbi")
        gr = self._num(rx_gain_dbi, "rx_gain_dbi")
        d = self._num(distance_m, "distance_m", positive=True)
        f = self._num(frequency_hz, "frequency_hz", positive=True)
        b = self._num(bandwidth_hz, "bandwidth_hz", positive=True)
        nf = self._num(noise_figure_db, "noise_figure_db", minimum=0.0)
        losses = self._num(misc_losses_db, "misc_losses_db", minimum=0.0)
        fspl = self._fspl_db(d, f)
        rx_power = tx + gt + gr - fspl - losses
        noise_floor = self._noise_floor_dbm(b, nf, self.noise_temperature_k)
        snr = rx_power - noise_floor
        result: Dict[str, Any] = {
            "eirp_dbm": tx + gt, "path_loss_db": fspl, "rx_power_dbm": rx_power,
            "noise_floor_dbm": noise_floor, "snr_db": snr, "margin_db": None, "margin_check": None,
        }
        if required_snr_db is not None:
            margin = snr - self._num(required_snr_db, "required_snr_db")
            needed = self._num(required_margin_db if required_margin_db is not None else self.required_link_margin_db,
                               "required_margin_db")
            result["margin_db"] = margin
            result["margin_check"] = self.evaluate_limit("link_margin_db", margin, needed, mode="min", unit="dB").to_dict()
        return self._finish("link_budget", result, distance_m=d, frequency_hz=f, bandwidth_hz=b)

    def shannon_capacity(self, bandwidth_hz: float, snr_db: float) -> Dict[str, float]:
        """AWGN channel capacity C = B log2(1 + SNR)."""
        b = self._num(bandwidth_hz, "bandwidth_hz", positive=True)
        snr = self._num(snr_db, "snr_db")
        spectral = math.log2(1.0 + 10.0 ** (snr / 10.0))
        return self._finish("shannon_capacity", {"capacity_bps": b * spectral, "spectral_efficiency_bps_hz": spectral},
                            bandwidth_hz=b, snr_db=snr)

    def ebn0_db(self, snr_db: float, bandwidth_hz: float, bit_rate_bps: float) -> float:
        """Eb/N0 (dB) = SNR + 10 log10(B / Rb)."""
        snr = self._num(snr_db, "snr_db")
        b = self._num(bandwidth_hz, "bandwidth_hz", positive=True)
        rb = self._num(bit_rate_bps, "bit_rate_bps", positive=True)
        return self._finish("ebn0_db", snr + self._db(b / rb, "power"), snr_db=snr, bandwidth_hz=b, bit_rate_bps=rb)

    def ber_awgn(self, modulation: str, ebn0_db: float, order: Optional[int] = None) -> float:
        """Bit-error rate over AWGN for BPSK/QPSK or square M-QAM (Gray-coded approximation)."""
        mode = self._choice(modulation, ("bpsk", "qpsk", "qam"), "modulation")
        ebn0 = 10.0 ** (self._num(ebn0_db, "ebn0_db") / 10.0)
        q = lambda x: 0.5 * math.erfc(x / math.sqrt(2.0))
        if mode in ("bpsk", "qpsk"):
            ber = q(math.sqrt(2.0 * ebn0))
        else:
            m = self._count(order, "order", minimum=4)
            root = math.isqrt(m)
            self._require(root * root == m and _is_power_of_two(m), "QAM order must be a square power of two (4, 16, 64, ...).", order=m)
            k = math.log2(m)
            ber = (4.0 / k) * (1.0 - 1.0 / root) * q(math.sqrt(3.0 * k / (m - 1.0) * ebn0))
        return self._finish("ber_awgn", min(1.0, ber), modulation=mode, ebn0_db=ebn0_db, order=order)

    def bandwidth_delay_product(self, bandwidth_bps: float, round_trip_time_s: float) -> Dict[str, float]:
        bw = self._num(bandwidth_bps, "bandwidth_bps", positive=True)
        rtt = self._num(round_trip_time_s, "round_trip_time_s", positive=True)
        return self._finish("bandwidth_delay_product", {"bits": bw * rtt, "bytes": bw * rtt / 8.0},
                            bandwidth_bps=bw, round_trip_time_s=rtt)

    def hamming_parity_bits(self, data_bits: int) -> Dict[str, float]:
        """Parity bits r for single-error correction: smallest r with 2^r >= m + r + 1."""
        m = self._count(data_bits, "data_bits")
        r = 1
        while 2 ** r < m + r + 1:
            r += 1
        return self._finish("hamming_parity_bits", {"parity_bits": r, "codeword_bits": m + r, "code_rate": m / (m + r)},
                            data_bits=m)

    def doppler_shift_hz(self, relative_velocity_m_s: float, frequency_hz: float) -> float:
        v = self._num(relative_velocity_m_s, "relative_velocity_m_s")
        f = self._num(frequency_hz, "frequency_hz", positive=True)
        return self._finish("doppler_shift_hz", v * f / self.constants["c"], relative_velocity_m_s=v, frequency_hz=f)

    def parabolic_antenna_gain_dbi(self, diameter_m: float, frequency_hz: float, efficiency: Optional[float] = None) -> float:
        """G = eta * (pi D / lambda)^2 in dBi."""
        d = self._num(diameter_m, "diameter_m", positive=True)
        f = self._num(frequency_hz, "frequency_hz", positive=True)
        eta = self._num(efficiency if efficiency is not None else self.antenna_efficiency, "efficiency", positive=True, maximum=1.0)
        gain = eta * (math.pi * d * f / self.constants["c"]) ** 2
        return self._finish("parabolic_antenna_gain_dbi", self._db(gain, "power"), diameter_m=d, frequency_hz=f, efficiency=eta)


# ======================================================================
# Mechanical engineering
# ======================================================================
class MechanicalEngineering(_StructuralMixin, _DomainBase):
    """Machine design, fluid power, thermal systems, and dynamics."""

    DOMAIN = "mechanical"

    CONFIG_FIELDS = (
        "bolt_torque_coefficient", "bolt_preload_proof_fraction", "gear_stage_efficiency",
        "bearing_ball_exponent", "bearing_roller_exponent", "thin_wall_ratio_limit",
        "pipe_roughness_m", "pump_efficiency", "laminar_reynolds_limit", "fluid_viscosity_pa_s",
        "endurance_ratio", "endurance_limit_cap_pa",
    )

    def _configure(self, mapping: Mapping[str, Any]) -> None:
        self.bolt_torque_coefficient = coerce_float(mapping.get("bolt_torque_coefficient", 0.2), 0.2, minimum=0.01)
        self.bolt_preload_proof_fraction = coerce_float(mapping.get("bolt_preload_proof_fraction", 0.75), 0.75, minimum=0.1, maximum=1.0)
        self.gear_stage_efficiency = coerce_float(mapping.get("gear_stage_efficiency", 0.98), 0.98, minimum=0.1, maximum=1.0)
        self.bearing_ball_exponent = coerce_float(mapping.get("bearing_ball_exponent", 3.0), 3.0, minimum=1.0)
        self.bearing_roller_exponent = coerce_float(mapping.get("bearing_roller_exponent", 10.0 / 3.0), 10.0 / 3.0, minimum=1.0)
        self.thin_wall_ratio_limit = coerce_float(mapping.get("thin_wall_ratio_limit", 10.0), 10.0, minimum=1.0)
        self.pipe_roughness_m = coerce_float(mapping.get("pipe_roughness_m", 4.5e-5), 4.5e-5, minimum=0.0)
        self.pump_efficiency = coerce_float(mapping.get("pump_efficiency", 0.7), 0.7, minimum=0.01, maximum=1.0)
        self.laminar_reynolds_limit = coerce_float(mapping.get("laminar_reynolds_limit", 2300.0), 2300.0, minimum=1.0)
        self.fluid_viscosity_pa_s = coerce_float(mapping.get("fluid_viscosity_pa_s", 8.9e-4), 8.9e-4, minimum=1.0e-9)
        self.endurance_ratio = coerce_float(mapping.get("endurance_ratio", 0.5), 0.5, minimum=0.01, maximum=1.0)
        self.endurance_limit_cap_pa = coerce_float(mapping.get("endurance_limit_cap_pa", 7.0e8), 7.0e8, minimum=1.0)

    def _validate_config(self) -> None:
        ensure_numeric_range(self.gear_stage_efficiency, "gear_stage_efficiency", minimum=0.1, maximum=1.0,
                             config=self.config, error_cls=BaseConfigurationError)

    # -- stress and machine elements -------------------------------------------
    def von_mises_stress(self, sigma_x_pa: float, sigma_y_pa: float = 0.0, tau_xy_pa: float = 0.0) -> float:
        """Plane-stress equivalent (distortion-energy) stress."""
        sx, sy, txy = (self._num(v, n) for v, n in ((sigma_x_pa, "sigma_x_pa"), (sigma_y_pa, "sigma_y_pa"), (tau_xy_pa, "tau_xy_pa")))
        return self._finish("von_mises_stress", math.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3.0 * txy ** 2),
                            sigma_x_pa=sx, sigma_y_pa=sy, tau_xy_pa=txy)

    def shaft_torsion(self, torque_nm: float, diameter_m: float, *, inner_diameter_m: Optional[float] = None,
                      bending_moment_nm: float = 0.0, material: Optional[str] = None) -> Dict[str, Any]:
        """Combined torsion and bending in a solid or hollow shaft, checked against yield."""
        torque = self._num(torque_nm, "torque_nm", minimum=0.0)
        moment = self._num(bending_moment_nm, "bending_moment_nm", minimum=0.0)
        shape = "hollow_circle" if inner_diameter_m is not None else "circle"
        props = self.section_properties(shape, diameter_m=diameter_m, inner_diameter_m=inner_diameter_m)
        c = props["extreme_fiber_m"]
        tau = torque * c / props["polar_moment_m4"]
        sigma = moment * c / props["second_moment_m4"]
        equivalent = math.sqrt(sigma ** 2 + 3.0 * tau ** 2)
        safety = self.factor_of_safety(self.material(material)["yield_strength"], equivalent)
        return self._finish("shaft_torsion",
                            {"shear_stress_pa": tau, "bending_stress_pa": sigma, "von_mises_pa": equivalent,
                             "safety": safety},
                            torque_nm=torque, diameter_m=diameter_m, bending_moment_nm=moment)

    def torque_from_power(self, power_w: float, speed_rpm: float) -> Dict[str, float]:
        """T = P / omega."""
        p = self._num(power_w, "power_w", positive=True)
        rpm = self._num(speed_rpm, "speed_rpm", positive=True)
        omega = rpm * 2.0 * math.pi / 60.0
        return self._finish("torque_from_power", {"torque_nm": p / omega, "angular_velocity_rad_s": omega},
                            power_w=p, speed_rpm=rpm)

    def spur_gear_geometry(self, module_mm: float, teeth: int, pressure_angle_deg: float = 20.0) -> Dict[str, float]:
        """Standard full-depth spur gear dimensions (millimetres)."""
        m = self._num(module_mm, "module_mm", positive=True)
        z = self._count(teeth, "teeth", minimum=6)
        angle = self._num(pressure_angle_deg, "pressure_angle_deg", positive=True, maximum=45.0)
        pitch = m * z
        return self._finish("spur_gear_geometry",
                            {"pitch_diameter_mm": pitch, "outside_diameter_mm": m * (z + 2), "root_diameter_mm": m * (z - 2.5),
                             "base_diameter_mm": pitch * math.cos(math.radians(angle)), "circular_pitch_mm": math.pi * m},
                            module_mm=m, teeth=z, pressure_angle_deg=angle)

    def gear_train(self, stages: Sequence[Sequence[int]], input_speed_rpm: float, input_torque_nm: float,
                   stage_efficiency: Optional[float] = None) -> Dict[str, float]:
        """Overall ratio, output speed, and output torque; ``stages`` is [(driver_teeth, driven_teeth), ...]."""
        pairs = ensure_list(stages)
        self._require(len(pairs) >= 1, "At least one gear stage is required.")
        ratio = 1.0
        for index, pair in enumerate(pairs):
            teeth = ensure_list(pair)
            self._require(len(teeth) == 2, f"stages[{index}] must be (driver_teeth, driven_teeth).", index=index)
            ratio *= self._count(teeth[1], f"stages[{index}].driven", minimum=6) / self._count(teeth[0], f"stages[{index}].driver", minimum=6)
        speed = self._num(input_speed_rpm, "input_speed_rpm", positive=True)
        torque = self._num(input_torque_nm, "input_torque_nm", minimum=0.0)
        eta = self._num(stage_efficiency if stage_efficiency is not None else self.gear_stage_efficiency,
                        "stage_efficiency", positive=True, maximum=1.0)
        overall = eta ** len(pairs)
        return self._finish("gear_train",
                            {"ratio": ratio, "output_speed_rpm": speed / ratio, "output_torque_nm": torque * ratio * overall,
                             "overall_efficiency": overall},
                            stages=len(pairs), input_speed_rpm=speed, input_torque_nm=torque)

    def bolt_preload_check(self, torque_nm: float, nominal_diameter_m: float, pitch_m: float, proof_strength_pa: float,
                           torque_coefficient: Optional[float] = None) -> Dict[str, Any]:
        """Preload F = T / (K d) and tensile stress against a fraction of proof strength."""
        t = self._num(torque_nm, "torque_nm", positive=True)
        d = self._num(nominal_diameter_m, "nominal_diameter_m", positive=True)
        pitch = self._num(pitch_m, "pitch_m", positive=True)
        proof = self._num(proof_strength_pa, "proof_strength_pa", positive=True)
        k = self._num(torque_coefficient if torque_coefficient is not None else self.bolt_torque_coefficient,
                      "torque_coefficient", positive=True)
        self._require(pitch < d, "'pitch_m' must be smaller than 'nominal_diameter_m'.", pitch_m=pitch, nominal_diameter_m=d)
        preload = t / (k * d)
        stress_area = math.pi / 4.0 * (d - 0.9382 * pitch) ** 2
        stress = preload / stress_area
        check = self.evaluate_limit("bolt_preload_stress", stress, self.bolt_preload_proof_fraction * proof, mode="max", unit="Pa")
        return self._finish("bolt_preload_check",
                            {"preload_n": preload, "tensile_stress_area_m2": stress_area, "tensile_stress_pa": stress,
                             "check": check.to_dict()},
                            torque_nm=t, nominal_diameter_m=d, pitch_m=pitch, proof_strength_pa=proof)

    def bearing_life(self, dynamic_capacity_n: float, equivalent_load_n: float, speed_rpm: float,
                     bearing_type: str = "ball") -> Dict[str, float]:
        """Basic rating life L10 in million revolutions and hours."""
        c = self._num(dynamic_capacity_n, "dynamic_capacity_n", positive=True)
        p = self._num(equivalent_load_n, "equivalent_load_n", positive=True)
        rpm = self._num(speed_rpm, "speed_rpm", positive=True)
        kind = self._choice(bearing_type, ("ball", "roller"), "bearing_type")
        exponent = self.bearing_ball_exponent if kind == "ball" else self.bearing_roller_exponent
        l10 = (c / p) ** exponent
        return self._finish("bearing_life", {"l10_million_rev": l10, "l10_hours": l10 * 1.0e6 / (60.0 * rpm)},
                            dynamic_capacity_n=c, equivalent_load_n=p, speed_rpm=rpm, bearing_type=kind)

    def fatigue_goodman(self, alternating_stress_pa: float, mean_stress_pa: float, *, material: Optional[str] = None,
                        marin_factor_product: float = 1.0) -> Dict[str, Any]:
        """Endurance limit estimate and Goodman safety factor for infinite life."""
        sa = self._num(alternating_stress_pa, "alternating_stress_pa", minimum=0.0)
        sm = self._num(mean_stress_pa, "mean_stress_pa", minimum=0.0)
        k = self._num(marin_factor_product, "marin_factor_product", positive=True, maximum=1.0)
        sut = self.material(material)["ultimate_strength"]
        se = min(self.endurance_ratio * sut, self.endurance_limit_cap_pa) * k
        usage = sa / se + sm / sut
        safety = self.factor_of_safety(1.0, usage, required=1.0)
        return self._finish("fatigue_goodman",
                            {"endurance_limit_pa": se, "goodman_factor_of_safety": safety["factor_of_safety"],
                             "infinite_life": safety["passed"], "check": safety["check"]},
                            alternating_stress_pa=sa, mean_stress_pa=sm, marin_factor_product=k)

    def pressure_vessel_thin_wall(self, pressure_pa: float, radius_m: float, thickness_m: float, *,
                                  shape: str = "cylinder", allowable_stress_pa: Optional[float] = None,
                                  material: Optional[str] = None) -> Dict[str, Any]:
        """Thin-wall stresses and minimum thickness for a cylinder or sphere."""
        p = self._num(pressure_pa, "pressure_pa", positive=True)
        r = self._num(radius_m, "radius_m", positive=True)
        t = self._num(thickness_m, "thickness_m", positive=True)
        form = self._choice(shape, ("cylinder", "sphere"), "shape")
        allowable = self._num(allowable_stress_pa if allowable_stress_pa is not None
                              else self.material(material)["yield_strength"] / self.shared.default_safety_factor,
                              "allowable_stress_pa", positive=True)
        longitudinal = p * r / (2.0 * t)
        hoop = p * r / t if form == "cylinder" else longitudinal
        required = (p * r / allowable) if form == "cylinder" else (p * r / (2.0 * allowable))
        check = self.evaluate_limit("pressure_vessel_stress", hoop, allowable, mode="max", unit="Pa")
        return self._finish("pressure_vessel_thin_wall",
                            {"hoop_stress_pa": hoop, "longitudinal_stress_pa": longitudinal,
                             "required_thickness_m": required, "thin_wall_valid": r / t >= self.thin_wall_ratio_limit,
                             "check": check.to_dict()},
                            pressure_pa=p, radius_m=r, thickness_m=t, shape=form)

    # -- fluids and thermal --------------------------------------------------------
    def pipe_flow(self, flow_m3_s: float, diameter_m: float, length_m: float, *, density_kg_m3: Optional[float] = None,
                  viscosity_pa_s: Optional[float] = None, roughness_m: Optional[float] = None,
                  static_head_m: float = 0.0) -> Dict[str, Any]:
        """Darcy-Weisbach friction loss (Swamee-Jain for turbulent flow) and pump hydraulic/shaft power."""
        q = self._num(flow_m3_s, "flow_m3_s", positive=True)
        d = self._num(diameter_m, "diameter_m", positive=True)
        length = self._num(length_m, "length_m", positive=True)
        rho = self._num(density_kg_m3 if density_kg_m3 is not None else self.constants["rho_water"], "density_kg_m3", positive=True)
        mu = self._num(viscosity_pa_s if viscosity_pa_s is not None else self.fluid_viscosity_pa_s, "viscosity_pa_s", positive=True)
        eps = self._num(roughness_m if roughness_m is not None else self.pipe_roughness_m, "roughness_m", minimum=0.0)
        head = self._num(static_head_m, "static_head_m")
        g = self.constants["g"]
        velocity = q / (math.pi * d ** 2 / 4.0)
        reynolds = rho * velocity * d / mu
        laminar = reynolds <= self.laminar_reynolds_limit
        friction = 64.0 / reynolds if laminar else 0.25 / math.log10(eps / (3.7 * d) + 5.74 / reynolds ** 0.9) ** 2
        loss = friction * (length / d) * velocity ** 2 / (2.0 * g)
        hydraulic = rho * g * q * (loss + head)
        return self._finish("pipe_flow",
                            {"velocity_m_s": velocity, "reynolds": reynolds, "regime": "laminar" if laminar else "turbulent",
                             "friction_factor": friction, "head_loss_m": loss, "pressure_drop_pa": rho * g * loss,
                             "hydraulic_power_w": hydraulic, "shaft_power_w": hydraulic / self.pump_efficiency},
                            flow_m3_s=q, diameter_m=d, length_m=length)

    def heat_exchanger_duty(self, overall_u_w_m2k: float, area_m2: float, hot_in_k: float, hot_out_k: float,
                            cold_in_k: float, cold_out_k: float, arrangement: str = "counterflow") -> Dict[str, float]:
        """LMTD and duty Q = U A LMTD for counterflow or parallel-flow exchangers."""
        u = self._num(overall_u_w_m2k, "overall_u_w_m2k", positive=True)
        area = self._num(area_m2, "area_m2", positive=True)
        th_in, th_out = self._num(hot_in_k, "hot_in_k", positive=True), self._num(hot_out_k, "hot_out_k", positive=True)
        tc_in, tc_out = self._num(cold_in_k, "cold_in_k", positive=True), self._num(cold_out_k, "cold_out_k", positive=True)
        mode = self._choice(arrangement, ("counterflow", "parallel"), "arrangement")
        dt1, dt2 = (th_in - tc_out, th_out - tc_in) if mode == "counterflow" else (th_in - tc_in, th_out - tc_out)
        self._require(dt1 > 0.0 and dt2 > 0.0, "Terminal temperature differences must be positive (check the temperatures).",
                      delta_t1=dt1, delta_t2=dt2)
        lmtd = dt1 if abs(dt1 - dt2) <= self.shared.numeric_tolerance else (dt1 - dt2) / math.log(dt1 / dt2)
        return self._finish("heat_exchanger_duty", {"lmtd_k": lmtd, "duty_w": u * area * lmtd},
                            overall_u_w_m2k=u, area_m2=area, arrangement=mode)

    def effectiveness_ntu(self, ntu: float, capacity_ratio: float, arrangement: str = "counterflow") -> float:
        """Heat-exchanger effectiveness from NTU and Cmin/Cmax."""
        n = self._num(ntu, "ntu", positive=True)
        cr = self._num(capacity_ratio, "capacity_ratio", minimum=0.0, maximum=1.0)
        mode = self._choice(arrangement, ("counterflow", "parallel"), "arrangement")
        if mode == "parallel":
            eff = (1.0 - math.exp(-n * (1.0 + cr))) / (1.0 + cr)
        elif abs(cr - 1.0) <= self.shared.numeric_tolerance:
            eff = n / (1.0 + n)
        else:
            eff = (1.0 - math.exp(-n * (1.0 - cr))) / (1.0 - cr * math.exp(-n * (1.0 - cr)))
        return self._finish("effectiveness_ntu", eff, ntu=n, capacity_ratio=cr, arrangement=mode)

    def carnot_limits(self, t_hot_k: float, t_cold_k: float) -> Dict[str, float]:
        """Carnot engine efficiency and refrigerator/heat-pump COP bounds."""
        hot = self._num(t_hot_k, "t_hot_k", positive=True)
        cold = self._num(t_cold_k, "t_cold_k", positive=True)
        self._require(hot > cold, "'t_hot_k' must exceed 't_cold_k'.", t_hot_k=hot, t_cold_k=cold)
        return self._finish("carnot_limits",
                            {"efficiency": 1.0 - cold / hot, "cop_cooling": cold / (hot - cold), "cop_heating": hot / (hot - cold)},
                            t_hot_k=hot, t_cold_k=cold)

    # -- dynamics and tolerancing -----------------------------------------------------
    def vibration_properties(self, stiffness_n_m: float, mass_kg: float, damping_ns_m: float = 0.0) -> Dict[str, Any]:
        """Natural frequency, damping ratio, damped frequency, and regime of a 1-DOF system."""
        k = self._num(stiffness_n_m, "stiffness_n_m", positive=True)
        m = self._num(mass_kg, "mass_kg", positive=True)
        c = self._num(damping_ns_m, "damping_ns_m", minimum=0.0)
        omega = math.sqrt(k / m)
        critical = 2.0 * math.sqrt(k * m)
        zeta = c / critical
        regime = "undamped" if c == 0.0 else "underdamped" if zeta < 1.0 else "critically_damped" if abs(zeta - 1.0) <= 1.0e-9 else "overdamped"
        return self._finish("vibration_properties",
                            {"natural_rad_s": omega, "natural_hz": omega / (2.0 * math.pi), "damping_ratio": zeta,
                             "critical_damping_ns_m": critical,
                             "damped_hz": omega * math.sqrt(1.0 - zeta ** 2) / (2.0 * math.pi) if zeta < 1.0 else 0.0,
                             "regime": regime},
                            stiffness_n_m=k, mass_kg=m, damping_ns_m=c)

    def tolerance_stack(self, tolerances: Sequence[float], method: str = "worst_case") -> float:
        """Total +/- tolerance by worst-case sum or root-sum-square."""
        values = self._values(tolerances, "tolerances", positive=True)
        mode = self._choice(method, ("worst_case", "rss"), "method")
        total = sum(values) if mode == "worst_case" else math.sqrt(sum(v * v for v in values))
        return self._finish("tolerance_stack", total, count=len(values), method=mode)


# ======================================================================
# Facade
# ======================================================================
class EngineeringEngine:
    """
    Centralized engineering engine for the SLAI Environment.

    Owns the five engineering domains, the shared material library, and a
    single bounded audit history. Defaults live in ``base_config.yaml`` under
    ``base_engineering``; runtime overrides are deep-merged on top.
    """

    DOMAINS: Dict[str, type] = {
        "medical": MedicalEngineering,
        "civil": CivilEngineering,
        "computer": ComputerEngineering,
        "electronics_comms": ElectronicsCommsEngineering,
        "mechanical": MechanicalEngineering,
    }

    CONFIG_FIELDS: Tuple[str, ...] = (
        "enable_history", "history_limit", "numeric_tolerance", "strict_checks",
        "default_safety_factor", "check_warn_utilization", "check_critical_utilization",
        "sampling_oversample_factor", "default_material", "materials",
    )

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        self.global_config = load_global_config()
        base_engineering_config = get_config_section("base_engineering") or {}
        ensure_mapping(
            base_engineering_config,
            "base_engineering",
            config=base_engineering_config,
            error_cls=BaseConfigurationError,
            component="EngineeringEngine",
            operation="__init__",
        )

        if config is None:
            self.engineering_config = dict(base_engineering_config)
        elif isinstance(config, Mapping):
            # Runtime overrides are supported, but canonical defaults remain in base_config.yaml.
            self.engineering_config = deep_merge_dicts(base_engineering_config, dict(config))
        else:
            raise BaseValidationError(
                "config must be None or a mapping of engineering overrides.",
                base_engineering_config,
                component="EngineeringEngine",
                operation="__init__",
                context={"received_type": type(config).__name__},
            )

        mapping = self.engineering_config

        self.enable_history = coerce_bool(mapping.get("enable_history", True), True)
        self.history_limit = coerce_int(mapping.get("history_limit", 200), 200, minimum=1)
        self.numeric_tolerance = coerce_float(mapping.get("numeric_tolerance", 1.0e-12), 1.0e-12, minimum=0.0)
        self.strict_checks = coerce_bool(mapping.get("strict_checks", False), False)
        self.default_safety_factor = coerce_float(mapping.get("default_safety_factor", 1.5), 1.5, minimum=1.0)
        self.check_warn_utilization = coerce_float(mapping.get("check_warn_utilization", 0.9), 0.9, minimum=0.0)
        self.check_critical_utilization = coerce_float(mapping.get("check_critical_utilization", 1.5), 1.5, minimum=1.0)
        self.sampling_oversample_factor = coerce_float(mapping.get("sampling_oversample_factor", 2.5), 2.5, minimum=2.0)
        self.materials = self._parse_materials(mapping.get("materials"))
        self.default_material = str(mapping.get("default_material", "structural_steel_a36")).strip().lower()
        self.constants = dict(PhysicsEngine.CONSTANTS)

        self._validate_config()

        self._recorder = _AuditRecorder(self.enable_history, self.history_limit)
        shared = _SharedContext(
            constants=self.constants,
            materials=self.materials,
            default_material=self.default_material,
            default_safety_factor=self.default_safety_factor,
            numeric_tolerance=self.numeric_tolerance,
            strict_checks=self.strict_checks,
            warn_utilization=self.check_warn_utilization,
            critical_utilization=self.check_critical_utilization,
            sampling_oversample_factor=self.sampling_oversample_factor,
        )
        self.medical: MedicalEngineering = MedicalEngineering(mapping.get("medical") or {}, self._recorder, shared)
        self.civil: CivilEngineering = CivilEngineering(mapping.get("civil") or {}, self._recorder, shared)
        self.computer: ComputerEngineering = ComputerEngineering(mapping.get("computer") or {}, self._recorder, shared)
        self.electronics_comms: ElectronicsCommsEngineering = ElectronicsCommsEngineering(
            mapping.get("electronics_comms") or {}, self._recorder, shared)
        self.mechanical: MechanicalEngineering = MechanicalEngineering(mapping.get("mechanical") or {}, self._recorder, shared)

        logger.info("Engineering Engine successfully initialized")

    # ------------------------------------------------------------------
    # Configuration and validation
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_materials(raw: Any) -> Dict[str, Dict[str, float]]:
        ensure_mapping(
            raw if raw is not None else {},
            "materials",
            config=raw if isinstance(raw, Mapping) else None,
            error_cls=BaseConfigurationError,
            component="EngineeringEngine",
            operation="configuration",
        )
        materials: Dict[str, Dict[str, float]] = {}
        for name, props in (raw or {}).items():
            ensure_keys(props, _MATERIAL_KEYS, f"materials.{name}", error_cls=BaseConfigurationError,
                        component="EngineeringEngine", operation="configuration")
            materials[str(name).strip().lower()] = {key: coerce_float(props[key], 0.0, minimum=0.0) for key in _MATERIAL_KEYS}
        return materials

    def _validate_config(self) -> None:
        ensure_condition(
            self.default_material in self.materials,
            f"default_material '{self.default_material}' is not defined in 'materials'.",
            config=self.engineering_config,
            error_cls=BaseConfigurationError,
            component="EngineeringEngine",
            operation="configuration",
            context={"available": sorted(self.materials)},
        )
        ensure_numeric_range(
            self.check_warn_utilization, "check_warn_utilization",
            minimum=0.0, maximum=1.0, config=self.engineering_config, error_cls=BaseConfigurationError,
        )

    def _config_as_dict(self) -> Dict[str, Any]:
        """Return the active engine-level configuration as a plain dictionary."""
        return {key: getattr(self, key) for key in self.CONFIG_FIELDS}

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    def recent_history(self, limit: int = 20, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        limit = coerce_int(limit, 20, minimum=1)
        entries = list(self._recorder.history)
        if domain is not None:
            entries = [entry for entry in entries if entry.get("domain") == domain]
        return entries[-limit:]

    def stats(self) -> Dict[str, Any]:
        return {
            "config": to_json_safe(self._config_as_dict()),
            "domain_config": {name: to_json_safe(getattr(self, name)._config_as_dict()) for name in self.DOMAINS},
            "stats": to_json_safe(self._recorder.stats),
            "history_length": len(self._recorder.history),
            "constants_count": len(self.constants),
        }


# ========== Accessor (shared engine instance) ==========
_engine: Optional[EngineeringEngine] = None


def get_engineering_engine(config: Optional[Mapping[str, Any]] = None, *, refresh: bool = False) -> EngineeringEngine:
    """Get or create the shared EngineeringEngine; pass ``refresh=True`` or ``config`` to rebuild it."""
    global _engine
    if _engine is None or refresh or config is not None:
        _engine = EngineeringEngine(config)
    return _engine


__all__ = [
    "MedicalEngineering",
    "CivilEngineering",
    "ComputerEngineering",
    "ElectronicsCommsEngineering",
    "MechanicalEngineering",
    "EngineeringEngine",
    "get_engineering_engine",
]


if __name__ == "__main__":
    print("\n=== Running Base Engineering ===\n")
    printer.status("TEST", "Base Engineering initialized", "info")

    engine = EngineeringEngine()

    bsa = engine.medical.body_surface_area(70.0, 175.0)
    printer.pretty("MEDICAL_BSA_M2", round(bsa, 4), "success")
    printer.pretty("MEDICAL_CARDIAC_OUTPUT", engine.medical.cardiac_output(72, 70, bsa), "success")
    printer.pretty("MEDICAL_LEAKAGE_CHECK", engine.medical.check_leakage_current(8.0, "CF").to_dict(), "success")

    section = engine.civil.section_properties("rectangle", width_m=0.3, height_m=0.5)
    printer.pretty(
        "CIVIL_BEAM",
        engine.civil.beam_response(6.0, 20000.0, section["second_moment_m4"], support="simply_supported", load_type="udl"),
        "success",
    )

    printer.pretty("COMPUTER_SPEEDUP", engine.computer.parallel_speedup(0.95, 16), "success")
    printer.pretty("COMPUTER_QUEUE", engine.computer.queue_mm1(70.0, 100.0), "success")

    printer.pretty(
        "COMMS_LINK_BUDGET",
        engine.electronics_comms.link_budget(20.0, 10.0, 10.0, 5000.0, 2.4e9, 20.0e6, 5.0, required_snr_db=10.0),
        "success",
    )
    printer.pretty("COMMS_BER_QPSK", engine.electronics_comms.ber_awgn("qpsk", 8.0), "success")

    printer.pretty("MECH_SHAFT", engine.mechanical.shaft_torsion(500.0, 0.04, bending_moment_nm=300.0), "success")
    printer.pretty("MECH_PIPE_FLOW", engine.mechanical.pipe_flow(0.01, 0.1, 100.0), "success")

    printer.pretty("RECENT_HISTORY", engine.recent_history(5), "success")
    printer.pretty("ENGINEERING_STATS", engine.stats(), "success")

    print("\n=== Test ran successfully ===\n")
