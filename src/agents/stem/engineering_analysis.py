"""Verified unit-aware entry points over existing SLAI engineering/physics methods.

This module owns dimensional checking and result reporting, NOT duplicate beam,
thermal, electrical, or kinetic-energy formulas. Existing Engineering and
Physics implementations remain authoritative. No simulation infrastructure.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .engineering import Engineering
from .physics import Physics
from .parametric import ParametricMath
from .stem_types import Dimension, Quantity, Unit, NumericResult
from .units.unit_system import UnitSystem
from .utils.stem_errors import STEMDomainError, STEMUnitError, STEMValidationError


class EngineeringAnalysis:
    """Unit-checked, traceable analytical results with known model assumptions."""

    def __init__(self, units: UnitSystem | None = None, engineering: Engineering | None = None,
                 physics: Physics | None = None) -> None:
        self.units = units or UnitSystem()
        self.engineering = engineering or Engineering()
        self.physics = physics or Physics()
        self.parser = ParametricMath(units=self.units)

    def _unit(self, symbol: str) -> Unit:
        # A registered symbol (including °C, Δ°C) takes precedence. Composite
        # multiplicative unit expressions are parsed through the safe AST.
        # Never pass user-supplied unit strings through eval or exec.
        try:
            return self.units.lookup(symbol)
        except STEMUnitError:
            pass
        result = self.parser.evaluate_expression(symbol)
        if not math.isclose(result.magnitude, 1.0, rel_tol=0, abs_tol=1e-14):
            raise STEMUnitError("Unit expression cannot contain numeric multipliers")
        return result.unit

    def _quantity(self, name: str, inputs: Mapping[str, Any], expected_unit: str,
                  *, positive: bool = False, nonnegative: bool = False,
                  allow_affine: bool = False) -> tuple[float, dict[str, Any]]:
        if name not in inputs:
            raise STEMValidationError("Missing required engineering parameter", context={"parameter": name})
        arg = inputs[name]
        if isinstance(arg, Quantity):
            q = arg
        elif isinstance(arg, Mapping) and set(arg) == {"value", "unit"}:
            unit = arg["unit"]
            raw_value = arg["value"]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (float, int)):
                raise STEMValidationError("Engineering quantity value must be numeric")
            q = Quantity(float(raw_value), self._unit(unit) if unit != "1" else self.parser.one)
        else:
            raise STEMValidationError("Engineering inputs require {value, unit} mappings or Quantity",
                                      context={"parameter": name})
        expected = self._unit(expected_unit)
        if q.unit.dimension != expected.dimension:
            raise STEMUnitError("Incorrect engineering input dimension",
                                context={"parameter": name, "received": q.unit.symbol,
                                         "expected": expected_unit})
        if q.unit.is_affine and not allow_affine:
            raise STEMUnitError("Use a linear unit or explicit temperature difference",
                                context={"parameter": name})
        si = q.unit.to_base(q.magnitude)
        if positive and si <= 0:
            raise STEMDomainError("Input must be strictly positive", context={"parameter": name})
        if nonnegative and si < 0:
            raise STEMDomainError("Input cannot be negative", context={"parameter": name})
        return si, q.to_dict()

    @staticmethod
    def _report(model: str, value: float, unit: Unit, inputs: Mapping[str, Any],
                assumptions: list[str], *, details: Mapping[str, Any] | None = None) -> NumericResult:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise STEMDomainError("Engineering model returned a non-finite result")
        return NumericResult(
            value=float(value), unit=unit, method=model,
            metadata={"inputs": dict(inputs), "assumptions": assumptions,
                      "details": dict(details or {}), "validation": "analytical_baseline"},
        )

    def axial_stress(self, force: Mapping[str, Any], area: Mapping[str, Any]) -> dict[str, Any]:
        inputs = {"force": force, "area": area}
        f, _ = self._quantity("force", inputs, "N")
        a, _ = self._quantity("area", inputs, "m**2", positive=True)
        value = self.engineering.axial_stress(f, a)
        return self._report("axial_stress", value, self.units.lookup("Pa"), inputs,
                            ["uniform axial stress", "constant cross-sectional area"])

    def heat_conduction_1d(self, conductivity: Mapping[str, Any], area: Mapping[str, Any],
                           delta_temperature: Mapping[str, Any], length: Mapping[str, Any]) -> dict[str, Any]:
        inputs = dict(conductivity=conductivity, area=area,
                      delta_temperature=delta_temperature, length=length)
        k, _ = self._quantity("conductivity", inputs, "W/(m*K)", positive=True)
        a, _ = self._quantity("area", inputs, "m**2", positive=True)
        dt, _ = self._quantity("delta_temperature", inputs, "K")
        d, _ = self._quantity("length", inputs, "m", positive=True)
        value = self.engineering.heat_conduction_1d(k, a, dt, d)
        return self._report("heat_conduction_1d", value, self.units.lookup("W"), inputs,
                            ["steady-state", "uniform slab", "constant conductivity",
                             "one-dimensional conduction", "delta_temperature is a difference, not absolute temperature"])

    def electrical_power(self, voltage: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
        inputs = {"voltage": voltage, "current": current}
        v, _ = self._quantity("voltage", inputs, "V")
        i, _ = self._quantity("current", inputs, "A")
        value = self.engineering.electrical_power(v, i)
        return self._report("electrical_power", value, self.units.lookup("W"), inputs,
                            ["instantaneous electrical power under the adopted sign convention"])

    def kinetic_energy(self, mass: Mapping[str, Any], velocity: Mapping[str, Any]) -> dict[str, Any]:
        inputs = {"mass": mass, "velocity": velocity}
        m, _ = self._quantity("mass", inputs, "kg", nonnegative=True)
        v, _ = self._quantity("velocity", inputs, "m/s")
        value = self.physics.kinetic_energy(m, v)
        return self._report("kinetic_energy", value, self.units.lookup("J"), inputs,
                            ["classical translational kinetic energy", "inertial frame"])

    def beam_response(self, span: Mapping[str, Any], load: Mapping[str, Any],
                      second_moment: Mapping[str, Any], youngs_modulus: Mapping[str, Any],
                      *, support: str = "simply_supported", load_type: str = "point_center") -> dict[str, Any]:
        """Delegate to the existing Civil Engineering beam-response implementation.

        This is an analytical small-deflection model; its exact supported
        support/load combinations are validated by the base EngineeringEngine.
        """
        if not isinstance(support, str) or not isinstance(load_type, str):
            raise STEMValidationError("support/load_type must be strings")
        inputs = dict(span=span, load=load, second_moment=second_moment,
                      youngs_modulus=youngs_modulus)
        L, _ = self._quantity("span", inputs, "m", positive=True)
        p, _ = self._quantity("load", inputs, "N/m" if load_type == "udl" else "N")
        inertia, _ = self._quantity("second_moment", inputs, "m**4", positive=True)
        E, _ = self._quantity("youngs_modulus", inputs, "Pa", positive=True)
        result = self.engineering.civil.beam_response(
            span_m=L, load=p, second_moment_m4=inertia,
            support=support, load_type=load_type, youngs_modulus_pa=E,
        )
        if not isinstance(result, Mapping):
            raise STEMDomainError("Unexpected beam_response contract")
        names = {"max_moment_nm": "N*m", "max_shear_n": "N",
                 "max_deflection_m": "m", "deflection_limit_m": "m"}
        tagged = {}
        for key, symbol in names.items():
            if key in result:
                value = float(result[key])
                if not math.isfinite(value):
                    raise STEMDomainError("Beam model returned a non-finite result")
                tagged[key] = NumericResult(value=value, unit=self._unit(symbol),
                                            method="base_engineering.civil.beam_response").to_dict()
        if not tagged:
            raise STEMDomainError("Beam response did not provide expected values")
        return {"status": "success", "method": "base_engineering.civil.beam_response",
                "inputs": inputs, "outputs": tagged,
                "assumptions": ["linear elasticity", "small deflections", "prismatic beam",
                                "base engine support/load case applicability"],
                "details": {"support": support, "load_type": load_type,
                            "base_engineering_check": result.get("deflection_check")},
                "validation": "analytical_baseline"}


__all__ = ["EngineeringAnalysis"]
