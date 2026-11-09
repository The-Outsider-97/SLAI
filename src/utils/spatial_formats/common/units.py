"""Explicit metric/international length unit handling."""
from .errors import ValidationError
from .helpers import finite

METERS_PER_UNIT = {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0, "in": 0.0254, "ft": 0.3048}

def require_unit(unit: str) -> str:
    if not isinstance(unit, str) or unit not in METERS_PER_UNIT:
        raise ValidationError("unsupported length unit: " + str(unit))
    return unit

def convert_length(value: float, source: str, target: str) -> float:
    require_unit(source)
    require_unit(target)
    result = finite(value) * METERS_PER_UNIT[source] / METERS_PER_UNIT[target]
    return finite(result, "converted length")

def convert_area(value: float, source: str, target: str) -> float:
    require_unit(source)
    require_unit(target)
    factor = METERS_PER_UNIT[source] / METERS_PER_UNIT[target]
    return finite(finite(value) * factor * factor, "converted area")

def convert_volume(value: float, source: str, target: str) -> float:
    require_unit(source)
    require_unit(target)
    factor = METERS_PER_UNIT[source] / METERS_PER_UNIT[target]
    return finite(finite(value) * factor ** 3, "converted volume")


# --------------------------------------------------------------------------------------------------------------------
# Physical-quantity units for typed parameters (SGEO parameter types).
#
# Every parameter type has ONE internal (SI) unit; user-facing units convert at the boundary through to_si/from_si only.
# Angles are radians internally; absolute temperature is kelvin and is distinct from a temperature *difference*
# (an affine offset applies to the former, never to the latter).
# --------------------------------------------------------------------------------------------------------------------
_PI = 3.141592653589793

#: parameter type -> {unit symbol: SI units per one unit}. The first symbol is the SPADRA display default.
QUANTITY_UNITS = {
    "length": {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0, "in": 0.0254, "ft": 0.3048},
    "area": {"mm2": 1e-6, "cm2": 1e-4, "m2": 1.0, "in2": 0.0254 ** 2, "ft2": 0.3048 ** 2},
    "volume": {"mm3": 1e-9, "cm3": 1e-6, "L": 1e-3, "m3": 1.0, "in3": 0.0254 ** 3, "ft3": 0.3048 ** 3},
    "angle": {"deg": _PI / 180.0, "rad": 1.0, "grad": _PI / 200.0},
    "mass": {"kg": 1.0, "g": 1e-3, "t": 1000.0, "lb": 0.45359237},
    "time": {"s": 1.0, "ms": 1e-3, "min": 60.0, "h": 3600.0},
    "frequency": {"Hz": 1.0, "kHz": 1e3},
    "angular_velocity": {"rad/s": 1.0, "deg/s": _PI / 180.0, "rpm": 2 * _PI / 60.0},
    "velocity": {"m/s": 1.0, "mm/s": 1e-3, "km/h": 1 / 3.6},
    "acceleration": {"m/s2": 1.0, "mm/s2": 1e-3},
    "force": {"N": 1.0, "kN": 1e3, "MN": 1e6, "lbf": 4.4482216152605},
    "pressure": {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9, "bar": 1e5, "psi": 6894.757293168361},
    "energy": {"J": 1.0, "kJ": 1e3, "MJ": 1e6, "Wh": 3600.0, "kWh": 3.6e6},
    "power": {"W": 1.0, "kW": 1e3, "MW": 1e6},
    "density": {"kg/m3": 1.0, "g/cm3": 1000.0},
    "second_moment": {"m4": 1.0, "cm4": 1e-8, "mm4": 1e-12},
    "number": {"": 1.0},
    "integer": {"": 1.0},
}
#: Temperature is affine: kelvin = (value + offset) * scale for an absolute reading, value * scale for a difference.
_TEMPERATURE = {"K": (1.0, 0.0), "C": (1.0, 273.15), "F": (5.0 / 9.0, 459.67)}
QUANTITY_UNITS["temperature"] = {k: v[0] for k, v in _TEMPERATURE.items()}
QUANTITY_UNITS["temperature_difference"] = {k: v[0] for k, v in _TEMPERATURE.items()}


def units_for(parameter_type: str) -> tuple:
    """The unit symbols a parameter type can be entered/shown in (the first is the default display unit)."""
    if parameter_type not in QUANTITY_UNITS:
        raise ValidationError("parameter type has no physical units: " + str(parameter_type))
    return tuple(QUANTITY_UNITS[parameter_type])


def _unit_factor(parameter_type: str, unit: str) -> float:
    table = QUANTITY_UNITS.get(parameter_type)
    if table is None:
        raise ValidationError("parameter type has no physical units: " + str(parameter_type))
    if unit not in table:
        raise ValidationError("unsupported %s unit: %s" % (parameter_type, unit), context={"allowed": sorted(table)})
    return table[unit]


def to_si(parameter_type: str, value: float, unit: str) -> float:
    """Convert ``value`` given in ``unit`` to the type's internal SI unit (the only entry point for entry conversion)."""
    factor = _unit_factor(parameter_type, unit)
    value = finite(value)
    if parameter_type == "temperature":
        scale, offset = _TEMPERATURE[unit]
        return finite((value + offset) * scale, "converted temperature")
    return finite(value * factor, "converted " + parameter_type)


def from_si(parameter_type: str, value: float, unit: str) -> float:
    """Convert an internal SI value to ``unit`` for display (inverse of :func:`to_si`)."""
    factor = _unit_factor(parameter_type, unit)
    value = finite(value)
    if parameter_type == "temperature":
        scale, offset = _TEMPERATURE[unit]
        return finite(value / scale - offset, "converted temperature")
    return finite(value / factor, "converted " + parameter_type)
