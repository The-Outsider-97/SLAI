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
