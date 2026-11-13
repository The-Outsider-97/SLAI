"""Dependency-injected bridge to SLAI STEM's authoritative unit system.

The spatial container remains readable without initializing the STEM Agent.
The module-level helpers preserve the historical, format-local conversion API;
new scientific integrations should use :class:`STEMQuantityAdapter`, which
delegates dimensional checks, unit composition and conversion to the actual
STEM ``UnitSystem`` supplied by the application composition layer.

No agent module is imported here, either eagerly or during a conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from ..common.errors import ValidationError
from ..common.helpers import finite
from ..common.units import convert_area, convert_length, convert_volume, require_unit

_POWERS = {"length": 1, "area": 2, "volume": 3}
_LOCAL_CONVERSIONS = {
    "length": convert_length,
    "area": convert_area,
    "volume": convert_volume,
}


class STEMUnitService(Protocol):
    """Minimal contract provided by ``src.agents.stem.units.UnitSystem``.

    Using a protocol avoids importing the STEM package, whose initializer also
    loads unrelated scientific modules. The owner supplies a configured service.
    """

    def dimensional_signature(self, unit: Any) -> Mapping[str, float]: ...

    def compose(self, parts: Sequence[Any], operation: str = "mul") -> Any: ...

    def convert(self, value: float, source: Any, target: Any) -> float: ...


def quantity(value: float, unit: str, kind: str = "length") -> dict[str, Any]:
    """Create a validated format-boundary quantity (legacy-compatible DTO).

    ``unit`` is a length unit even for area and volume; ``kind`` determines
    the corresponding exponent. This is the on-disk SGEO/SDRW representation.
    """

    if kind not in _POWERS:
        raise ValidationError("unsupported physical quantity")
    return {"value": finite(value), "unit": require_unit(unit), "dimension": kind}


def _record(value: dict) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError("quantity must be a mapping")
    kind = value.get("dimension")
    if not isinstance(kind, str) or kind not in _POWERS:
        raise ValidationError("unknown quantity dimension")
    unit = value.get("unit")
    if not isinstance(unit, str):
        raise ValidationError("quantity unit must be a string")
    return quantity(value.get("value"), unit, kind)


def convert_quantity(record: dict, target_unit: str) -> dict[str, Any]:
    """Retain deterministic, agent-free conversion for existing file callers.

    This compatibility function uses the format's limited, validated length
    scale table. Scientific integrations should use ``STEMQuantityAdapter``.
    """

    item = _record(record)
    kind = item["dimension"]
    converted = _LOCAL_CONVERSIONS[kind](item["value"], item["unit"], target_unit)
    return quantity(converted, target_unit, kind)


@dataclass(frozen=True, slots=True)
class STEMQuantityAdapter:
    """Convert spatial DTOs with an injected, real STEM UnitSystem.

    The adapter never stores global agent state or installs a fallback solver.
    STEM unit definitions and dimensional analysis remain authoritative. A
    unit must be supported both by the spatial format and the supplied STEM
    registry; missing registry entries fail explicitly.
    """

    unit_system: STEMUnitService

    def __post_init__(self) -> None:
        if self.unit_system is None or any(
            not callable(getattr(self.unit_system, name, None))
            for name in ("dimensional_signature", "compose", "convert")
        ):
            raise TypeError("STEMQuantityAdapter requires a STEM unit service")

    def _require_length(self, unit: str) -> None:
        require_unit(unit)
        signature = dict(self.unit_system.dimensional_signature(unit))
        if signature != {"L": 1}:
            raise ValidationError(
                "spatial unit must have the STEM length dimension",
                context={"unit": unit, "dimension": signature},
            )

    def quantity(self, value: float, unit: str, kind: str = "length") -> dict[str, Any]:
        """Validate the DTO against both the format schema and STEM registry."""

        item = quantity(value, unit, kind)
        self._require_length(unit)
        return item

    def convert_quantity(self, record: dict, target_unit: str) -> dict[str, Any]:
        """Delegate conversion and dimensional algebra to STEM.

        For squared/cubed units the existing STEM ``compose`` creates the
        dimensional units. No independent scientific conversion engine or
        agent import is required inside the spatial-format package.
        """

        item = _record(record)
        self._require_length(item["unit"])
        self._require_length(target_unit)
        power = _POWERS[item["dimension"]]
        source = (
            item["unit"] if power == 1
            else self.unit_system.compose((item["unit"],) * power)
        )
        target = (
            target_unit if power == 1
            else self.unit_system.compose((target_unit,) * power)
        )
        result = finite(self.unit_system.convert(item["value"], source, target),
                        "converted STEM quantity")
        return quantity(result, target_unit, item["dimension"])


__all__ = ["STEMUnitService", "STEMQuantityAdapter", "quantity", "convert_quantity"]
