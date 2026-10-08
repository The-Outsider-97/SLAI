"""Structured SI-aware quantity and unit registry for SLAI STEM.

Conversions use ``base = value * scale + offset``. Affine units are therefore
safe for temperature conversion but prohibited in multiplicative unit algebra.
"""
from __future__ import annotations

__version__ = "2.3.0"

import math

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..stem_types import Dimension, Quantity, Unit
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.stem_errors import STEMIncommensurableUnitError, STEMPrefixError, STEMUnitError, STEMUnitSystemError
from ..utils.stem_helpers import ensure_finite_number, ensure_mapping, ensure_non_empty_string, ensure_positive, linear_unit_only
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("STEM Unit System")
printer = PrettyPrinter()


class UnitSystem:
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.unit_config = dict(get_config_section("unit_system", config=self.config) or {})
        if config:
            self.unit_config.update(dict(config))
        self.default_system = str(self.unit_config.get("default_system", "SI"))
        self.si_only = bool(self.unit_config.get("si_only", False))
        self.allow_non_si_with_conversion = bool(self.unit_config.get("allow_non_si_with_conversion", True))
        self._units: Dict[str, Unit] = {}
        self._prefixes: Dict[str, Tuple[str, float]] = {}
        self._initialize_prefixes()
        self._initialize_units()

    def _initialize_prefixes(self) -> None:
        entries = self.unit_config.get("prefixes") or []
        if not entries:
            entries = [
                {"symbol":"Q","name":"quetta","factor":1e30},{"symbol":"R","name":"ronna","factor":1e27},
                {"symbol":"Y","name":"yotta","factor":1e24},{"symbol":"Z","name":"zetta","factor":1e21},
                {"symbol":"E","name":"exa","factor":1e18},{"symbol":"P","name":"peta","factor":1e15},
                {"symbol":"T","name":"tera","factor":1e12},{"symbol":"G","name":"giga","factor":1e9},
                {"symbol":"M","name":"mega","factor":1e6},{"symbol":"k","name":"kilo","factor":1e3},
                {"symbol":"c","name":"centi","factor":1e-2},{"symbol":"m","name":"milli","factor":1e-3},
                {"symbol":"µ","name":"micro","factor":1e-6},{"symbol":"n","name":"nano","factor":1e-9},
            ]
        for item in entries:
            self.register_prefix(str(item["symbol"]), str(item["name"]), float(item["factor"]))

    def _initialize_units(self) -> None:
        base = self.unit_config.get("base_units") or {
            "m":{"name":"metre","dimension":{"L":1}}, "kg":{"name":"kilogram","dimension":{"M":1}},
            "s":{"name":"second","dimension":{"T":1}}, "A":{"name":"ampere","dimension":{"I":1}},
            "K":{"name":"kelvin","dimension":{"Θ":1}}, "mol":{"name":"mole","dimension":{"N":1}},
            "cd":{"name":"candela","dimension":{"J":1}},
        }
        for symbol, spec in base.items():
            self.register_base(str(symbol), str(spec.get("name", symbol)), Dimension(spec.get("dimension", {})))
        for symbol, spec in (self.unit_config.get("derived_units") or {}).items():
            self.register_derived(str(symbol), str(spec.get("name", symbol)), Dimension(spec.get("dimension", {})), float(spec.get("scale", 1.0)))
        for symbol, spec in (self.unit_config.get("affine_units") or {}).items():
            self.register_affine(str(symbol), str(spec.get("name", symbol)), Dimension(spec.get("dimension", {})), float(spec.get("scale", 1.0)), float(spec.get("offset", 0.0)))

    def register_base(self, symbol: str, name: str, dimension: Dimension) -> Unit:
        return self._register(symbol, name, dimension, 1.0, 0.0)

    def register_derived(self, symbol: str, name: str, dimension: Dimension, scale: float = 1.0) -> Unit:
        return self._register(symbol, name, dimension, scale, 0.0)

    def register_affine(self, symbol: str, name: str, dimension: Dimension, scale: float, offset: float) -> Unit:
        return self._register(symbol, name, dimension, scale, offset)

    def _register(self, symbol: str, name: str, dimension: Dimension, scale: float, offset: float) -> Unit:
        sym = ensure_non_empty_string(symbol, "unit symbol", error_cls=STEMUnitSystemError)
        if not isinstance(dimension, Dimension):
            raise STEMUnitSystemError("dimension must be a Dimension")
        unit = Unit(sym, ensure_non_empty_string(name, "unit name", error_cls=STEMUnitSystemError), dimension, ensure_positive(scale, "scale", error_cls=STEMUnitSystemError), ensure_finite_number(offset, "offset", error_cls=STEMUnitSystemError), self.default_system)
        existing = self._units.get(sym)
        if existing is not None and existing != unit:
            raise STEMUnitSystemError("Unit symbol already registered with different definition", context={"symbol": sym})
        self._units[sym] = unit
        return unit

    def lookup(self, symbol: str) -> Unit:
        sym = ensure_non_empty_string(symbol, "symbol", error_cls=STEMUnitError)
        if sym in self._units:
            return self._units[sym]
        return self.parse_prefixed(sym)

    def known_units(self) -> Mapping[str, Unit]:
        return dict(self._units)

    def register_prefix(self, symbol: str, name: str, factor: float) -> None:
        sym = ensure_non_empty_string(symbol, "prefix symbol", error_cls=STEMPrefixError)
        label = ensure_non_empty_string(name, "prefix name", error_cls=STEMPrefixError)
        f = ensure_positive(factor, "prefix factor", error_cls=STEMPrefixError)
        self._prefixes[sym] = (label, f)

    def apply_prefix(self, symbol: str, prefix: str) -> Unit:
        base = self.lookup(symbol)
        linear_unit_only(base, error_cls=STEMPrefixError)
        if prefix not in self._prefixes:
            raise STEMPrefixError("Unknown SI prefix", context={"prefix": prefix})
        name, factor = self._prefixes[prefix]
        # SI prefixes do not combine with kg; gram-based handling is kept explicit.
        if base.symbol == "kg":
            raise STEMPrefixError("Apply mass prefixes to gram-derived units explicitly, not kilogram")
        return Unit(prefix + base.symbol, f"{name}{base.name or base.symbol}", base.dimension, base.scale * factor, 0.0, base.system)

    def parse_prefixed(self, symbol: str) -> Unit:
        sym = ensure_non_empty_string(symbol, "symbol", error_cls=STEMPrefixError)
        for prefix in sorted(self._prefixes, key=len, reverse=True):
            if not sym.startswith(prefix) or len(sym) <= len(prefix):
                continue
            base_symbol = sym[len(prefix):]
            if base_symbol in self._units and not self._units[base_symbol].is_affine and base_symbol != "kg":
                return self.apply_prefix(base_symbol, prefix)
        raise STEMUnitError("Unknown unit", context={"symbol": sym})

    def format_prefixed(self, unit: Unit) -> str:
        if not isinstance(unit, Unit):
            raise STEMUnitError("unit must be a Unit")
        return unit.symbol

    def _resolve_unit(self, value: Any) -> Unit:
        if isinstance(value, Unit):
            return value
        if isinstance(value, str):
            return self.lookup(value)
        raise STEMUnitError("Expected unit or unit symbol", context={"type": type(value).__name__})

    def convert(self, value: float, source: Any, target: Any) -> float:
        src, dst = self._resolve_unit(source), self._resolve_unit(target)
        self.require_compatible(src, dst)
        return src.convert(ensure_finite_number(value, "value", error_cls=STEMUnitError), dst)

    def convert_quantity(self, quantity: Quantity, target: Any) -> Quantity:
        if not isinstance(quantity, Quantity):
            raise STEMUnitError("quantity must be Quantity")
        return quantity.convert(self._resolve_unit(target))

    def is_compatible(self, a: Any, b: Any) -> bool:
        return self._resolve_unit(a).dimension == self._resolve_unit(b).dimension

    def require_compatible(self, a: Any, b: Any, context: str = "conversion") -> None:
        left, right = self._resolve_unit(a), self._resolve_unit(b)
        if left.dimension != right.dimension:
            raise STEMIncommensurableUnitError("Units are not dimensionally compatible", context={"operation": context, "left": left.symbol, "right": right.symbol})

    def to_base(self, value: float, unit: Any) -> float:
        return self._resolve_unit(unit).to_base(value)

    def from_base(self, value: float, unit: Any) -> float:
        return self._resolve_unit(unit).from_base(value)

    def normalize(self, unit: Any) -> Unit:
        source = self._resolve_unit(unit)
        candidates = [item for item in self._units.values() if item.dimension == source.dimension and item.scale == 1.0 and item.offset == 0.0]
        if candidates:
            return sorted(candidates, key=lambda item: (len(item.symbol), item.symbol))[0]
        return Unit(f"SI[{source.dimension}]", dimension=source.dimension, scale=1.0, offset=0.0, system="SI")

    def compose(self, parts: Sequence[Any], operation: str = "mul") -> Unit:
        if not parts:
            return Unit("1", "dimensionless", Dimension(), 1.0, 0.0, self.default_system)
        resolved = [self._resolve_unit(part) for part in parts]
        if any(unit.is_affine for unit in resolved):
            raise STEMUnitError("Affine units cannot be composed")
        result = resolved[0]
        if operation == "mul":
            for unit in resolved[1:]:
                result = result * unit
        elif operation == "div":
            for unit in resolved[1:]:
                result = result / unit
        else:
            raise STEMUnitError("operation must be 'mul' or 'div'")
        return result

    def is_affine(self, unit: Any) -> bool:
        return self._resolve_unit(unit).is_affine

    def dimensional_signature(self, unit: Any) -> Mapping[str, float]:
        return dict(self._resolve_unit(unit).dimension.exponents)

    def set_si_only(self, flag: bool) -> None:
        self.si_only = bool(flag)

    def accepts(self, unit: Any) -> bool:
        try:
            resolved = self._resolve_unit(unit)
        except STEMUnitError:
            return False
        return not self.si_only or resolved.system == "SI"

    def to_dict(self) -> Mapping[str, Any]:
        return {"default_system": self.default_system, "si_only": self.si_only, "units": {symbol: unit.to_dict() for symbol, unit in self._units.items()}, "prefixes": {symbol: {"name": name, "factor": factor} for symbol, (name, factor) in self._prefixes.items()}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UnitSystem":
        instance = cls({"base_units": {}, "derived_units": {}, "affine_units": {}, "prefixes": []})
        instance.default_system = str(data.get("default_system", "SI"))
        instance.si_only = bool(data.get("si_only", False))
        for symbol, spec in (data.get("prefixes") or {}).items():
            instance.register_prefix(str(symbol), str(spec["name"]), float(spec["factor"]))
        for symbol, spec in (data.get("units") or {}).items():
            unit = Unit.from_dict(spec)
            instance._units[str(symbol)] = unit
        return instance


__all__ = ["UnitSystem"]
