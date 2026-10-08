"""
SI unit registry, prefix engine, and conversion policy for the STEM subsystem.

Ownership
- base units;
- derived units;
- prefixes;
- scale conversion;
- affine conversion where relevant;
- unit composition;
- unit normalization;
- dimension compatibility against a unit registry;
- SI/non-SI conversion policy.

This module does NOT own dimension algebra. Dimension algebra is delegated to
``stem_types.Dimension`` and the shared helpers in ``stem_helpers.py``.

Sources
- BIPM. (2026). The International System of Units (SI), 9th ed., version 4.01.
  DOI 10.59161/AUEZ1291.
- ISO 80000-1:2022.
"""

from __future__ import annotations

__version__ = "2.3.0"

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.stem_errors import *
from ..utils.stem_helpers import *
from ..stem_types import Dimension, Quantity, Unit
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Unit System")
printer = PrettyPrinter()


_DEFAULT_PREFIXES: Tuple[Tuple[str, str, float], ...] = (
    ("Q", "quetta", 1e30),
    ("R", "ronna", 1e27),
    ("Y", "yotta", 1e24),
    ("Z", "zetta", 1e21),
    ("E", "exa", 1e18),
    ("P", "peta", 1e15),
    ("T", "tera", 1e12),
    ("G", "giga", 1e9),
    ("M", "mega", 1e6),
    ("k", "kilo", 1e3),
    ("h", "hecto", 1e2),
    ("da", "deca", 1e1),
    ("d", "deci", 1e-1),
    ("c", "centi", 1e-2),
    ("m", "milli", 1e-3),
    ("µ", "micro", 1e-6),
    ("n", "nano", 1e-9),
    ("p", "pico", 1e-12),
    ("f", "femto", 1e-15),
    ("a", "atto", 1e-18),
    ("z", "zepto", 1e-21),
    ("y", "yocto", 1e-24),
    ("r", "ronto", 1e-27),
    ("q", "quecto", 1e-30),
)


_DEFAULT_BASE_UNITS: Tuple[Tuple[str, str, Dict[str, float]], ...] = (
    ("m", "metre", {"L": 1.0}),
    ("kg", "kilogram", {"M": 1.0}),
    ("s", "second", {"T": 1.0}),
    ("A", "ampere", {"I": 1.0}),
    ("K", "kelvin", {"Θ": 1.0}),
    ("mol", "mole", {"N": 1.0}),
    ("cd", "candela", {"J": 1.0}),
)


_DEFAULT_DERIVED_UNITS: Tuple[Tuple[str, str, Dict[str, float], float], ...] = (
    ("Hz", "hertz", {"T": -1.0}, 1.0),
    ("N", "newton", {"M": 1.0, "L": 1.0, "T": -2.0}, 1.0),
    ("Pa", "pascal", {"M": 1.0, "L": -1.0, "T": -2.0}, 1.0),
    ("J", "joule", {"M": 1.0, "L": 2.0, "T": -2.0}, 1.0),
    ("W", "watt", {"M": 1.0, "L": 2.0, "T": -3.0}, 1.0),
    ("C", "coulomb", {"I": 1.0, "T": 1.0}, 1.0),
    ("V", "volt", {"M": 1.0, "L": 2.0, "T": -3.0, "I": -1.0}, 1.0),
    ("F", "farad", {"M": -1.0, "L": -2.0, "T": 4.0, "I": 2.0}, 1.0),
    ("Ω", "ohm", {"M": 1.0, "L": 2.0, "T": -3.0, "I": -2.0}, 1.0),
    ("S", "siemens", {"M": -1.0, "L": -2.0, "T": 3.0, "I": 2.0}, 1.0),
    ("Wb", "weber", {"M": 1.0, "L": 2.0, "T": -2.0, "I": -1.0}, 1.0),
    ("T", "tesla", {"M": 1.0, "T": -2.0, "I": -1.0}, 1.0),
    ("H", "henry", {"M": 1.0, "L": 2.0, "T": -2.0, "I": -2.0}, 1.0),
    ("lm", "lumen", {"J": 1.0}, 1.0),
    ("lx", "lux", {"J": 1.0, "L": -2.0}, 1.0),
    ("Bq", "becquerel", {"T": -1.0}, 1.0),
    ("Gy", "gray", {"L": 2.0, "T": -2.0}, 1.0),
    ("Sv", "sievert", {"L": 2.0, "T": -2.0}, 1.0),
    ("kat", "katal", {"N": 1.0, "T": -1.0}, 1.0),
)


_DEFAULT_AFFINE_UNITS: Tuple[Tuple[str, str, Dict[str, float], float, float], ...] = (
    ("°C", "degree_celsius", {"Θ": 1.0}, 1.0, 273.15),
    ("°F", "degree_fahrenheit", {"Θ": 1.0}, 5.0 / 9.0, 255.3722222222222),
)


class UnitSystem:
    """
    SI unit registry, prefix engine, and conversion policy layer.

    Stores units structurally and delegates dimension algebra to
    :class:`~stem_types.Dimension`. Affine units are never prefixed,
    composed, or exponentiated; they must first be converted to a linear
    base representation.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = load_global_config()
        self.us_config = dict(get_config_section("unit_system", config=self.config) or {})
        if config:
            self.us_config.update(dict(config))

        self._default_system = str(self.us_config.get("default_system", "SI"))
        self._si_only = bool(self.us_config.get("si_only", False))
        self._allow_non_si = bool(self.us_config.get("allow_non_si_with_conversion", True))

        self._units: Dict[str, Unit] = {}
        self._prefixes: Dict[str, Tuple[str, float]] = {}

        self._initialize_prefixes()
        self._initialize_units()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_prefixes(self) -> None:
        cfg_prefixes = self.us_config.get("prefixes") or []
        if cfg_prefixes:
            for entry in cfg_prefixes:
                if isinstance(entry, Mapping):
                    symbol = str(entry.get("symbol", "")).strip()
                    name = str(entry.get("name", "")).strip()
                    factor = float(entry.get("factor", 1.0))
                    if symbol:
                        self.register_prefix(symbol, name or symbol, factor)
        if not self._prefixes:
            for symbol, name, factor in _DEFAULT_PREFIXES:
                self.register_prefix(symbol, name, factor)

    def _initialize_units(self) -> None:
        base_cfg = self.us_config.get("base_units") or {}
        if base_cfg:
            for symbol, spec in base_cfg.items():
                if not isinstance(spec, Mapping):
                    continue
                dimension = Dimension(dict(spec.get("dimension") or {}))
                self.register_base(str(symbol), str(spec.get("name", symbol)), dimension)
        else:
            for symbol, name, dim in _DEFAULT_BASE_UNITS:
                self.register_base(symbol, name, Dimension(dim))

        derived_cfg = self.us_config.get("derived_units") or {}
        if derived_cfg:
            for symbol, spec in derived_cfg.items():
                if not isinstance(spec, Mapping):
                    continue
                dimension = Dimension(dict(spec.get("dimension") or {}))
                scale = float(spec.get("scale", 1.0))
                self.register_derived(str(symbol), str(spec.get("name", symbol)), dimension, scale)
        else:
            for symbol, name, dim, scale in _DEFAULT_DERIVED_UNITS:
                self.register_derived(symbol, name, Dimension(dim), scale)

        affine_cfg = self.us_config.get("affine_units") or {}
        if affine_cfg:
            for symbol, spec in affine_cfg.items():
                if not isinstance(spec, Mapping):
                    continue
                dimension = Dimension(dict(spec.get("dimension") or {}))
                scale = float(spec.get("scale", 1.0))
                offset = float(spec.get("offset", 0.0))
                self.register_affine(
                    str(symbol), str(spec.get("name", symbol)), dimension, scale, offset,
                )
        else:
            for symbol, name, dim, scale, offset in _DEFAULT_AFFINE_UNITS:
                self.register_affine(symbol, name, Dimension(dim), scale, offset)

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def register_base(self, symbol: str, name: str, dimension: Dimension) -> Unit:
        """Register a base unit (scale 1.0, offset 0.0)."""
        return self._register(symbol, name, dimension, 1.0, 0.0)

    def register_derived(
        self,
        symbol: str,
        name: str,
        dimension: Dimension,
        scale: float = 1.0,
    ) -> Unit:
        """Register a derived linear unit."""
        return self._register(symbol, name, dimension, scale, 0.0)

    def register_affine(
        self,
        symbol: str,
        name: str,
        dimension: Dimension,
        scale: float,
        offset: float,
    ) -> Unit:
        """Register an affine unit (scale and offset relative to base)."""
        return self._register(symbol, name, dimension, scale, offset)

    def _register(
        self,
        symbol: str,
        name: str,
        dimension: Dimension,
        scale: float,
        offset: float,
    ) -> Unit:
        sym = ensure_non_empty_string(symbol, "symbol", error_cls=STEMUnitSystemError)
        nm = ensure_non_empty_string(name, "name", error_cls=STEMUnitSystemError)
        if not isinstance(dimension, Dimension):
            raise STEMUnitSystemError(
                "dimension must be a Dimension instance",
                context={"received_type": type(dimension).__name__},
            )
        scale_f = ensure_positive(scale, "scale", allow_zero=False, error_cls=STEMUnitSystemError)
        unit = Unit(
            symbol=sym,
            name=nm,
            dimension=dimension,
            scale=scale_f,
            offset=float(offset),
            system=self._default_system,
        )
        self._units[sym] = unit
        return unit

    def lookup(self, symbol: str) -> Unit:
        """Retrieve a unit by symbol; falls back to prefixed parsing."""
        sym = ensure_non_empty_string(symbol, "symbol", error_cls=STEMUnitError)
        if sym in self._units:
            return self._units[sym]
        return self.parse_prefixed(sym)

    def known_units(self) -> Mapping[str, Unit]:
        """Return the registered unit mapping as a copy."""
        return dict(self._units)

    # ------------------------------------------------------------------
    # Prefixes
    # ------------------------------------------------------------------

    def register_prefix(self, symbol: str, name: str, factor: float) -> None:
        """Register an SI prefix symbol with its decimal factor."""
        sym = ensure_non_empty_string(symbol, "symbol", error_cls=STEMPrefixError)
        nm = ensure_non_empty_string(name, "name", error_cls=STEMPrefixError)
        factor_f = ensure_positive(factor, "factor", allow_zero=False, error_cls=STEMPrefixError)
        self._prefixes[sym] = (nm, factor_f)

    def apply_prefix(self, symbol: str, prefix: str) -> Unit:
        """Construct a prefixed unit from a linear base unit."""
        base = self.lookup(symbol)
        if base.is_affine:
            raise STEMPrefixError(
                "affine units cannot take an SI prefix",
                context={"symbol": base.symbol},
            )
        if prefix not in self._prefixes:
            raise STEMPrefixError(
                "unknown SI prefix",
                context={"prefix": prefix, "known": sorted(self._prefixes.keys())},
            )
        prefix_name, factor = self._prefixes[prefix]
        return Unit(
            symbol=f"{prefix}{base.symbol}",
            name=f"{prefix_name}{base.name}" if base.name else None,
            dimension=base.dimension,
            scale=base.scale * factor,
            offset=0.0,
            system=base.system,
        )

    def parse_prefixed(self, symbol: str) -> Unit:
        """Parse a prefixed symbol like ``km``, ``µs``, or ``GHz``."""
        sym = ensure_non_empty_string(symbol, "symbol", error_cls=STEMUnitError)
        if sym in self._units:
            return self._units[sym]
        prefix, remainder = parse_si_prefix(sym, self._prefixes)
        if prefix is None:
            raise STEMUnitError(
                "unknown unit symbol",
                context={"symbol": sym},
            )
        if remainder not in self._units:
            raise STEMUnitError(
                "unknown base unit after prefix",
                context={"symbol": sym, "prefix": prefix, "remainder": remainder},
            )
        base = self._units[remainder]
        if base.is_affine:
            raise STEMPrefixError(
                "affine units cannot be prefixed",
                context={"symbol": remainder, "prefix": prefix},
            )
        prefix_name, factor = self._prefixes[prefix]
        return Unit(
            symbol=sym,
            name=f"{prefix_name}{base.name}" if base.name else None,
            dimension=base.dimension,
            scale=base.scale * factor,
            offset=0.0,
            system=base.system,
        )

    def format_prefixed(self, unit: Unit) -> str:
        """Return the shortest standard SI representation of a unit."""
        if not isinstance(unit, Unit):
            raise STEMUnitError("format_prefixed requires a Unit")
        if unit.is_affine:
            return unit.symbol
        if unit.symbol in self._units and self._units[unit.symbol].scale == unit.scale:
            return unit.symbol
        # Try to find the closest SI prefix for the unit's scale
        best_prefix = ""
        best_error = float("inf")
        for prefix, (_, factor) in self._prefixes.items():
            error = abs(unit.scale / factor - 1.0)
            if error < best_error:
                best_error = error
                best_prefix = prefix
        if not best_prefix:
            return unit.symbol
        return f"{best_prefix}{unit.symbol}"

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _resolve_unit(self, value: Any) -> Unit:
        if isinstance(value, Unit):
            return value
        if isinstance(value, str):
            return self.lookup(value)
        raise STEMUnitSystemError(
            "expected Unit or symbol string",
            context={"received_type": type(value).__name__},
        )

    def convert(self, value: float, source: Any, target: Any) -> float:
        """Convert a scalar value between two units."""
        src = self._resolve_unit(source)
        tgt = self._resolve_unit(target)
        if src.dimension != tgt.dimension:
            raise STEMUnitConversionError(
                "incommensurable dimensions for conversion",
                context={
                    "source": src.symbol,
                    "target": tgt.symbol,
                    "source_dim": src.dimension.to_dict(),
                    "target_dim": tgt.dimension.to_dict(),
                },
            )
        return affine_safe_convert(float(value), src, tgt, error_cls=STEMUnitConversionError)

    def convert_quantity(self, quantity: Quantity, target: Any) -> Quantity:
        """Convert a quantity to the target unit."""
        if not isinstance(quantity, Quantity):
            raise STEMUnitConversionError(
                "convert_quantity requires a Quantity",
                context={"received_type": type(quantity).__name__},
            )
        tgt = self._resolve_unit(target)
        return quantity.convert(tgt)

    def is_compatible(self, a: Any, b: Any) -> bool:
        """Return True when the two units share the same dimension."""
        try:
            return self._resolve_unit(a).dimension == self._resolve_unit(b).dimension
        except STEMUnitError:
            return False

    def require_compatible(self, a: Any, b: Any, context: str = "conversion") -> None:
        """Raise ``STEMIncommensurableUnitError`` when units differ dimensionally."""
        ua = self._resolve_unit(a)
        ub = self._resolve_unit(b)
        if ua.dimension != ub.dimension:
            raise STEMIncommensurableUnitError(
                f"incommensurable units in {context}",
                context={
                    "a": ua.symbol,
                    "b": ub.symbol,
                    "a_dim": ua.dimension.to_dict(),
                    "b_dim": ub.dimension.to_dict(),
                    "operation": context,
                },
            )

    def to_base(self, value: float, unit: Any) -> float:
        """Convert a value in the given unit to its base representation."""
        u = self._resolve_unit(unit)
        return u.to_base(float(value))

    def from_base(self, value: float, unit: Any) -> float:
        """Convert a base-unit value into the given unit."""
        u = self._resolve_unit(unit)
        return u.from_base(float(value))

    def normalize(self, unit: Any) -> Unit:
        """Return the coherent (scale 1.0, offset 0.0) form of a unit."""
        u = self._resolve_unit(unit)
        if u.scale == 1.0 and u.offset == 0.0:
            return u
        return Unit(
            symbol=u.symbol,
            name=u.name,
            dimension=u.dimension,
            scale=1.0,
            offset=0.0,
            system=u.system,
        )

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(self, parts: Sequence[Any], operation: str = "mul") -> Unit:
        """Combine a sequence of linear units via multiplication or division."""
        seq = ensure_sequence(parts, "parts", error_cls=STEMUnitSystemError)
        if not seq:
            raise STEMUnitSystemError("parts must not be empty")
        units = [self._resolve_unit(part) for part in seq]
        for unit in units:
            if unit.is_affine:
                raise STEMUnitSystemError(
                    "affine units cannot be composed",
                    context={"symbol": unit.symbol},
                )
        if operation == "mul":
            result = units[0]
            for unit in units[1:]:
                result = result * unit
            return result
        if operation == "div":
            result = units[0]
            for unit in units[1:]:
                result = result / unit
            return result
        raise STEMUnitSystemError(
            "unsupported compose operation",
            context={"operation": operation, "allowed": ["mul", "div"]},
        )

    def is_affine(self, unit: Any) -> bool:
        """Return True when the unit has a nonzero offset."""
        return self._resolve_unit(unit).is_affine

    def dimensional_signature(self, unit: Any) -> Mapping[str, float]:
        """Return the dimension exponents of a unit as a plain mapping."""
        return dict(self._resolve_unit(unit).dimension.exponents)

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def set_si_only(self, flag: bool) -> None:
        """Set whether the system enforces SI-only acceptance."""
        self._si_only = bool(flag)

    def accepts(self, unit: Any) -> bool:
        """Return True when the current policy accepts the given unit."""
        try:
            u = self._resolve_unit(unit)
        except STEMUnitError:
            return False
        if not self._si_only:
            return True
        return u.system == "SI"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Mapping[str, Any]:
        """Return a JSON-safe representation of the registry and policy."""
        return {
            "default_system": self._default_system,
            "si_only": self._si_only,
            "allow_non_si_with_conversion": self._allow_non_si,
            "units": {sym: unit.to_dict() for sym, unit in self._units.items()},
            "prefixes": {
                sym: {"name": name, "factor": factor}
                for sym, (name, factor) in self._prefixes.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UnitSystem":
        """Rebuild a ``UnitSystem`` instance from a serialized mapping."""
        ensure_mapping(data, "data", error_cls=STEMUnitSystemError)
        instance = cls()
        for symbol, spec in (data.get("units") or {}).items():
            if not isinstance(spec, Mapping):
                continue
            instance._register(
                symbol=str(symbol),
                name=str(spec.get("name") or symbol),
                dimension=Dimension(dict(spec.get("dimension") or {})),
                scale=float(spec.get("scale", 1.0)),
                offset=float(spec.get("offset", 0.0)),
            )
        for symbol, spec in (data.get("prefixes") or {}).items():
            if isinstance(spec, Mapping):
                instance.register_prefix(
                    str(symbol),
                    str(spec.get("name") or symbol),
                    float(spec.get("factor", 1.0)),
                )
        instance._si_only = bool(data.get("si_only", instance._si_only))
        instance._allow_non_si = bool(
            data.get("allow_non_si_with_conversion", instance._allow_non_si)
        )
        return instance


__all__ = ["UnitSystem"]