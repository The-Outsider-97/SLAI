"""Explicit monetary rounding and exact allocation.

The Financial subsystem never rounds implicitly. When a caller needs an amount
in a currency's smallest settlement unit (an invoice line, a cash rebalance), it
supplies a ``RoundingPolicy`` and this module quantizes to the currency's ISO
4217 minor units (see ``data.currency_catalog``). ``allocate`` splits an amount
by weights so that the parts sum EXACTLY to the whole, using the largest
remainder (Hamilton) method, which is the standard remedy for "penny lost"
errors when 1/3 of a currency amount is not representable.
"""
from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Sequence
from decimal import (ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_DOWN, ROUND_HALF_EVEN,
                     ROUND_HALF_UP, ROUND_UP, Decimal, localcontext)
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..data.currency_catalog import DEFAULT_CATALOG, CurrencyCatalog
from ..financial_types import FinancialAmount
from ..utils.financial_errors import *
from ..utils.financial_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Financial Rounding")
printer = PrettyPrinter()


class RoundingMode(str, Enum):
    """Rounding to the minor unit; ties only matter for the HALF_* modes."""

    HALF_UP = "half_up"          # ties away from zero
    HALF_EVEN = "half_even"      # banker's rounding (ties to even digit)
    HALF_DOWN = "half_down"      # ties toward zero
    UP = "up"                    # away from zero
    DOWN = "down"                # toward zero (truncate)
    FLOOR = "floor"              # toward negative infinity
    CEILING = "ceiling"          # toward positive infinity


_DECIMAL_MODES = {
    RoundingMode.HALF_UP: ROUND_HALF_UP, RoundingMode.HALF_EVEN: ROUND_HALF_EVEN,
    RoundingMode.HALF_DOWN: ROUND_HALF_DOWN, RoundingMode.UP: ROUND_UP,
    RoundingMode.DOWN: ROUND_DOWN, RoundingMode.FLOOR: ROUND_FLOOR,
    RoundingMode.CEILING: ROUND_CEILING,
}


@dataclass(frozen=True, slots=True)
class RoundingPolicy:
    """The caller's declared rounding mode (there is deliberately no default)."""

    mode: RoundingMode | str

    def __post_init__(self) -> None:
        if isinstance(self.mode, RoundingMode):
            return
        if not isinstance(self.mode, str):
            raise FinancialValidationError("Rounding mode must be a string")
        try:
            object.__setattr__(self, "mode", RoundingMode(self.mode.strip().lower()))
        except ValueError as exc:
            raise UnsupportedConventionError("Unsupported rounding mode", context={"mode": self.mode[:32]}) from exc

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode.value}  # type: ignore[union-attr]


class MoneyRounder:
    """Quantize and allocate currency amounts to ISO 4217 minor units."""

    def __init__(self, catalog: CurrencyCatalog | None = None):
        if catalog is not None and not isinstance(catalog, CurrencyCatalog):
            raise FinancialValidationError("catalog must be CurrencyCatalog or None")
        self.catalog = catalog if catalog is not None else DEFAULT_CATALOG

    def _exponent(self, amount: FinancialAmount) -> int:
        return self.catalog.resolve_minor_units(amount.currency)

    def quantize(self, amount: FinancialAmount, policy: RoundingPolicy) -> FinancialAmount:
        """Round to the currency's minor unit under ``policy``."""
        if not isinstance(amount, FinancialAmount):
            raise FinancialValidationError("amount must be FinancialAmount")
        if not isinstance(policy, RoundingPolicy):
            raise FinancialValidationError("policy must be RoundingPolicy")
        places = self._exponent(amount)
        step = Decimal(1).scaleb(-places)
        with localcontext() as ctx:
            ctx.prec = max(28, amount.amount.adjusted() + places + 4)
            rounded = amount.amount.quantize(step, rounding=_DECIMAL_MODES[policy.mode])  # type: ignore[index]
        return FinancialAmount(rounded, amount.currency)

    def is_representable(self, amount: FinancialAmount) -> bool:
        """True when the amount is already an exact multiple of the minor unit."""
        places = self._exponent(amount)
        return amount.amount == amount.amount.quantize(Decimal(1).scaleb(-places), rounding=ROUND_DOWN)

    def allocate(self, amount: FinancialAmount, weights: Sequence[Any]) -> tuple[FinancialAmount, ...]:
        """Split ``amount`` into parts proportional to ``weights``, summing exactly.

        The amount must be representable in minor units (quantize it first).
        Each part is the floor of its exact share in minor units; the leftover
        minor units go one each to the largest fractional remainders, ties to
        the earliest index. Zero weights always receive zero. Negative amounts
        are allocated by magnitude and negated.
        """
        if not isinstance(amount, FinancialAmount):
            raise FinancialValidationError("amount must be FinancialAmount")
        if not isinstance(weights, (list, tuple)) or not 1 <= len(weights) <= 4096:
            raise FinancialValidationError("weights must be a bounded nonempty list/tuple")
        ratios = [to_decimal(w, "weight") for w in weights]
        if any(w < 0 for w in ratios) or sum(ratios, Decimal(0)) <= 0:
            raise FinancialDomainError("Allocation weights must be nonnegative with a positive sum")
        if not self.is_representable(amount):
            raise FinancialDomainError("Quantize the amount to minor units before allocating")
        places = self._exponent(amount)
        scale = Decimal(1).scaleb(places)
        sign = -1 if amount.amount < 0 else 1
        with localcontext() as ctx:
            ctx.prec = max(60, amount.amount.adjusted() + places + 30)
            total_units = int(abs(amount.amount) * scale)
            weight_sum = sum(ratios, Decimal(0))
            exact = [Decimal(total_units) * w / weight_sum for w in ratios]
            base = [int(x) for x in exact]
            leftover = total_units - sum(base)
            order = sorted(range(len(ratios)), key=lambda i: (-(exact[i] - base[i]), i))
            for i in order:
                if leftover == 0:
                    break
                if ratios[i] > 0:
                    base[i] += 1
                    leftover -= 1
            if leftover:
                raise FinancialIntegrityError("Allocation could not distribute all minor units")
            parts = tuple(FinancialAmount(Decimal(sign * u) / scale, amount.currency) for u in base)
        return parts

    def split_evenly(self, amount: FinancialAmount, parts: int) -> tuple[FinancialAmount, ...]:
        require_positive_int(parts, "parts", maximum=4096)
        return self.allocate(amount, [1] * parts)


__all__ = ["RoundingMode", "RoundingPolicy", "MoneyRounder"]

if __name__ == "__main__":
    print("\n=== Running Financial Rounding ===\n")
    printer.status("TEST", "Financial Rounding initialized", "info")
    rounder = MoneyRounder()
    half_even, half_up = RoundingPolicy("half_even"), RoundingPolicy(RoundingMode.HALF_UP)
    assert rounder.quantize(FinancialAmount.of("2.345", "EUR"), half_even).amount == Decimal("2.34")
    assert rounder.quantize(FinancialAmount.of("2.345", "EUR"), half_up).amount == Decimal("2.35")
    assert rounder.quantize(FinancialAmount.of("-2.345", "EUR"), half_up).amount == Decimal("-2.35")
    assert rounder.quantize(FinancialAmount.of("1234.5", "JPY"), RoundingPolicy("floor")).amount == Decimal("1234")
    assert rounder.quantize(FinancialAmount.of("1.0005", "KWD"), RoundingPolicy("ceiling")).amount == Decimal("1.001")
    assert [p.amount for p in rounder.split_evenly(FinancialAmount.of("100.00", "EUR"), 3)] == [Decimal("33.34"), Decimal("33.33"), Decimal("33.33")]
    assert [p.amount for p in rounder.allocate(FinancialAmount.of("100", "JPY"), [1, 1, 1])] == [Decimal(34), Decimal(33), Decimal(33)]
    uneven = rounder.allocate(FinancialAmount.of("-0.05", "EUR"), [3, 7, 0])
    assert [p.amount for p in uneven] == [Decimal("-0.02"), Decimal("-0.03"), Decimal("0.00")]
    assert sum((p.amount for p in uneven), Decimal(0)) == Decimal("-0.05")
    for bad in (lambda: rounder.allocate(FinancialAmount.of("1.005", "EUR"), [1]),
                lambda: rounder.quantize(FinancialAmount.of("1", "XAU"), half_up),
                lambda: RoundingPolicy("banana")):
        try:
            bad()
        except FinancialError:
            pass
        else:
            raise AssertionError("invalid rounding use must be rejected")
    print("\n=== Test ran successfully ===\n")
