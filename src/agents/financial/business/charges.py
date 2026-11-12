"""Itemized price construction: discounts, fees and taxes on a base amount.

A deterministic, jurisdiction-neutral calculator. It applies caller-declared
charges in a fixed, documented order and returns every line so an invoice or a
cost estimate can be audited. It does not know tax law: tax rates, applicability
and place-of-supply rules are inputs, never inferred.

Order of application (each stage uses the running subtotal):
  1. DISCOUNT_PERCENT  compound sequentially in the order given (a 10% then a
                       20% discount leave 72%, not 70%);
  2. DISCOUNT_AMOUNT   subtracted; total discount can never push the price
                       below zero (the line shows the amount actually applied);
  3. FEE_PERCENT / FEE_AMOUNT  added on the discounted subtotal;
  4. TAX_PERCENT       exclusive tax on (discounted subtotal + fees).
Optional ``active_from`` / ``active_until`` bound a charge to a half-open date
window [from, until). With a ``RoundingPolicy`` every line is rounded to the
currency's minor unit first and the total is the sum of the rounded lines, so a
printed invoice always adds up (line-level rounding).
"""
from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext
from enum import Enum
from typing import Any

from .rounding import MoneyRounder, RoundingPolicy
from ..financial_types import FinancialAmount
from ..utils.financial_errors import *
from ..utils.financial_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Financial Charges")
printer = PrettyPrinter()


class ChargeKind(str, Enum):
    DISCOUNT_PERCENT = "discount_percent"
    DISCOUNT_AMOUNT = "discount_amount"
    FEE_PERCENT = "fee_percent"
    FEE_AMOUNT = "fee_amount"
    TAX_PERCENT = "tax_percent"


_PERCENT = {ChargeKind.DISCOUNT_PERCENT, ChargeKind.FEE_PERCENT, ChargeKind.TAX_PERCENT}
_STAGE = {ChargeKind.DISCOUNT_PERCENT: 1, ChargeKind.DISCOUNT_AMOUNT: 2,
          ChargeKind.FEE_PERCENT: 3, ChargeKind.FEE_AMOUNT: 3, ChargeKind.TAX_PERCENT: 4}


@dataclass(frozen=True, slots=True)
class Charge:
    """One declared adjustment. Percent kinds take a fraction (0.2 = 20%)."""

    name: str
    kind: ChargeKind | str
    value: Decimal
    active_from: date | None = None
    active_until: date | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_text(self.name, "name", max_length=128))
        kind = self.kind if isinstance(self.kind, ChargeKind) else None
        if kind is None:
            try:
                kind = ChargeKind(str(self.kind).strip().lower())
            except ValueError as exc:
                raise UnsupportedConventionError("Unsupported charge kind", context={"kind": str(self.kind)[:32]}) from exc
        object.__setattr__(self, "kind", kind)
        value = to_decimal(self.value, "value")
        if value < 0:
            raise FinancialDomainError("Charge value must be nonnegative")
        if kind in _PERCENT and value > 1:
            raise FinancialDomainError("Percent charges are fractions in [0, 1]")
        object.__setattr__(self, "value", value)
        start = None if self.active_from is None else to_date(self.active_from, "active_from")
        end = None if self.active_until is None else to_date(self.active_until, "active_until")
        if start is not None and end is not None and end <= start:
            raise FinancialDomainError("active_until must follow active_from")
        object.__setattr__(self, "active_from", start)
        object.__setattr__(self, "active_until", end)

    @classmethod
    def from_dict(cls, data: Any) -> "Charge":
        if not isinstance(data, dict) and not hasattr(data, "get"):
            raise FinancialValidationError("Charge payload must be a mapping")
        missing = [k for k in ("name", "kind", "value") if k not in data]
        if missing:
            raise FinancialValidationError("Charge payload is missing required fields", context={"missing_fields": missing})
        return cls(data["name"], data["kind"], data["value"], data.get("active_from"), data.get("active_until"))

    def is_active(self, on: date | None) -> bool:
        if on is None:
            return self.active_from is None and self.active_until is None
        return (self.active_from is None or self.active_from <= on) and (self.active_until is None or on < self.active_until)

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class ChargeLine:
    name: str
    kind: ChargeKind
    amount: FinancialAmount  # signed: discounts negative, fees and taxes positive

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class ChargeBreakdown:
    base: FinancialAmount
    lines: tuple[ChargeLine, ...]
    subtotal: FinancialAmount   # base + discounts + fees (before tax)
    tax: FinancialAmount
    total: FinancialAmount
    rounded: bool

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


def apply_charges(
    base: FinancialAmount, charges: Sequence[Charge] = (), *, on: date | str | None = None,
    precision: int = 36, policy: RoundingPolicy | None = None, rounder: MoneyRounder | None = None,
) -> ChargeBreakdown:
    """Itemize ``base`` under ``charges`` as of ``on`` (see module docstring)."""
    if not isinstance(base, FinancialAmount) or base.amount < 0:
        raise FinancialDomainError("Charge base must be a nonnegative FinancialAmount")
    if not isinstance(charges, (list, tuple)) or len(charges) > 256 or any(not isinstance(c, Charge) for c in charges):
        raise FinancialValidationError("charges must be a bounded list/tuple of Charge values")
    require_positive_int(precision, "precision", minimum=16, maximum=256)
    if policy is not None and not isinstance(policy, RoundingPolicy):
        raise FinancialValidationError("policy must be RoundingPolicy or None")
    day = None if on is None else to_date(on, "on")
    round_ = rounder if rounder is not None else MoneyRounder()
    code = base.currency
    zero = Decimal(0)

    def fit(value: Decimal) -> FinancialAmount:
        amount = FinancialAmount(value, code)
        return round_.quantize(amount, policy) if policy is not None else amount

    active = sorted((c for c in charges if c.is_active(day)), key=lambda c: _STAGE[c.kind])  # type: ignore[index]
    lines: list[ChargeLine] = []
    with localcontext() as ctx:
        ctx.prec = precision
        running = fit(base.amount).amount
        for charge in active:
            kind: ChargeKind = charge.kind  # type: ignore[assignment]
            if _STAGE[kind] == 4:
                continue
            if kind is ChargeKind.DISCOUNT_PERCENT:
                delta = -(running * charge.value)
            elif kind is ChargeKind.DISCOUNT_AMOUNT:
                delta = -min(charge.value, running)
            elif kind is ChargeKind.FEE_PERCENT:
                delta = running * charge.value
            else:
                delta = charge.value
            line = fit(delta)
            if kind in (ChargeKind.DISCOUNT_PERCENT, ChargeKind.DISCOUNT_AMOUNT) and line.amount < -running:
                line = fit(-running)
            lines.append(ChargeLine(charge.name, kind, line))
            running += line.amount
        subtotal = running
        tax = zero
        for charge in active:
            if charge.kind is ChargeKind.TAX_PERCENT:
                line = fit(subtotal * charge.value)
                lines.append(ChargeLine(charge.name, ChargeKind.TAX_PERCENT, line))
                tax += line.amount
        total = subtotal + tax
    return ChargeBreakdown(
        FinancialAmount(fit(base.amount).amount, code), tuple(lines), FinancialAmount(subtotal, code),
        FinancialAmount(tax, code), FinancialAmount(total, code), policy is not None)


def split_inclusive_tax(gross: FinancialAmount, rate: Any, *, precision: int = 36,
                        policy: RoundingPolicy | None = None, rounder: MoneyRounder | None = None) -> tuple[FinancialAmount, FinancialAmount]:
    """Decompose a tax-inclusive price into (net, tax): net = gross / (1 + rate).

    With a policy the tax is rounded and net is gross - tax so the pair sums
    to the printed gross amount exactly.
    """
    if not isinstance(gross, FinancialAmount):
        raise FinancialValidationError("gross must be FinancialAmount")
    fraction = to_decimal(rate, "rate")
    if fraction < 0 or fraction > 10:
        raise FinancialDomainError("Tax rate must be a nonnegative fraction")
    round_ = rounder if rounder is not None else MoneyRounder()
    with localcontext() as ctx:
        ctx.prec = precision
        net = FinancialAmount(gross.amount / (1 + fraction), gross.currency)
    tax = gross - net
    if policy is not None:
        tax = round_.quantize(tax, policy)
        net = gross - tax
    return net, tax


__all__ = ["ChargeKind", "Charge", "ChargeLine", "ChargeBreakdown", "apply_charges", "split_inclusive_tax"]

if __name__ == "__main__":
    print("\n=== Running Financial Charges ===\n")
    printer.status("TEST", "Financial Charges initialized", "info")
    base = FinancialAmount.of("100", "EUR")
    bundle = (Charge("tax", "tax_percent", "0.21"), Charge("promo", "discount_percent", "0.10"),
              Charge("loyalty", "discount_percent", "0.20"), Charge("setup", "fee_amount", "5"),
              Charge("coupon", "discount_amount", "2"))
    result = apply_charges(base, bundle)
    assert [line.name for line in result.lines] == ["promo", "loyalty", "coupon", "setup", "tax"]
    assert result.subtotal.amount == Decimal("75.00") and result.tax.amount == Decimal("15.75")
    assert result.total.amount == Decimal("90.75") and not result.rounded
    cap = apply_charges(base, [Charge("big", "discount_amount", "500")])
    assert cap.total.amount == 0 and cap.lines[0].amount.amount == -100
    window = Charge("intro", "discount_percent", "0.5", active_until=date(2024, 4, 1))
    assert apply_charges(base, [window], on="2024-03-31").total.amount == 50
    assert apply_charges(base, [window], on="2024-04-01").total.amount == 100
    assert apply_charges(base, [window]).total.amount == 100, "date-bounded charges need an explicit date"
    odd = apply_charges(FinancialAmount.of("9.99", "EUR"), [Charge("vat", "tax_percent", "0.21"), Charge("d", "discount_percent", "0.15")],
                        policy=RoundingPolicy("half_up"))
    assert [line.amount.amount for line in odd.lines] == [Decimal("-1.50"), Decimal("1.78")]
    assert odd.total == odd.subtotal + odd.tax and odd.total.amount == Decimal("10.27")
    net, tax = split_inclusive_tax(FinancialAmount.of("121", "EUR"), "0.21")
    assert net.amount == 100 and tax.amount == 21
    n2, t2 = split_inclusive_tax(FinancialAmount.of("9.99", "EUR"), "0.21", policy=RoundingPolicy("half_up"))
    assert n2 + t2 == FinancialAmount.of("9.99", "EUR")
    for bad in (lambda: Charge("x", "discount_percent", "1.5"), lambda: Charge("x", "mystery", "1"),
                lambda: apply_charges(FinancialAmount.of("-1", "EUR"))):
        try:
            bad()
        except FinancialError:
            pass
        else:
            raise AssertionError("invalid charge input must be rejected")
    print("\n=== Test ran successfully ===\n")
