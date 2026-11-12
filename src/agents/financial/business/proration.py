"""Day-based proration of recurring charges within a billing period.

The fraction of a billing period is counted in whole calendar days over
half-open intervals: ``used_days / period_days``. Calendar-day proration is the
common billing convention but is a *contractual* choice (some providers charge
per full period or per 30-day month); this module offers only the exact
actual-days convention and never infers a provider's policy. Results are exact
to the supplied calculation precision and unrounded; apply an explicit
``RoundingPolicy`` (business.rounding) for invoice amounts.

Three standard adjustments are provided:
  * partial period charge   - price x (days of service / period days);
  * cancellation credit     - price x (unused days / period days), a positive
                              amount owed back to the customer;
  * plan change adjustment  - (new - old) price x (remaining days / period days),
                              positive means an extra charge.
"""
from __future__ import annotations

__version__ = "2.4.0"

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext
from typing import Any

from ..financial_types import FinancialAmount, FinancialPeriod
from ..utils.financial_errors import *
from ..utils.financial_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Financial Proration")
printer = PrettyPrinter()


@dataclass(frozen=True, slots=True)
class ProrationResult:
    amount: FinancialAmount
    used_days: int
    period_days: int
    fraction: Decimal

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


def period_fraction(period: FinancialPeriod, start: date | str, end: date | str, *, precision: int = 36) -> tuple[int, int, Decimal]:
    """(days in [start, end), days in period, fraction) for a sub-interval of ``period``."""
    if not isinstance(period, FinancialPeriod):
        raise FinancialValidationError("period must be FinancialPeriod")
    require_positive_int(precision, "precision", minimum=16, maximum=256)
    begin, finish = to_date(start, "start"), to_date(end, "end")
    if begin > finish:
        raise FinancialDomainError("Proration interval must not end before it starts")
    if begin < period.start or finish > period.end:
        raise FinancialDomainError("Proration interval must lie within the billing period")
    total = (period.end - period.start).days
    used = (finish - begin).days
    with localcontext() as ctx:
        ctx.prec = precision
        return used, total, Decimal(used) / Decimal(total)


def _scaled(price: FinancialAmount, days: int, total: int, precision: int) -> FinancialAmount:
    with localcontext() as ctx:
        ctx.prec = precision
        return FinancialAmount(price.amount * Decimal(days) / Decimal(total), price.currency)


def partial_period_charge(price: FinancialAmount, period: FinancialPeriod, *, service_start: date | str,
                          service_end: date | str | None = None, precision: int = 36) -> ProrationResult:
    """Charge for service over [service_start, service_end or period end)."""
    if not isinstance(price, FinancialAmount) or price.amount < 0:
        raise FinancialDomainError("Period price must be a nonnegative FinancialAmount")
    end = period.end if service_end is None else service_end
    used, total, fraction = period_fraction(period, service_start, end, precision=precision)
    return ProrationResult(_scaled(price, used, total, precision), used, total, fraction)


def cancellation_credit(price: FinancialAmount, period: FinancialPeriod, *, cancelled_on: date | str,
                        precision: int = 36) -> ProrationResult:
    """Unused-time credit when service ends on ``cancelled_on`` (exclusive end of use)."""
    if not isinstance(price, FinancialAmount) or price.amount < 0:
        raise FinancialDomainError("Period price must be a nonnegative FinancialAmount")
    unused, total, fraction = period_fraction(period, cancelled_on, period.end, precision=precision)
    return ProrationResult(_scaled(price, unused, total, precision), unused, total, fraction)


def plan_change_adjustment(old_price: FinancialAmount, new_price: FinancialAmount, period: FinancialPeriod, *,
                           change_on: date | str, precision: int = 36) -> ProrationResult:
    """Extra charge (positive) or credit (negative) for switching plan mid-period."""
    if not isinstance(old_price, FinancialAmount) or not isinstance(new_price, FinancialAmount):
        raise FinancialValidationError("prices must be FinancialAmount")
    require_same_currency(old_price.currency, new_price.currency, operation="plan change")
    if old_price.amount < 0 or new_price.amount < 0:
        raise FinancialDomainError("Period prices must be nonnegative")
    remaining, total, fraction = period_fraction(period, change_on, period.end, precision=precision)
    return ProrationResult(_scaled(new_price - old_price, remaining, total, precision), remaining, total, fraction)


__all__ = ["ProrationResult", "period_fraction", "partial_period_charge", "cancellation_credit", "plan_change_adjustment"]

if __name__ == "__main__":
    print("\n=== Running Financial Proration ===\n")
    printer.status("TEST", "Financial Proration initialized", "info")
    march = FinancialPeriod(date(2024, 3, 1), date(2024, 4, 1))
    price = FinancialAmount.of("31", "EUR")
    assert partial_period_charge(price, march, service_start="2024-03-11").amount.amount == Decimal("21")
    assert partial_period_charge(price, march, service_start="2024-03-01").amount.amount == Decimal("31")
    credit = cancellation_credit(price, march, cancelled_on="2024-03-16")
    assert credit.used_days == 16 and credit.period_days == 31 and credit.amount.amount == Decimal("16")
    upgrade = plan_change_adjustment(FinancialAmount.of("31", "EUR"), FinancialAmount.of("62", "EUR"), march, change_on="2024-03-16")
    assert upgrade.amount.amount == Decimal("16")
    downgrade = plan_change_adjustment(FinancialAmount.of("62", "EUR"), FinancialAmount.of("31", "EUR"), march, change_on="2024-03-16")
    assert downgrade.amount.amount == Decimal("-16")
    leap = FinancialPeriod(date(2024, 2, 1), date(2024, 3, 1))
    assert partial_period_charge(FinancialAmount.of("29", "EUR"), leap, service_start="2024-02-15").amount.amount == Decimal("15")
    for bad in (lambda: cancellation_credit(price, march, cancelled_on="2024-04-02"),
                lambda: partial_period_charge(price, march, service_start="2024-02-28")):
        try:
            bad()
        except FinancialDomainError:
            pass
        else:
            raise AssertionError("out-of-period proration must be rejected")
    print("\n=== Test ran successfully ===\n")
