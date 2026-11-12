"""Recurring-revenue movement metrics and customer lifetime value.

Operates on *contract* monthly recurring revenue (MRR) per customer at two
instants, such as the values produced by ``FinancialSubscription``; it is not
revenue recognition and not cash. All amounts share one currency.

MRR movement over a period, comparing opening and closing snapshots:
  new          customer absent (or zero) at open, positive at close
  expansion    positive at both, higher at close
  contraction  positive at both, lower at close
  churn        positive at open, zero or absent at close
and the identity  closing = opening + new + expansion - contraction - churn
holds exactly (checked on every call). A customer that left and returned
inside the period is classified only by its two snapshots; reactivation needs
event history that two snapshots cannot contain.

Retention measures (definitions as in common SaaS practice):
  GRR  = (opening - contraction - churn) / opening        (<= 1)
  NRR  = (opening + expansion - contraction - churn) / opening
  quick ratio = (new + expansion) / (contraction + churn)

Customer lifetime value with a constant per-period retention rate r and
discount rate d, margin m per active period (Berger & Nasr, 1998, J.
Interactive Marketing 12(1); Gupta & Lehmann, 2003, J. Interactive Marketing
17(1)):  CLV = m * (1+d) / (1+d-r) when the first margin is received now, or
CLV = m * r / (1+d-r) when only future margins count. Retention is assumed
constant and independent of tenure, which real customer bases rarely satisfy;
treat the result as a stylized estimate, not a forecast.
"""
from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Any

from ..financial_types import FinancialAmount
from ..utils.financial_errors import *
from ..utils.financial_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Financial Recurring Metrics")
printer = PrettyPrinter()


@dataclass(frozen=True, slots=True)
class MrrMovement:
    currency: str
    opening: FinancialAmount
    new: FinancialAmount
    expansion: FinancialAmount
    contraction: FinancialAmount
    churn: FinancialAmount
    closing: FinancialAmount
    opening_customers: int
    closing_customers: int
    new_customers: int
    churned_customers: int

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class RetentionMetrics:
    gross_revenue_retention: Decimal | None
    net_revenue_retention: Decimal | None
    logo_churn_rate: Decimal | None
    quick_ratio: Decimal | None
    average_mrr_per_customer: FinancialAmount | None

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


class RecurringMetrics:
    """Deterministic MRR movement, retention and CLV calculations."""

    def __init__(self, *, precision: int):
        self.precision = require_positive_int(precision, "precision", minimum=16, maximum=256)

    @staticmethod
    def _snapshot(values: Mapping[str, FinancialAmount], currency: str, name: str) -> dict[str, Decimal]:
        if not isinstance(values, Mapping):
            raise FinancialValidationError(f"{name} must map customer ids to FinancialAmount")
        if len(values) > 1_000_000:
            raise FinancialValidationError(f"{name} exceeds the supported customer count")
        checked: dict[str, Decimal] = {}
        for key, amount in values.items():
            customer = normalize_identifier(key, "customer_id")
            if not isinstance(amount, FinancialAmount):
                raise FinancialValidationError(f"{name} values must be FinancialAmount")
            require_same_currency(currency, amount.currency, operation="MRR snapshot")
            if amount.amount < 0:
                raise FinancialDomainError("MRR cannot be negative")
            checked[customer] = amount.amount
        return checked

    def movement(self, opening: Mapping[str, FinancialAmount], closing: Mapping[str, FinancialAmount], *,
                 currency: str) -> MrrMovement:
        code = normalize_currency_code(currency)
        before, after = self._snapshot(opening, code, "opening"), self._snapshot(closing, code, "closing")
        zero = Decimal(0)
        new = expansion = contraction = churn = zero
        new_n = churn_n = 0
        for customer in set(before) | set(after):
            o, c = before.get(customer, zero), after.get(customer, zero)
            if o == 0 and c > 0:
                new += c
                new_n += 1
            elif o > 0 and c == 0:
                churn += o
                churn_n += 1
            elif c > o > 0:
                expansion += c - o
            elif 0 < c < o:
                contraction += o - c
        total_open, total_close = sum(before.values(), zero), sum(after.values(), zero)
        if total_open + new + expansion - contraction - churn != total_close:
            raise FinancialIntegrityError("MRR movement does not reconcile to closing MRR")
        amount = lambda v: FinancialAmount(v, code)  # noqa: E731
        return MrrMovement(code, amount(total_open), amount(new), amount(expansion), amount(contraction),
                           amount(churn), amount(total_close), sum(1 for v in before.values() if v > 0),
                           sum(1 for v in after.values() if v > 0), new_n, churn_n)

    def retention(self, movement: MrrMovement) -> RetentionMetrics:
        """Metrics are ``None`` when their denominator is zero (undefined, not zero)."""
        if not isinstance(movement, MrrMovement):
            raise FinancialValidationError("movement must be MrrMovement")
        o = movement.opening.amount
        lost = movement.contraction.amount + movement.churn.amount
        gained = movement.new.amount + movement.expansion.amount
        with localcontext() as ctx:
            ctx.prec = self.precision
            grr = (o - lost) / o if o > 0 else None
            nrr = (o + movement.expansion.amount - lost) / o if o > 0 else None
            logo = Decimal(movement.churned_customers) / Decimal(movement.opening_customers) if movement.opening_customers else None
            quick = gained / lost if lost > 0 else None
            arpa = (FinancialAmount(movement.closing.amount / Decimal(movement.closing_customers), movement.currency)
                    if movement.closing_customers else None)
        return RetentionMetrics(grr, nrr, logo, quick, arpa)

    def customer_lifetime_value(self, margin_per_period: FinancialAmount, *, retention_rate: Any,
                                discount_rate_per_period: Any, include_current_period: bool = False) -> FinancialAmount:
        """Expected discounted margin of one customer (see module docstring)."""
        if not isinstance(margin_per_period, FinancialAmount):
            raise FinancialValidationError("margin_per_period must be FinancialAmount")
        require_bool(include_current_period, "include_current_period")
        r = to_decimal(retention_rate, "retention_rate")
        d = to_decimal(discount_rate_per_period, "discount_rate_per_period")
        if not 0 <= r <= 1:
            raise FinancialDomainError("Retention rate must lie in [0, 1]")
        if d < 0:
            raise FinancialDomainError("Discount rate per period must be nonnegative")
        with localcontext() as ctx:
            ctx.prec = self.precision
            denominator = 1 + d - r
            if denominator <= 0:
                raise FinancialDomainError("CLV diverges unless retention is below 1 plus the discount rate")
            factor = (1 + d) / denominator if include_current_period else r / denominator
            return FinancialAmount(margin_per_period.amount * factor, margin_per_period.currency)


__all__ = ["MrrMovement", "RetentionMetrics", "RecurringMetrics"]

if __name__ == "__main__":
    print("\n=== Running Financial Recurring Metrics ===\n")
    printer.status("TEST", "Financial Recurring Metrics initialized", "info")
    eur = lambda v: FinancialAmount.of(v, "EUR")  # noqa: E731
    metrics = RecurringMetrics(precision=36)
    opening = {"a": eur(100), "b": eur(50), "c": eur(50), "d": eur(0)}
    closing = {"a": eur(120), "b": eur(30), "c": eur(0), "e": eur(40)}
    m = metrics.movement(opening, closing, currency="EUR")
    assert (m.opening.amount, m.new.amount, m.expansion.amount, m.contraction.amount, m.churn.amount, m.closing.amount) == (200, 40, 20, 20, 50, 190)
    assert (m.opening_customers, m.closing_customers, m.new_customers, m.churned_customers) == (3, 3, 1, 1)
    r = metrics.retention(m)
    assert r.gross_revenue_retention == Decimal("0.65") and r.net_revenue_retention == Decimal("0.75")
    assert abs(r.logo_churn_rate - Decimal(1) / 3) < Decimal("1e-25")
    assert abs(r.quick_ratio - Decimal(6) / 7) < Decimal("1e-25")
    empty = metrics.retention(metrics.movement({}, {"x": eur(10)}, currency="EUR"))
    assert empty.gross_revenue_retention is None and empty.quick_ratio is None
    margin = eur(10)
    future = metrics.customer_lifetime_value(margin, retention_rate="0.8", discount_rate_per_period="0.1")
    assert abs(future.amount - Decimal(10) * Decimal("0.8") / Decimal("0.3")) < Decimal("1e-25")
    now_ = metrics.customer_lifetime_value(margin, retention_rate="0.8", discount_rate_per_period="0.1", include_current_period=True)
    assert abs(now_.amount - future.amount - margin.amount) < Decimal("1e-25"), "CLV(now) = CLV(future) + m"
    for bad in (lambda: metrics.customer_lifetime_value(margin, retention_rate="1", discount_rate_per_period="0"),
                lambda: metrics.customer_lifetime_value(margin, retention_rate="1.2", discount_rate_per_period="0.1"),
                lambda: metrics.movement({"a": eur(1)}, {"a": FinancialAmount.of(1, "USD")}, currency="EUR")):
        try:
            bad()
        except FinancialError:
            pass
        else:
            raise AssertionError("invalid recurring-metric input must be rejected")
    print("\n=== Test ran successfully ===\n")
