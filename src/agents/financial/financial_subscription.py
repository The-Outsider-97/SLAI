"""Contractual subscription scheduling and recurring-value analytics for SLAI.

Subscription analysis is read-only. Plans and charges are *projections*, not
provider invoices, payment authorizations, accounting revenue recognition, or
proof of entitlement. Real payment actions remain in src.functions.payment.
All calendar scheduling is delegated to FinancialCashflow to avoid duplicate
month-end/recurrence logic. Itemized pricing, proration, rounding and recurring
revenue metrics come from the shared ``business`` modules, and currency
conversion from ``data.fx``. No hidden FX conversion or assumed rounding.
"""
from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, localcontext
from typing import Any, Sequence

from .business.charges import Charge, ChargeBreakdown, apply_charges
from .business.proration import ProrationResult, cancellation_credit, partial_period_charge
from .business.recurring_metrics import MrrMovement, RecurringMetrics, RetentionMetrics
from .business.rounding import RoundingPolicy
from .data.fx import FxConverter, FxPolicy
from .financial_types import CashFlow, FinancialAmount, FinancialPeriod
from .financial_cashflow import FinancialCashflow, CashflowSummary
from .utils.config_loader import load_global_config, get_config_section
from .utils.financial_errors import *
from .utils.financial_helpers import *
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Financial Subscription")
printer = PrettyPrinter()


@dataclass(frozen=True, slots=True)
class BillingInterval:
    unit: str
    count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.unit, str) or self.unit not in {"day", "week", "month", "year"}:
            raise UnsupportedConventionError("Billing interval must use day/week/month/year")
        require_positive_int(self.count, "billing_interval.count", maximum=10000)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BillingInterval":
        if not isinstance(data, Mapping) or "unit" not in data:
            raise FinancialValidationError("BillingInterval requires unit")
        return cls(data["unit"], data.get("count", 1))

    def to_dict(self) -> dict[str, Any]:
        return {"unit": self.unit, "count": self.count}


@dataclass(frozen=True, slots=True)
class SubscriptionContract:
    """Immutable service billing contract, excluding provider credentials.

    ``first_charge_on`` is first *scheduled* payment, not an inference from
    a free-trial duration. Cancellation/end dates are exclusive. A discount
    fraction applies to the contractual price until ``discount_until`` (exclusive).
    """
    subscription_id: str
    service_name: str
    price: FinancialAmount
    interval: BillingInterval
    starts_on: date
    first_charge_on: date
    ends_on: date | None = None
    category: str | None = None
    discount_fraction: Decimal = Decimal("0")
    discount_until: date | None = None
    direction: str = "expense"
    charges: tuple[Charge, ...] = ()
    rounding: RoundingPolicy | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "subscription_id", normalize_identifier(self.subscription_id, "subscription_id"))
        object.__setattr__(self, "service_name", require_text(self.service_name, "service_name", max_length=256))
        if not isinstance(self.price, FinancialAmount) or self.price.amount < 0:
            raise FinancialDomainError("Subscription price must be nonnegative FinancialAmount")
        if not isinstance(self.interval, BillingInterval):
            raise FinancialValidationError("Subscription interval must be BillingInterval")
        start = to_date(self.starts_on, "starts_on")
        first = to_date(self.first_charge_on, "first_charge_on")
        if first < start:
            raise FinancialDomainError("First charge cannot predate subscription start")
        object.__setattr__(self, "starts_on", start)
        object.__setattr__(self, "first_charge_on", first)
        if self.ends_on is not None:
            ending = to_date(self.ends_on, "ends_on")
            if ending <= start:
                raise FinancialDomainError("Subscription end must follow start date")
            object.__setattr__(self, "ends_on", ending)
        if self.discount_until is not None:
            object.__setattr__(self, "discount_until", to_date(self.discount_until, "discount_until"))
        fraction = to_decimal(self.discount_fraction, "discount_fraction")
        if not Decimal(0) <= fraction <= Decimal(1):
            raise FinancialDomainError("Subscription discount_fraction must be between 0 and 1")
        object.__setattr__(self, "discount_fraction", fraction)
        if self.category is not None:
            object.__setattr__(self, "category", require_text(self.category, "category", max_length=128))
        if self.direction not in {"expense", "revenue"}:
            raise FinancialValidationError("Subscription direction must be expense/revenue")
        if not isinstance(self.charges, (list, tuple)) or any(not isinstance(c, Charge) for c in self.charges):
            raise FinancialValidationError("Subscription charges must be a list/tuple of Charge values")
        object.__setattr__(self, "charges", tuple(self.charges))
        if self.rounding is not None and not isinstance(self.rounding, RoundingPolicy):
            raise FinancialValidationError("Subscription rounding must be RoundingPolicy or None")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubscriptionContract":
        """Build from a JSON-style mapping (money as {amount, currency}, dates ISO)."""
        if not isinstance(data, Mapping):
            raise FinancialValidationError("SubscriptionContract must be a mapping")
        missing = [k for k in ("subscription_id", "service_name", "price", "interval", "starts_on", "first_charge_on")
                   if k not in data]
        if missing:
            raise FinancialValidationError("SubscriptionContract is missing required fields", context={"missing_fields": missing})
        interval = data["interval"]
        rounding = data.get("rounding")
        return cls(
            subscription_id=data["subscription_id"], service_name=data["service_name"],
            price=FinancialAmount.from_dict(data["price"]),
            interval=interval if isinstance(interval, BillingInterval) else BillingInterval.from_dict(interval),
            starts_on=data["starts_on"], first_charge_on=data["first_charge_on"], ends_on=data.get("ends_on"),
            category=data.get("category"), discount_fraction=data.get("discount_fraction", Decimal("0")),
            discount_until=data.get("discount_until"), direction=data.get("direction", "expense"),
            charges=tuple(c if isinstance(c, Charge) else Charge.from_dict(c) for c in data.get("charges", ())),
            rounding=None if rounding is None else (rounding if isinstance(rounding, RoundingPolicy)
                                                    else RoundingPolicy(rounding["mode"] if isinstance(rounding, Mapping) else rounding)),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)

    def is_billable_on(self, day: date | str) -> bool:
        when = to_date(day, "day")
        return self.first_charge_on <= when and (self.ends_on is None or when < self.ends_on)


class FinancialSubscription:
    """Read-only renewal, cash-flow and monthly recurring-value analysis."""

    def __init__(self):
        self.config = load_global_config()
        self.financial_subscription_config = get_config_section("financial_subscription", config=self.config)
        try:
            self.max_forecast_days = require_positive_int(
                self.financial_subscription_config["max_forecast_days"], "max_forecast_days", maximum=365250)
            self.max_subscriptions = require_positive_int(
                self.financial_subscription_config["max_subscriptions"], "max_subscriptions", maximum=100000)
        except KeyError as exc:
            raise FinancialConfigurationError("Missing financial_subscription configuration key", context={"key": str(exc)}) from exc
        self.cashflow = FinancialCashflow()
        # Finance numerical precision has a single owner: FinancialCalculation.
        self.precision = self.cashflow.calculation.precision
        self.metrics = RecurringMetrics(precision=self.precision)

    def _validate_horizon(self, start: date | str, end_exclusive: date | str) -> tuple[date, date]:
        begin, end = to_date(start, "start"), to_date(end_exclusive, "end_exclusive")
        if begin >= end or (end - begin).days > self.max_forecast_days:
            raise FinancialDomainError("Forecast horizon is empty or exceeds configured limit")
        return begin, end

    def invoice_at(self, contract: SubscriptionContract, on: date | str) -> ChargeBreakdown:
        """Itemized *unsigned* price due for one billing occurrence on ``on``.

        The contractual discount fraction (until ``discount_until``) is applied
        as a percentage discount together with the contract's declared
        ``charges``; see ``business.charges`` for ordering and rounding rules.
        """
        if not isinstance(contract, SubscriptionContract):
            raise FinancialValidationError("contract must be SubscriptionContract")
        when = to_date(on, "on")
        declared = list(contract.charges)
        if contract.discount_fraction > 0:
            declared.append(Charge("contract discount", "discount_percent", contract.discount_fraction,
                                   active_until=contract.discount_until))
        return apply_charges(contract.price, declared, on=when, precision=self.precision, policy=contract.rounding)

    def _price_at(self, contract: SubscriptionContract, when: date) -> FinancialAmount:
        """Signed cash amount of one occurrence (expenses negative), tax included."""
        due = self.invoice_at(contract, when).total.amount
        return FinancialAmount(due if contract.direction == "revenue" else -due, contract.price.currency)

    def _recurring_price_at(self, contract: SubscriptionContract, when: date) -> FinancialAmount:
        """Signed recurring-value amount: discounts and fees included, tax excluded."""
        due = self.invoice_at(contract, when).subtotal.amount
        return FinancialAmount(due if contract.direction == "revenue" else -due, contract.price.currency)

    def projected_charges(self, contract: SubscriptionContract, *, start: date | str,
                          end_exclusive: date | str) -> tuple[CashFlow, ...]:
        if not isinstance(contract, SubscriptionContract):
            raise FinancialValidationError("contract must be SubscriptionContract")
        begin, end = self._validate_horizon(start, end_exclusive)
        if contract.ends_on is not None:
            end = min(end, contract.ends_on)
        if end <= contract.first_charge_on or end <= begin:
            return ()
        base = self.cashflow.recurring_schedule(
            contract.price, contract.first_charge_on, interval_unit=contract.interval.unit,
            interval_count=contract.interval.count, start_inclusive=begin,
            end_exclusive=end, reference=contract.subscription_id, category=contract.category,
        )
        return tuple(CashFlow(on_date=f.on_date, amount=self._price_at(contract, f.on_date),
                              reference=contract.subscription_id, category=contract.category)
                     for f in base)

    def forecast(self, contracts: Sequence[SubscriptionContract], *, start: date | str,
                 end_exclusive: date | str) -> tuple[CashFlow, ...]:
        self._validate_horizon(start, end_exclusive)
        if not isinstance(contracts, (list, tuple)) or len(contracts) > self.max_subscriptions:
            raise FinancialValidationError("contracts must be a bounded list/tuple")
        if not all(isinstance(c, SubscriptionContract) for c in contracts):
            raise FinancialValidationError("contracts must contain SubscriptionContract values")
        identifiers = [c.subscription_id for c in contracts]
        if len(set(identifiers)) != len(identifiers):
            raise FinancialIntegrityError("Duplicate subscription IDs in forecast")
        # Mixed currencies are preserved in the flow list, not implicitly summed.
        flows = [flow for contract in contracts for flow in self.projected_charges(contract, start=start, end_exclusive=end_exclusive)]
        return tuple(sorted(flows, key=lambda x: (x.on_date, x.reference or "")))

    def forecast_by_currency(self, contracts: Sequence[SubscriptionContract], *,
                             start: date | str, end_exclusive: date | str) -> dict[str, CashflowSummary]:
        begin, end = self._validate_horizon(start, end_exclusive)
        if not isinstance(contracts, (list, tuple)):
            raise FinancialValidationError("contracts must be list/tuple")
        flows = self.forecast(contracts, start=begin, end_exclusive=end)
        currencies = sorted({str(c.price.currency) for c in contracts})
        return {code: self.cashflow.summarize(
            [flow for flow in flows if flow.amount.currency.code == code], # type: ignore
            start=begin, end_exclusive=end, currency=code) for code in currencies}

    def monthly_recurring_value(self, contract: SubscriptionContract, *, as_of: date | str) -> FinancialAmount:
        """Contract MRR normalization, not recognized revenue or realized cash.

        Only month/year cadence has an exact calendar-month normalization.
        Weekly/daily subscriptions require an explicit analytical annualization
        convention from the caller and are not guessed here.
        """
        if not isinstance(contract, SubscriptionContract):
            raise FinancialValidationError("contract must be SubscriptionContract")
        day = to_date(as_of, "as_of")
        zero = FinancialAmount.of(0, contract.price.currency)
        if not contract.is_billable_on(day):
            return zero
        months = (contract.interval.count if contract.interval.unit == "month"
                  else contract.interval.count * 12 if contract.interval.unit == "year" else None)
        if months is None:
            raise UnsupportedConventionError("MRR requires month/year billing cadence")
        with localcontext() as ctx:
            ctx.prec = self.precision
            monthly = self._recurring_price_at(contract, day).amount / Decimal(months)
        return FinancialAmount(monthly, contract.price.currency)

    def annual_recurring_value(self, contract: SubscriptionContract, *, as_of: date | str) -> FinancialAmount:
        monthly = self.monthly_recurring_value(contract, as_of=as_of)
        with localcontext() as ctx:
            ctx.prec = self.precision
            return FinancialAmount(monthly.amount * 12, monthly.currency)


    # -------------------------------------------------- billing periods & proration
    def billing_period(self, contract: SubscriptionContract, day: date | str) -> FinancialPeriod:
        """Billing period [charge date, next charge date) that contains ``day``.

        Dates come from FinancialCashflow, the single owner of recurrence and
        month-end logic. ``day`` must not precede the first charge date.
        """
        if not isinstance(contract, SubscriptionContract):
            raise FinancialValidationError("contract must be SubscriptionContract")
        when = to_date(day, "day")
        if when < contract.first_charge_on:
            raise FinancialDomainError("Day precedes the first charge date of the contract")
        unit_days = {"day": 1, "week": 7, "month": 31, "year": 366}[contract.interval.unit]
        horizon = when + timedelta(days=unit_days * contract.interval.count + 1)
        dates = [f.on_date for f in self.cashflow.recurring_schedule(
            contract.price, contract.first_charge_on, interval_unit=contract.interval.unit,
            interval_count=contract.interval.count, end_exclusive=horizon)]
        for begin, end in zip(dates, dates[1:]):
            if begin <= when < end:
                return FinancialPeriod(begin, end)
        raise FinancialDomainError("Billing period could not be determined for the requested day")

    def prorated_charge(self, contract: SubscriptionContract, *, service_start: date | str,
                        service_end: date | str | None = None) -> ProrationResult:
        """Pre-tax charge for part of the billing period that contains ``service_start``."""
        period = self.billing_period(contract, service_start)
        due = self._recurring_price_at(contract, period.start)
        result = partial_period_charge(
            FinancialAmount(abs(due.amount), due.currency), period, service_start=service_start,
            service_end=service_end, precision=self.precision)
        sign = 1 if contract.direction == "revenue" else -1
        return ProrationResult(FinancialAmount(sign * result.amount.amount, result.amount.currency),
                               result.used_days, result.period_days, result.fraction)

    def cancellation_credit(self, contract: SubscriptionContract, *, cancelled_on: date | str) -> ProrationResult:
        """Credit for unused time after ``cancelled_on`` (cash back to the payer).

        Signed from the contract holder's perspective: a refund owed by a
        revenue contract is negative, a refund received on an expense is positive.
        """
        period = self.billing_period(contract, cancelled_on)
        due = self._recurring_price_at(contract, period.start)
        result = cancellation_credit(FinancialAmount(abs(due.amount), due.currency), period,
                                     cancelled_on=cancelled_on, precision=self.precision)
        sign = -1 if contract.direction == "revenue" else 1
        return ProrationResult(FinancialAmount(sign * result.amount.amount, result.amount.currency),
                               result.used_days, result.period_days, result.fraction)

    # ------------------------------------------------------------ currency & metrics
    def forecast_in_currency(self, contracts: Sequence[SubscriptionContract], *, start: date | str,
                             end_exclusive: date | str, to_currency: str, as_of: Any,
                             converter: FxConverter, policy: FxPolicy | None = None) -> tuple[CashFlow, ...]:
        """Forecast re-expressed in one currency at a single as-of spot rate per currency.

        This is a translation at today's rate, NOT a forecast of future exchange
        rates; the rate legs are available via ``converter.rate`` for audit.
        """
        if not isinstance(converter, FxConverter):
            raise FinancialValidationError("converter must be FxConverter")
        flows = self.forecast(contracts, start=start, end_exclusive=end_exclusive)
        rates: dict[str, Decimal] = {}
        converted: list[CashFlow] = []
        with localcontext() as ctx:
            ctx.prec = self.precision
            for flow in flows:
                code = flow.amount.currency.code  # type: ignore[union-attr]
                if code not in rates:
                    rates[code], _ = converter.rate(code, to_currency, as_of=as_of, policy=policy)
                converted.append(CashFlow(flow.on_date, FinancialAmount(flow.amount.amount * rates[code], to_currency),
                                          reference=flow.reference, category=flow.category))
        return tuple(converted)

    def recurring_value_movement(self, contracts: Sequence[SubscriptionContract], *, opening: date | str,
                                 closing: date | str, currency: str) -> tuple[MrrMovement, RetentionMetrics]:
        """MRR movement and retention between two dates for same-direction contracts.

        Each contract is one account (``subscription_id``). Only month/year
        cadences have an exact MRR; others raise. Values are magnitudes.
        """
        begin, end = to_date(opening, "opening"), to_date(closing, "closing")
        if begin >= end:
            raise FinancialDomainError("Movement window requires opening < closing")
        if not isinstance(contracts, (list, tuple)) or len(contracts) > self.max_subscriptions:
            raise FinancialValidationError("contracts must be a bounded list/tuple")
        if not all(isinstance(c, SubscriptionContract) for c in contracts):
            raise FinancialValidationError("contracts must contain SubscriptionContract values")
        code = normalize_currency_code(currency)
        selected = [c for c in contracts if c.price.currency.code == code]  # type: ignore[union-attr]
        if len({c.direction for c in selected}) > 1:
            raise FinancialDomainError("Do not mix revenue and expense contracts in one MRR movement")
        if len({c.subscription_id for c in selected}) != len(selected):
            raise FinancialIntegrityError("Duplicate subscription IDs in MRR movement")
        snapshot = lambda day: {c.subscription_id: abs(self.monthly_recurring_value(c, as_of=day))  # noqa: E731
                                for c in selected}
        movement = self.metrics.movement(snapshot(begin), snapshot(end), currency=code)
        return movement, self.metrics.retention(movement)


    def customer_lifetime_value(self, margin_per_period: FinancialAmount, *, retention_rate: Any,
                                discount_rate_per_period: Any, include_current_period: bool = False) -> FinancialAmount:
        """Expected discounted margin of one customer (Berger & Nasr 1998; Gupta & Lehmann 2003)."""
        return self.metrics.customer_lifetime_value(
            margin_per_period, retention_rate=retention_rate, discount_rate_per_period=discount_rate_per_period,
            include_current_period=include_current_period)

__all__ = ["BillingInterval", "SubscriptionContract", "FinancialSubscription"]

if __name__ == "__main__":
    print("\n=== Running Financial Subscription ===\n")
    printer.status("TEST", "Financial Subscription initialized", "info")
    service = FinancialSubscription()
    contract = SubscriptionContract("demo-plan", "Example", FinancialAmount.of("30", "EUR"),
        BillingInterval("month"), date(2024, 1, 31), date(2024, 2, 29),
        discount_fraction=Decimal("0.2"), discount_until=date(2024, 4, 1))
    charges = service.projected_charges(contract, start="2024-01-01", end_exclusive="2024-06-01")
    assert [c.on_date.isoformat() for c in charges] == ["2024-02-29", "2024-03-31", "2024-04-30", "2024-05-31"]
    assert [c.amount.amount for c in charges] == [-24, -24, -30, -30]
    assert service.monthly_recurring_value(contract, as_of="2024-03-01").amount == -24
    assert service.forecast_by_currency([contract], start="2024-01-01", end_exclusive="2024-06-01")["EUR"].count == 4
    # itemized pricing, proration and metrics
    taxed = SubscriptionContract("taxed", "Taxed", FinancialAmount.of("30", "EUR"), BillingInterval("month"),
        date(2024, 1, 1), date(2024, 1, 1), charges=(Charge("VAT", "tax_percent", "0.21"),), direction="revenue")
    assert service.invoice_at(taxed, "2024-02-01").total.amount == Decimal("36.30")
    assert service.monthly_recurring_value(taxed, as_of="2024-02-01").amount == 30, "MRR excludes tax"
    assert service.billing_period(taxed, "2024-02-15") == FinancialPeriod(date(2024, 2, 1), date(2024, 3, 1))
    assert abs(service.prorated_charge(taxed, service_start="2024-02-15").amount.amount - Decimal(30) * 15 / 29) < Decimal("1e-25")
    credit = service.cancellation_credit(taxed, cancelled_on="2024-02-16")
    assert credit.used_days == 14 and credit.period_days == 29 and credit.amount.amount < 0
    other = SubscriptionContract("other", "Other", FinancialAmount.of("10", "EUR"), BillingInterval("month"),
        date(2024, 1, 1), date(2024, 1, 1), direction="revenue")
    movement, retention = service.recurring_value_movement([taxed, other], opening="2024-02-01", closing="2024-03-01", currency="EUR")
    assert movement.opening.amount == 40 and movement.closing.amount == 40 and retention.gross_revenue_retention == 1
    round_to_cent = SubscriptionContract("r", "R", FinancialAmount.of("9.99", "EUR"), BillingInterval("month"),
        date(2024, 1, 1), date(2024, 1, 1), charges=(Charge("VAT", "tax_percent", "0.21"),), rounding=RoundingPolicy("half_up"))
    assert [c.amount.amount for c in service.projected_charges(round_to_cent, start="2024-01-01", end_exclusive="2024-02-01")] == [Decimal("-12.09")]
    assert SubscriptionContract.from_dict(taxed.to_dict()) == taxed and SubscriptionContract.from_dict(round_to_cent.to_dict()) == round_to_cent
    print("\n=== Test ran successfully ===\n")
