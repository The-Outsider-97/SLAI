"""Dated cash-flow schedules and aggregation for SLAI Financial.

Cash flow ownership is restricted to schedules, recurrence, balances and
aggregation. Discounting and IRR are delegated to FinancialCalculation.
An outflow has a negative sign; all results are single-currency unless
explicitly grouped by currency by a calling application.
"""
from __future__ import annotations

__version__ = "2.4.0"

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Sequence

from .financial_types import CashFlow, FinancialAmount, InterestRate
from .financial_calculation import FinancialCalculation
from .utils.config_loader import load_global_config, get_config_section
from .utils.financial_errors import *
from .utils.financial_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Financial Cashflow")
printer = PrettyPrinter()


@dataclass(frozen=True, slots=True)
class CashflowSummary:
    start: date
    end: date
    currency: str
    inflows: FinancialAmount
    outflows: FinancialAmount
    net: FinancialAmount
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(),
                "currency": self.currency, "inflows": self.inflows.to_dict(),
                "outflows": self.outflows.to_dict(), "net": self.net.to_dict(), "count": self.count}


class FinancialCashflow:
    """Canonical recurring-date and balance scheduler used by subscriptions."""

    def __init__(self):
        self.config = load_global_config()
        self.financial_cashflow_config = get_config_section("financial_cashflow", config=self.config)
        try:
            self.max_schedule_items = require_positive_int(self.financial_cashflow_config["max_schedule_items"], "max_schedule_items", maximum=1000000)
        except KeyError as exc:
            raise FinancialConfigurationError("Missing financial_cashflow configuration key", context={"key": str(exc)}) from exc
        self.calculation = FinancialCalculation()

    @staticmethod
    def _add_months(anchor: date, months: int, *, preserve_end_of_month: bool) -> date:
        """Advance from the original anchor (not previous occurrence).

        Avoids Jan31 -> Feb28 -> Mar28 drift. Invalid months/overflow fail.
        """
        if months < 0:
            raise FinancialDomainError("Recurrence offset cannot be negative")
        index = (anchor.year - 1) * 12 + (anchor.month - 1) + months
        if index > (9999 * 12 - 1):
            raise FinancialDomainError("Recurring date exceeds supported calendar")
        year, month0 = divmod(index, 12)
        year, month = year + 1, month0 + 1
        last = calendar.monthrange(year, month)[1]
        day = last if preserve_end_of_month else min(anchor.day, last)
        return date(year, month, day)

    @staticmethod
    def _occurrence(anchor: date, interval_unit: str, interval_count: int, occurrence: int, preserve_end_of_month: bool) -> date:
        if interval_unit in ("month", "year"):
            months = interval_count * occurrence * (12 if interval_unit == "year" else 1)
            return FinancialCashflow._add_months(anchor, months, preserve_end_of_month=preserve_end_of_month)
        if interval_unit in ("day", "week"):
            days = interval_count * occurrence * (7 if interval_unit == "week" else 1)
            try:
                return anchor + timedelta(days=days)
            except OverflowError as exc:
                raise FinancialDomainError("Recurring date exceeds supported calendar", cause=exc) from exc
        raise FinancialValidationError("Unsupported recurring interval unit")

    def recurring_schedule(
        self, amount: FinancialAmount, first_date: date | str, *, interval_unit: str,
        interval_count: int = 1, end_exclusive: date | str, start_inclusive: date | str | None = None,
        reference: str | None = None, category: str | None = None,
        preserve_end_of_month: bool | None = None,
    ) -> tuple[CashFlow, ...]:
        """Exact recurring calendar dates, [start_inclusive, end_exclusive).

        Calendar-month/year recurrences retain the initial anchor day. An anchor
        on the final day of a month remains month-end by default. No pro-rating.
        """
        if not isinstance(amount, FinancialAmount):
            raise FinancialValidationError("Recurring amount must be FinancialAmount")
        if not isinstance(interval_unit, str) or interval_unit not in {"day", "week", "month", "year"}:
            raise FinancialValidationError("interval_unit must be day/week/month/year")
        require_positive_int(interval_count, "interval_count", maximum=10000)
        anchor = to_date(first_date, "first_date")
        horizon = to_date(end_exclusive, "end_exclusive")
        floor = anchor if start_inclusive is None else to_date(start_inclusive, "start_inclusive")
        if horizon <= anchor or horizon <= floor:
            return ()
        if preserve_end_of_month is None:
            eom = anchor.day == calendar.monthrange(anchor.year, anchor.month)[1]
        elif type(preserve_end_of_month) is bool:
            eom = preserve_end_of_month
        else:
            raise FinancialValidationError("preserve_end_of_month must be bool or None")
        schedule: list[CashFlow] = []
        for number in range(self.max_schedule_items + 1):
            occurrence = self._occurrence(anchor, interval_unit, interval_count, number, eom)
            if occurrence >= horizon:
                return tuple(schedule)
            if number >= self.max_schedule_items:
                raise FinancialDomainError("Recurring schedule exceeds configured max_schedule_items")
            if occurrence >= floor:
                schedule.append(CashFlow(on_date=occurrence, amount=amount, reference=reference, category=category))
        raise FinancialDomainError("Recurring schedule exceeds configured max_schedule_items")

    @staticmethod
    def _check_flows(flows: Sequence[CashFlow], currency: str | None = None) -> str:
        if not all(isinstance(flow, CashFlow) for flow in flows):
            raise FinancialValidationError("flows must contain CashFlow objects")
        if currency is None:
            if not flows:
                raise FinancialDomainError("Currency is required for empty cash-flow sequence")
            currency_code = str(flows[0].amount.currency)
        else:
            currency_code = str(currency)
        for flow in flows:
            require_same_currency(currency_code, flow.amount.currency, operation="cashflow")
        return currency_code

    def summarize(self, flows: Sequence[CashFlow], *, start: date | str,
                  end_exclusive: date | str, currency: str | None = None) -> CashflowSummary:
        begin, end = to_date(start, "start"), to_date(end_exclusive, "end_exclusive")
        if begin >= end:
            raise FinancialDomainError("Cash-flow summary requires start < end")
        code = self._check_flows(flows, currency)
        received = FinancialAmount.of(0, code)
        paid = FinancialAmount.of(0, code)
        count = 0
        for flow in flows:
            if begin <= flow.on_date < end:
                if flow.amount.amount >= 0:
                    received += flow.amount
                else:
                    paid += flow.amount
                count += 1
        return CashflowSummary(begin, end, code, received, paid, received + paid, count)

    def monthly_totals(self, flows: Sequence[CashFlow], *, currency: str | None = None) -> dict[str, FinancialAmount]:
        code = self._check_flows(flows, currency)
        totals: dict[str, FinancialAmount] = {}
        for flow in sorted(flows, key=lambda f: f.on_date):
            month = flow.on_date.strftime("%Y-%m")
            totals[month] = totals.get(month, FinancialAmount.of(0, code)) + flow.amount
        return totals

    def running_balances(self, flows: Sequence[CashFlow], opening_balance: FinancialAmount, *, as_of: date | str) -> tuple[tuple[date, FinancialAmount], ...]:
        if not isinstance(opening_balance, FinancialAmount):
            raise FinancialValidationError("opening_balance must be FinancialAmount")
        day = to_date(as_of, "as_of")
        self._check_flows(flows, str(opening_balance.currency))
        balance = opening_balance
        history: list[tuple[date, FinancialAmount]] = []
        for flow in sorted(flows, key=lambda x: x.on_date):
            if flow.on_date < day:
                raise FinancialDomainError("Balance timeline includes flows before opening date")
            balance += flow.amount
            history.append((flow.on_date, balance))
        return tuple(history)

    def discounted_value(self, flows: Sequence[CashFlow], rate: InterestRate, *, valuation_date: date | str) -> FinancialAmount:
        """Delegate discounting to the single financial formula owner."""
        return self.calculation.net_present_value(flows, rate, valuation_date)


__all__ = ["FinancialCashflow", "CashflowSummary"]

if __name__ == "__main__":
    print("\n=== Running Financial Cashflow ===\n")
    printer.status("TEST", "Financial Cashflow initialized", "info")
    engine = FinancialCashflow()
    amount = FinancialAmount.of("-19.99", "EUR")
    dates = engine.recurring_schedule(amount, date(2024, 1, 31), interval_unit="month",
        end_exclusive=date(2024, 5, 1), reference="subscription")
    assert [x.on_date.isoformat() for x in dates] == ["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"]
    summary = engine.summarize(dates, start=date(2024, 1, 1), end_exclusive=date(2024, 5, 1))
    assert summary.count == 4 and summary.net.amount == Decimal("-79.96")
    assert len(engine.monthly_totals(dates)) == 4
    assert engine.running_balances(dates, FinancialAmount.of("100", "EUR"), as_of="2024-01-01")[-1][1].amount == Decimal("20.04")
    print("\n=== Test ran successfully ===\n")
