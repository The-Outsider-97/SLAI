"""Deterministic financial mathematics for SLAI v2.4.

Financial conventions are explicit.  Uses Decimal for amounts and SLAI's FinancialAmount/CashFlow/InterestRate types.
No provider access, training, agent delegation, currency conversion or automatic monetary rounding occurs.
Numerical outputs are subject to the configured Decimal precision.
"""
from __future__ import annotations

__version__ = "2.4.0"

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, DecimalException, localcontext
from typing import Any, Sequence
import calendar

from .financial_types import CashFlow, DayCountConvention, FinancialAmount, InterestRate, RateConvention
from .utils.config_loader import get_config_section, load_global_config
from .utils.financial_errors import *
from .utils.financial_helpers import *
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Financial Calculation")
printer = PrettyPrinter()


@dataclass(frozen=True, slots=True)
class LoanInstallment:
    period: int
    payment: FinancialAmount
    interest: FinancialAmount
    principal: FinancialAmount
    balance: FinancialAmount

    def to_dict(self) -> dict[str, Any]:
        return {"period": self.period, "payment": self.payment.to_dict(),
                "interest": self.interest.to_dict(), "principal": self.principal.to_dict(),
                "balance": self.balance.to_dict()}


class FinancialCalculation:
    """Finance-specific arithmetic, discounted cash flow and debt schedules."""

    def __init__(self):
        self.config = load_global_config()
        self.financial_calculation_config = get_config_section("financial_calculation", config=self.config)
        cfg = self.financial_calculation_config
        try:
            self.precision = require_positive_int(cfg["precision"], "precision", minimum=16, maximum=256)
            self.max_iterations = require_positive_int(cfg["max_iterations"], "max_iterations", minimum=16, maximum=4096)
            self.irr_tolerance = to_decimal(cfg["irr_tolerance"], "irr_tolerance")
            self.irr_scan_points = require_positive_int(cfg["irr_scan_points"], "irr_scan_points", minimum=32, maximum=4096)
            self.max_amortization_periods = require_positive_int(
                cfg["max_amortization_periods"], "max_amortization_periods", maximum=100000)
        except KeyError as exc:
            raise FinancialConfigurationError("Missing financial_calculation configuration key", context={"key": str(exc)}) from exc
        if not Decimal("1e-30") <= self.irr_tolerance <= Decimal("0.0001"):
            raise FinancialConfigurationError("irr_tolerance must be between 1e-30 and 0.0001")

    def year_fraction(self, start: date | str, end: date | str, convention: DayCountConvention | str) -> Decimal:
        """Day-count fractions: ACT/365F, ACT/360, ACT/ACT ISDA, 30/360 US and 30E/360.

        All intervals are [start, end).  US 30/360 uses the NASD end-of-February
        handling for both endpoints and 31st-day adjustment.
        """
        a, b = to_date(start, "start"), to_date(end, "end")
        if b < a:
            raise FinancialDomainError("Year fraction requires end >= start")
        try:
            dc = convention if isinstance(convention, DayCountConvention) else DayCountConvention(convention)
        except ValueError as exc:
            raise UnsupportedConventionError("Unsupported day-count convention") from exc
        days = (b - a).days
        # Decimal division must follow the same precision policy as other financial arithmetic.
        if dc is DayCountConvention.ACT_365F:
            with localcontext() as ctx:
                ctx.prec = self.precision
                return Decimal(days) / Decimal(365)
        if dc is DayCountConvention.ACT_360:
            with localcontext() as ctx:
                ctx.prec = self.precision
                return Decimal(days) / Decimal(360)
        if dc is DayCountConvention.ACT_ACT_ISDA:
            current, total = a, Decimal(0)
            with localcontext() as ctx:
                ctx.prec = self.precision
                while current < b:
                    following = min(b, date(current.year + 1, 1, 1)) if current.year < 9999 else b
                    year_days = 366 if calendar.isleap(current.year) else 365
                    total += Decimal((following - current).days) / Decimal(year_days)
                    current = following
            return total
        y1, m1, d1 = a.year, a.month, a.day
        y2, m2, d2 = b.year, b.month, b.day
        if dc is DayCountConvention.THIRTYE_360:
            d1, d2 = min(d1, 30), min(d2, 30)
        else:
            last_feb_a = m1 == 2 and d1 == calendar.monthrange(y1, 2)[1]
            last_feb_b = m2 == 2 and d2 == calendar.monthrange(y2, 2)[1]
            if last_feb_a:
                d1 = 30
            if last_feb_b and last_feb_a:
                d2 = 30
            if d1 == 31:
                d1 = 30
            if d2 == 31 and d1 >= 30:
                d2 = 30
        with localcontext() as ctx:
            ctx.prec = self.precision
            return Decimal(360 * (y2 - y1) + 30 * (m2 - m1) + d2 - d1) / Decimal(360)

    def _discount_factor(self, rate: InterestRate, years: Decimal) -> Decimal:
        if not isinstance(rate, InterestRate):
            raise FinancialValidationError("rate must be InterestRate")
        try:
            with localcontext() as ctx:
                ctx.prec = self.precision
                if rate.convention is RateConvention.CONTINUOUS:
                    return (-rate.value * years).exp()
                if rate.convention is RateConvention.NOMINAL:
                    if rate.periods_per_year is None:
                        raise FinancialValidationError("Nominal rate requires periods_per_year")
                    n = Decimal(rate.periods_per_year)
                    base = Decimal(1) + rate.value / n
                    exponent = -(n * years)
                elif rate.convention is RateConvention.EFFECTIVE:
                    base, exponent = Decimal(1) + rate.value, -years
                else:
                    raise UnsupportedConventionError("Unsupported rate convention")
                if base <= 0:
                    raise FinancialDomainError("Interest rate produces non-positive compounding base")
                return base.__pow__(exponent)
        except DecimalException as exc:
            raise FinancialPrecisionError("Decimal discount-factor evaluation failed", cause=exc) from exc

    def discount_factor(self, rate: InterestRate, start: date | str, end: date | str) -> Decimal:
        if not isinstance(rate, InterestRate):
            raise FinancialValidationError("rate must be InterestRate")
        years = self.year_fraction(start, end, rate.day_count)
        return self._discount_factor(rate, years)

    def present_value(self, future: FinancialAmount, rate: InterestRate, start: date | str, end: date | str) -> FinancialAmount:
        if not isinstance(future, FinancialAmount):
            raise FinancialValidationError("future must be FinancialAmount")
        try:
            with localcontext() as ctx:
                ctx.prec = self.precision
                return FinancialAmount(future.amount * self.discount_factor(rate, start, end), future.currency)
        except DecimalException as exc:
            raise FinancialPrecisionError("Present-value evaluation failed", cause=exc) from exc

    def future_value(self, present: FinancialAmount, rate: InterestRate, start: date | str, end: date | str) -> FinancialAmount:
        if not isinstance(present, FinancialAmount):
            raise FinancialValidationError("present must be FinancialAmount")
        try:
            with localcontext() as ctx:
                ctx.prec = self.precision
                df = self.discount_factor(rate, start, end)
                if not df:
                    raise FinancialDomainError("Discount factor cannot be zero")
                return FinancialAmount(present.amount / df, present.currency)
        except DecimalException as exc:
            raise FinancialPrecisionError("Future-value evaluation failed", cause=exc) from exc

    def net_present_value(self, flows: Sequence[CashFlow], rate: InterestRate, valuation_date: date | str) -> FinancialAmount:
        if not flows:
            raise FinancialDomainError("NPV requires at least one dated cash flow")
        valuation = to_date(valuation_date, "valuation_date")
        if not all(isinstance(flow, CashFlow) for flow in flows):
            raise FinancialValidationError("flows must contain CashFlow values")
        currency = flows[0].amount.currency
        with localcontext() as ctx:
            ctx.prec = self.precision
            total = Decimal(0)
            for flow in flows:
                require_same_currency(currency, flow.amount.currency, operation="net_present_value")
                if flow.on_date < valuation:
                    raise FinancialDomainError("NPV cash flow precedes valuation date")
                total += flow.amount.amount * self.discount_factor(rate, valuation, flow.on_date)
        return FinancialAmount(total, currency)

    def periodic_rate(self, rate: InterestRate, periods_per_year: int) -> Decimal:
        """Effective rate for equal intervals; not an actual/actual calendar accrual."""
        require_positive_int(periods_per_year, "periods_per_year", maximum=366)
        if not isinstance(rate, InterestRate):
            raise FinancialValidationError("rate must be InterestRate")
        with localcontext() as ctx:
            ctx.prec = self.precision
            p = Decimal(periods_per_year)
            if rate.convention is RateConvention.NOMINAL:
                if rate.periods_per_year is None:
                    raise FinancialValidationError("Nominal rate requires periods_per_year")
                periods = Decimal(rate.periods_per_year)
                base = Decimal(1) + rate.value / periods
                power = periods / p
                if base <= 0:
                    raise FinancialDomainError("Nominal rate yields non-positive compounding base")
                return base.__pow__(power) - 1
            if rate.convention is RateConvention.EFFECTIVE:
                if rate.value <= -1:
                    raise FinancialDomainError("Effective annual rate must exceed -100%")
                return (1 + rate.value).__pow__(Decimal(1) / p) - 1
            if rate.convention is RateConvention.CONTINUOUS:
                return (rate.value / p).exp() - 1
            raise UnsupportedConventionError("Unsupported rate convention")

    def loan_payment(self, principal: FinancialAmount, rate: InterestRate, periods: int, periods_per_year: int) -> FinancialAmount:
        if not isinstance(principal, FinancialAmount) or principal.amount <= 0:
            raise FinancialDomainError("Loan principal must be positive FinancialAmount")
        require_positive_int(periods, "periods", maximum=100000)
        r = self.periodic_rate(rate, periods_per_year)
        with localcontext() as ctx:
            ctx.prec = self.precision
            if r == 0:
                payment = principal.amount / Decimal(periods)
            else:
                base = 1 + r
                payment = principal.amount * r / (1 - base.__pow__(-periods))
            if not payment.is_finite() or payment <= 0:
                raise FinancialDomainError("Loan payment is outside supported domain")
        return FinancialAmount(payment, principal.currency)

    def amortization_schedule(self, principal: FinancialAmount, rate: InterestRate,
                              periods: int, periods_per_year: int) -> tuple[LoanInstallment, ...]:
        """Unrounded analytical amortization; caller must apply settlement rounding."""
        require_positive_int(periods, "periods", maximum=self.max_amortization_periods)
        payment = self.loan_payment(principal, rate, periods, periods_per_year)
        r = self.periodic_rate(rate, periods_per_year)
        balance = principal.amount
        schedule: list[LoanInstallment] = []
        with localcontext() as ctx:
            ctx.prec = self.precision
            for index in range(1, periods + 1):
                interest = balance * r
                paid = payment.amount if index != periods else balance + interest
                principal_paid = paid - interest
                balance = balance - principal_paid
                if index == periods:
                    balance = Decimal(0)
                money = lambda amount: FinancialAmount(amount, principal.currency)
                schedule.append(LoanInstallment(index, money(paid), money(interest), money(principal_paid), money(balance)))
        return tuple(schedule)

    def xirr(self, flows: Sequence[CashFlow], *, valuation_date: date | str | None = None) -> InterestRate:
        """Find one annual effective date-aware IRR, refusing ambiguous roots.

        Bounded grid scanning detects sign-changing roots, then bisects. Even-
        multiplicity roots may not trigger sign changes; such cases fail closed.
        """
        if len(flows) < 2 or not all(isinstance(x, CashFlow) for x in flows):
            raise FinancialDomainError("XIRR requires at least two dated CashFlow values")
        require_same_currency(flows[0].amount.currency, flows[-1].amount.currency, operation="xirr")
        for flow in flows:
            require_same_currency(flows[0].amount.currency, flow.amount.currency, operation="xirr")
        if not any(f.amount.amount > 0 for f in flows) or not any(f.amount.amount < 0 for f in flows):
            raise FinancialDomainError("XIRR requires both positive and negative cash flows")
        base_date = min(f.on_date for f in flows) if valuation_date is None else to_date(valuation_date, "valuation_date")
        if any(f.on_date < base_date for f in flows):
            raise FinancialDomainError("XIRR cash flow precedes valuation date")
        if len({f.on_date for f in flows}) < 2:
            raise FinancialDomainError("XIRR requires distinct cash-flow dates")
        with localcontext() as ctx:
            ctx.prec = self.precision
            dated = tuple((f.amount.amount, self.year_fraction(base_date, f.on_date, DayCountConvention.ACT_365F)) for f in flows)
            def f(rate: Decimal) -> Decimal:
                base = 1 + rate
                if base <= 0:
                    raise FinancialDomainError("XIRR discount base must remain positive")
                try:
                    return sum((amount / base.__pow__(years) for amount, years in dated), Decimal(0))
                except DecimalException as exc:
                    raise FinancialPrecisionError("XIRR numerical evaluation failed", cause=exc) from exc
            # Grid uniform in ln(1+r) covers both negative rates and high returns.
            low_log, high_log = Decimal("-16"), Decimal("10")
            roots: list[Decimal] = []
            brackets: list[tuple[Decimal, Decimal]] = []
            previous_rate = low_log.exp() - 1
            previous_value = f(previous_rate)
            for i in range(1, self.irr_scan_points + 1):
                x = low_log + (high_log - low_log) * Decimal(i) / self.irr_scan_points
                rate = x.exp() - 1
                value = f(rate)
                if value == 0:
                    roots.append(rate)
                if (previous_value < 0 < value) or (previous_value > 0 > value):
                    brackets.append((previous_rate, rate))
                previous_rate, previous_value = rate, value
            if len(roots) + len(brackets) != 1:
                if not roots and not brackets:
                    raise FinancialConvergenceError("XIRR root not bracketed in configured search domain")
                raise FinancialDomainError("XIRR is ambiguous: multiple candidate roots")
            if roots:
                return InterestRate(roots[0], RateConvention.EFFECTIVE)
            lo, hi = brackets[0]
            f_lo = f(lo)
            for _ in range(self.max_iterations):
                mid = (lo + hi) / 2
                value = f(mid)
                if value == 0 or hi - lo <= self.irr_tolerance * max(Decimal(1), abs(mid)):
                    return InterestRate(mid, RateConvention.EFFECTIVE)
                if (f_lo > 0 and value > 0) or (f_lo < 0 and value < 0):
                    lo, f_lo = mid, value
                else:
                    hi = mid
        raise FinancialConvergenceError("XIRR bisection exceeded maximum iterations")


__all__ = ["FinancialCalculation", "LoanInstallment"]

if __name__ == "__main__":
    print("\n=== Running Financial Calculation ===\n")
    printer.status("TEST", "Financial Calculation initialized", "info")
    engine = FinancialCalculation()
    from decimal import Decimal as D
    p = FinancialAmount.of("1000", "EUR")
    r = InterestRate(D("0.05"), RateConvention.EFFECTIVE)
    start, end = date(2024, 1, 1), date(2025, 1, 1)
    assert abs(engine.year_fraction(start, end, DayCountConvention.ACT_365F) - D(366) / D(365)) < D("1e-25")
    assert abs(engine.future_value(p, r, start, end).amount - D("1050")) < D("0.30")
    assert abs(engine.present_value(engine.future_value(p, r, start, end), r, start, end).amount - p.amount) < D("1e-20")
    assert engine.loan_payment(p, r, 12, 12).amount > 0
    rows = engine.amortization_schedule(p, r, 12, 12)
    assert len(rows) == 12 and rows[-1].balance.amount == 0
    flows = (CashFlow(start, -p), CashFlow(end, FinancialAmount.of("1100", "EUR")))
    irr = engine.xirr(flows)
    assert abs(irr.value - D("0.1")) < D("0.001")
    assert engine.net_present_value((CashFlow(end, p),), r, start).amount > 0
    print("\n=== Test ran successfully ===\n")
