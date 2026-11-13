"""SLAI v2.4: economic indicators, market observations, and accounting analysis.

Domain scope: source-bound macroeconomic data, purchasing power, price-only
returns, effective nominal/real rate interpretation, and descriptive financial
statement ratios. This is *not* an IFRS ledger, economic forecasting model,
price provider, or macroeconomic causal-inference engine.

Academic grounding: Fisher (1930), Fama & French (1993); financial-statement
ratio interpretation follows ordinary accounting analysis, not a claim of
IFRS recognition/certification. FinancialData owns point-in-time storage and
FinancialCalculation owns time conventions/numerical precision.
"""
from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, DecimalException, localcontext
from typing import Any

from .data.series import ReturnSeries, observation_as_of, price_returns
from .financial_calculation import FinancialCalculation
from .financial_data import FinancialData
from .financial_types import (
    Currency, FinancialAmount, FinancialPeriod, InterestRate, MarketObservation,
    RateConvention, DayCountConvention,
)
from .utils.config_loader import get_config_section, load_global_config
from .utils.financial_errors import *
from .utils.financial_helpers import *
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Financial Economics")
printer = PrettyPrinter()


@dataclass(frozen=True, slots=True)
class AccountingSnapshot:
    """Explicit statement line items from a named reporting basis.

    Dates and basis are declarations from the caller; no recognition rules are
    inferred. Period covers flow items; ``as_of`` identifies stock balances.
    The application is responsible for validating reported figures and sources.
    """

    as_of: date
    currency: Currency | str
    items: Mapping[str, FinancialAmount]
    period: FinancialPeriod | None = None
    basis: str = "reported"
    source_reference: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", to_date(self.as_of, "as_of"))
        code = normalize_currency_code(self.currency.code if isinstance(self.currency, Currency) else self.currency)
        object.__setattr__(self, "currency", Currency(code))
        if self.period is not None and not isinstance(self.period, FinancialPeriod):
            raise FinancialValidationError("period must be FinancialPeriod or None")
        if self.basis not in ("reported", "adjusted", "projected"):
            raise FinancialValidationError("basis must be reported, adjusted, or projected")
        if not isinstance(self.items, Mapping) or not self.items:
            raise FinancialValidationError("AccountingSnapshot requires nonempty items")
        normalized: dict[str, FinancialAmount] = {}
        for key, value in self.items.items():
            name = normalize_identifier(key, "accounting field", max_length=80)
            if not isinstance(value, FinancialAmount):
                raise FinancialValidationError("Accounting fields must contain FinancialAmount")
            require_same_currency(code, value.currency, operation="accounting snapshot")
            if name in normalized:
                raise FinancialValidationError("Duplicate accounting field")
            normalized[name] = value
        object.__setattr__(self, "items", freeze_mapping(normalized, "items"))
        if self.source_reference is not None:
            object.__setattr__(self, "source_reference", require_text(self.source_reference, "source_reference", max_length=1024))

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class AccountingRatioAnalysis:
    """Ratio outputs may be absent when figures/denominators are unavailable."""

    as_of: date
    currency: str
    ratios: Mapping[str, Decimal]
    unavailable: Mapping[str, str]
    basis: str
    source_reference: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ratios", freeze_mapping(self.ratios, "ratios"))
        object.__setattr__(self, "unavailable", freeze_mapping(self.unavailable, "unavailable"))

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class BalanceReconciliation:
    as_of: date
    assets: FinancialAmount
    liabilities_plus_equity: FinancialAmount
    difference: FinancialAmount
    tolerance: Decimal
    reconciled: bool

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


class FinancialEconomics:
    """Read-only macroeconomic, price-series, and financial-statement analysis.

    ``data`` may be supplied to share one FinancialData store across agents.
    No market data or financial statements are downloaded or synthesized.
    """

    def __init__(self, data: FinancialData | None = None):
        self.config = load_global_config()
        self.financial_economics_config = get_config_section("financial_economics", config=self.config)
        cfg = self.financial_economics_config
        try:
            self.max_series_points = require_positive_int(cfg["max_series_points"], "max_series_points", maximum=1000000)
            self.max_accounting_items = require_positive_int(cfg["max_accounting_items"], "max_accounting_items", maximum=4096)
        except KeyError as exc:
            raise FinancialConfigurationError("Missing financial_economics configuration key", context={"key": str(exc)}) from exc
        if data is not None and not isinstance(data, FinancialData):
            raise FinancialValidationError("data must be FinancialData or None")
        self.data = data if data is not None else FinancialData()
        self.calculation = FinancialCalculation()

    def economic_series(
        self, indicator_id: str, field_name: str, *, from_time: datetime | str,
        until_time: datetime | str, as_of: datetime | str, vintage: str = "latest",
    ) -> tuple[MarketObservation, ...]:
        """Retrieve only observations genuinely available by ``as_of``.

        Macroeconomic series are revised. ``vintage="latest"`` (default) yields
        one value per reference period, the newest revision published by
        ``as_of`` (Croushore & Stark, 2001); ``vintage="all"`` returns every
        revision for audit.
        """
        points = self.data.observation_history(
            indicator_id, field_name, from_time=from_time, until_time=until_time,
            as_of=as_of, vintage=vintage,
        )
        if len(points) > self.max_series_points:
            raise FinancialDomainError("Indicator series exceeds configured limit")
        logger.debug("Financial economic series read | points=%d", len(points))
        return points

    def index_level(
        self, indicator_id: str, field_name: str, *, on: datetime | str,
        as_of: datetime | str, max_age: timedelta,
    ) -> MarketObservation:
        """Index level in force at ``on``: the latest published level, never interpolated.

        The level must have been known at ``as_of`` and be no older than
        ``max_age`` at ``on`` (a monthly index needs a window above one month).
        """
        day = to_utc_datetime(on, "on")
        if not isinstance(max_age, timedelta) or max_age <= timedelta(0):
            raise FinancialValidationError("max_age must be a positive timedelta")
        window = self.economic_series(
            indicator_id, field_name, from_time=day - max_age,
            until_time=day + timedelta(microseconds=1), as_of=as_of)
        return observation_as_of(window, day, max_age=max_age, known_by=to_utc_datetime(as_of, "as_of"))

    def cumulative_inflation(
        self, indicator_id: str, field_name: str, *, start: datetime | str,
        end: datetime | str, as_of: datetime | str, max_age: timedelta,
    ) -> Decimal:
        """Price-index change between two instants using only data known at ``as_of``."""
        first = self.index_level(indicator_id, field_name, on=start, as_of=as_of, max_age=max_age)
        last = self.index_level(indicator_id, field_name, on=end, as_of=as_of, max_age=max_age)
        if first.unit != last.unit:
            raise InvalidMarketDataError("Index levels use different units")
        if first.observed_at >= last.observed_at:
            raise FinancialDomainError("Inflation window needs two distinct published index levels")
        return self.inflation_rate(first.value, last.value)

    def price_return_series(
        self, observations: Sequence[MarketObservation], *, as_of: datetime | str, kind: str = "simple",
    ) -> ReturnSeries:
        """Price-only return series (see data.series for conventions and limits)."""
        return price_returns(observations, as_of=as_of, precision=self.calculation.precision,
                             kind=kind, max_points=self.max_series_points)

    @staticmethod
    def _positive_index(value: Any, name: str) -> Decimal:
        result = to_decimal(value, name)
        if result <= 0:
            raise FinancialDomainError("Price index levels must be strictly positive", context={"field": name})
        return result

    def inflation_rate(self, earlier_index: Any, later_index: Any) -> Decimal:
        """Cumulative price-index change, not an annualized inflation rate."""
        a, b = self._positive_index(earlier_index, "earlier_index"), self._positive_index(later_index, "later_index")
        with localcontext() as ctx:
            ctx.prec = self.calculation.precision
            return b / a - 1

    def annualized_inflation_rate(
        self, earlier_index: Any, later_index: Any, *, start: date | str,
        end: date | str, day_count: DayCountConvention | str = DayCountConvention.ACT_365F,
    ) -> Decimal:
        a, b = self._positive_index(earlier_index, "earlier_index"), self._positive_index(later_index, "later_index")
        years = self.calculation.year_fraction(start, end, day_count)
        if years <= 0:
            raise FinancialDomainError("Annualized inflation requires a positive time interval")
        try:
            with localcontext() as ctx:
                ctx.prec = self.calculation.precision
                return (b / a) ** (Decimal(1) / years) - 1
        except DecimalException as exc:
            raise FinancialPrecisionError("Annualized inflation calculation failed", cause=exc) from exc

    def adjust_purchasing_power(
        self, amount: FinancialAmount, *, from_index: Any, to_index: Any,
    ) -> FinancialAmount:
        """Scale nominal currency units by price-index levels (no FX conversion)."""
        if not isinstance(amount, FinancialAmount):
            raise FinancialValidationError("amount must be FinancialAmount")
        a, b = self._positive_index(from_index, "from_index"), self._positive_index(to_index, "to_index")
        with localcontext() as ctx:
            ctx.prec = self.calculation.precision
            return FinancialAmount(amount.amount * b / a, amount.currency)

    def real_effective_rate(self, nominal: InterestRate, inflation: Any) -> InterestRate:
        """Exact Fisher relation; annual effective rates and same period only.

        ``inflation`` must itself be an annual effective inflation fraction.
        Other rate conventions must first be converted by the calculation owner.
        """
        if not isinstance(nominal, InterestRate):
            raise FinancialValidationError("nominal must be InterestRate")
        if nominal.convention is not RateConvention.EFFECTIVE:
            raise UnsupportedConventionError("Fisher relation requires annual effective nominal rate")
        inf = to_decimal(inflation, "annual_inflation")
        if inf <= -1 or nominal.value <= -1:
            raise FinancialDomainError("Annual effective factors must be positive")
        with localcontext() as ctx:
            ctx.prec = self.calculation.precision
            real = (1 + nominal.value) / (1 + inf) - 1
        return InterestRate(real, RateConvention.EFFECTIVE, day_count=nominal.day_count)

    def price_return(self, earlier: MarketObservation, later: MarketObservation) -> Decimal:
        """Price-only return: excludes dividends, splits, tax and transaction cost.

        Observations must be pre-adjusted to a comparable basis by their provider.
        """
        if not isinstance(earlier, MarketObservation) or not isinstance(later, MarketObservation):
            raise FinancialValidationError("price_return requires two MarketObservation objects")
        if (earlier.instrument_id, earlier.field_name) != (later.instrument_id, later.field_name):
            raise FinancialValidationError("Price observations refer to different series")
        if earlier.unit != later.unit:
            raise FinancialValidationError("Price observations use different units")
        if earlier.currency is None or later.currency is None:
            raise FinancialValidationError("Price returns require currency-qualified observations")
        require_same_currency(earlier.currency, later.currency, operation="price_return")
        if earlier.observed_at >= later.observed_at:
            raise FinancialDomainError("Price return requires strictly increasing observation times")
        if earlier.value <= 0 or later.value < 0:
            raise FinancialDomainError("Price observations must have positive start and nonnegative end")
        with localcontext() as ctx:
            ctx.prec = self.calculation.precision
            return later.value / earlier.value - 1

    def accounting_ratios(self, snapshot: AccountingSnapshot) -> AccountingRatioAnalysis:
        """Produce standard *descriptive* ratios without substituting missing data.

        Net margin = net_income/revenue; current ratio = current_assets/current_liabilities;
        debt/equity = total_liabilities/equity; operating margin = operating_income/revenue;
        cash conversion = operating_cash_flow/net_income.
        """
        if not isinstance(snapshot, AccountingSnapshot):
            raise FinancialValidationError("snapshot must be AccountingSnapshot")
        if len(snapshot.items) > self.max_accounting_items:
            raise FinancialDomainError("Accounting snapshot exceeds configured max_accounting_items")
        specifications = {
            "net_margin": ("net_income", "revenue"),
            "operating_margin": ("operating_income", "revenue"),
            "current_ratio": ("current_assets", "current_liabilities"),
            "debt_to_equity": ("total_liabilities", "total_equity"),
            "operating_cash_conversion": ("operating_cash_flow", "net_income"),
        }
        values: dict[str, Decimal] = {}
        unavailable: dict[str, str] = {}
        with localcontext() as ctx:
            ctx.prec = self.calculation.precision
            for name, (numerator, denominator) in specifications.items():
                missing = [x for x in (numerator, denominator) if x not in snapshot.items]
                if missing:
                    unavailable[name] = "Missing statement line(s): " + ", ".join(missing)
                elif snapshot.items[denominator].amount == 0:
                    unavailable[name] = "Zero denominator"
                else:
                    values[name] = snapshot.items[numerator].amount / snapshot.items[denominator].amount
        logger.debug("Accounting ratios evaluated | available=%d | unavailable=%d", len(values), len(unavailable))
        return AccountingRatioAnalysis(snapshot.as_of, snapshot.currency.code, values, unavailable,
                                       snapshot.basis, snapshot.source_reference)

    def reconcile_balance_sheet(self, snapshot: AccountingSnapshot, *, tolerance: Any = "0") -> BalanceReconciliation:
        if not isinstance(snapshot, AccountingSnapshot):
            raise FinancialValidationError("snapshot must be AccountingSnapshot")
        tol = to_decimal(tolerance, "tolerance")
        if tol < 0:
            raise FinancialDomainError("Reconciliation tolerance must not be negative")
        missing = [name for name in ("total_assets", "total_liabilities", "total_equity") if name not in snapshot.items]
        if missing:
            raise FinancialValidationError("Cannot reconcile missing balance-sheet fields", context={"missing_fields": missing})
        assets, debt, equity = (snapshot.items[name] for name in ("total_assets", "total_liabilities", "total_equity"))
        combined = debt + equity
        difference = assets - combined
        return BalanceReconciliation(snapshot.as_of, assets, combined, difference, tol, abs(difference.amount) <= tol)


__all__ = ["AccountingSnapshot", "AccountingRatioAnalysis", "BalanceReconciliation", "FinancialEconomics"]

if __name__ == "__main__":
    print("\n=== Running Financial Economics ===\n")
    printer.status("TEST", "Financial Economics initialized", "info")
    service = FinancialEconomics()
    assert service.inflation_rate("100", "105") == Decimal("0.05")
    assert service.adjust_purchasing_power(FinancialAmount.of("200", "EUR"), from_index=100, to_index=110).amount == 220
    rate = service.real_effective_rate(InterestRate(Decimal("0.06"), RateConvention.EFFECTIVE), "0.02")
    assert abs(rate.value - Decimal("0.0392156862745098")) < Decimal("1e-12")
    from datetime import timezone
    from .financial_types import FinancialDataSource
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def cpi(value: str, month: int, published_day: int, revision: str | None = None) -> MarketObservation:
        return MarketObservation("CPI-EA", "index", Decimal(value), datetime(2024, month, 1, tzinfo=timezone.utc),
                                 FinancialDataSource("stat-office", datetime(2024, month, published_day, tzinfo=timezone.utc),
                                                     revision=revision), unit="index")

    service.data.ingest_observations([cpi("100", 1, 15), cpi("102", 2, 15), cpi("101.5", 2, 28, "r2"), cpi("103", 3, 15)])
    window = dict(from_time=t0, until_time=datetime(2024, 4, 1, tzinfo=timezone.utc), as_of=datetime(2024, 3, 20, tzinfo=timezone.utc))
    assert [p.value for p in service.economic_series("CPI-EA", "index", **window)] == [Decimal("100"), Decimal("101.5"), Decimal("103")]
    assert len(service.economic_series("CPI-EA", "index", vintage="all", **window)) == 4
    early = dict(window, as_of=datetime(2024, 2, 20, tzinfo=timezone.utc))
    assert [p.value for p in service.economic_series("CPI-EA", "index", **early)] == [Decimal("100"), Decimal("102")]
    assert service.cumulative_inflation("CPI-EA", "index", start=t0, end=datetime(2024, 3, 1, tzinfo=timezone.utc),
                                        as_of=datetime(2024, 3, 20, tzinfo=timezone.utc), max_age=timedelta(days=40)) == Decimal("0.03")
    snap = AccountingSnapshot("2025-12-31", "EUR", {
        "revenue": FinancialAmount.of("100", "EUR"),
        "net_income": FinancialAmount.of("10", "EUR"),
        "total_assets": FinancialAmount.of("200", "EUR"),
        "total_liabilities": FinancialAmount.of("50", "EUR"),
        "total_equity": FinancialAmount.of("150", "EUR"),
    })
    assert service.accounting_ratios(snap).ratios["net_margin"] == Decimal("0.1")
    assert service.reconcile_balance_sheet(snap).reconciled
    print("\n=== Test ran successfully ===\n")
