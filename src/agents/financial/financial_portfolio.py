"""SLAI v2.4: point-in-time portfolio valuation and allocation research.

This module is read-only. Prices come only from FinancialData, which enforces
observed_at/retrieved_at limits. A quote is not a total-return series; dividends,
splits, execution costs, taxes, short margin and external cash flows are not
silently modeled. Different currency sleeves are NEVER summed without a sourced
explicit FX conversion. Rebalance results are amount differences, not orders.

Currency conversion (data.fx), rounding and cost estimation (business.*), and
performance measurement and return moments (business.performance) are shared
modules, not re-implemented here.

Academic basis: Markowitz (1952), Sharpe (1964), Fama & French (1993); returns
and moments: Dietz (1966), Bacon (2008), Michaud (1989), DeMiguel et al. (2009).
"""
from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from decimal import Decimal, localcontext
from types import MappingProxyType
from typing import Any

from .business.charges import Charge, apply_charges
from .business.performance import (ExternalFlow, PerformanceMeasurement, RelativeStatistics, ReturnMoments,
                                   ReturnStatistics, TimeWeightedResult, ValuationPoint)
from .business.rounding import MoneyRounder, RoundingPolicy
from .data.fx import FxConversion, FxPolicy
from .data.series import ReturnSeries, align_returns, price_returns
from .financial_data import FinancialData
from .financial_types import (FinancialAmount, FinancialInstrument, InterestRate, MarketObservation,
                              InstrumentType)
from .financial_valuation import FinancialValuation, HistoricalRisk
from .utils.config_loader import get_config_section, load_global_config
from .utils.financial_errors import *
from .utils.financial_helpers import *
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Financial Portfolio")
printer = PrettyPrinter()


@dataclass(frozen=True, slots=True)
class PortfolioHolding:
    """Long-only position with cost basis per unit (not a broker account record).

    Cost basis excludes fees and corporate actions unless the caller supplies an
    adjusted authoritative cost; quantity can be zero but not negative.
    For bonds, quote_multiplier must explicitly normalize provider price units
    into currency units per held bond (e.g. 100 quoted per par-1000 instrument
    requires a factor of 10). Accrued interest is not inferred.
    """

    instrument: FinancialInstrument
    quantity: Decimal
    cost_per_unit: FinancialAmount
    quote_field: str = "close"
    quote_source_id: str | None = None
    quote_multiplier: Decimal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.instrument, FinancialInstrument):
            raise FinancialValidationError("instrument must be FinancialInstrument")
        qty = to_decimal(self.quantity, "quantity")
        if qty < 0:
            raise FinancialDomainError("Long-only PortfolioHolding quantity must be nonnegative")
        object.__setattr__(self, "quantity", qty)
        allowed = {InstrumentType.EQUITY, InstrumentType.ETF, InstrumentType.FUND, InstrumentType.BOND}
        if self.instrument.instrument_type not in allowed:
            raise UnsupportedConventionError("Portfolio unit-price valuation does not support this instrument type")
        if self.instrument.instrument_type is InstrumentType.BOND and self.quote_multiplier is None:
            raise UnsupportedConventionError("Bonds require explicit quote_multiplier for the source's price convention")
        multiplier = Decimal(1) if self.quote_multiplier is None else to_decimal(self.quote_multiplier, "quote_multiplier")
        if multiplier <= 0:
            raise FinancialDomainError("quote_multiplier must be positive")
        object.__setattr__(self, "quote_multiplier", multiplier)
        if not isinstance(self.cost_per_unit, FinancialAmount) or self.cost_per_unit.amount < 0:
            raise FinancialDomainError("cost_per_unit must be nonnegative FinancialAmount")
        require_same_currency(self.instrument.currency, self.cost_per_unit.currency, operation="holding cost basis")
        object.__setattr__(self, "quote_field", normalize_identifier(self.quote_field, "quote_field", max_length=64))
        if self.quote_source_id is not None:
            object.__setattr__(self, "quote_source_id", normalize_identifier(self.quote_source_id, "quote_source_id"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PortfolioHolding":
        if not isinstance(data, Mapping):
            raise FinancialValidationError("PortfolioHolding must be a mapping")
        missing = [k for k in ("instrument", "quantity", "cost_per_unit") if k not in data]
        if missing:
            raise FinancialValidationError("PortfolioHolding is missing required fields", context={"missing_fields": missing})
        instrument = data["instrument"]
        cost = data["cost_per_unit"]
        multiplier = data.get("quote_multiplier")
        return cls(
            instrument=instrument if isinstance(instrument, FinancialInstrument) else FinancialInstrument.from_dict(instrument),
            quantity=data["quantity"],
            cost_per_unit=cost if isinstance(cost, FinancialAmount) else FinancialAmount.from_dict(cost),
            quote_field=data.get("quote_field", "close"), quote_source_id=data.get("quote_source_id"),
            quote_multiplier=None if multiplier is None else to_decimal(multiplier, "quote_multiplier"),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class PositionValuation:
    holding: PortfolioHolding
    quote: MarketObservation
    market_value: FinancialAmount
    cost_basis: FinancialAmount
    unrealized_pnl: FinancialAmount
    price_return_on_cost: Decimal | None

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    as_of: datetime
    positions: tuple[PositionValuation, ...]
    market_value_by_currency: Mapping[str, FinancialAmount]
    cost_basis_by_currency: Mapping[str, FinancialAmount]
    unrealized_pnl_by_currency: Mapping[str, FinancialAmount]

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", to_utc_datetime(self.as_of, "as_of"))
        if not isinstance(self.positions, (tuple, list)) or any(not isinstance(p, PositionValuation) for p in self.positions):
            raise FinancialValidationError("positions must contain PositionValuation values")
        object.__setattr__(self, "positions", tuple(self.positions))
        for key in ("market_value_by_currency", "cost_basis_by_currency", "unrealized_pnl_by_currency"):
            object.__setattr__(self, key, freeze_mapping(getattr(self, key), key))

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class PortfolioAllocation:
    """Hypothetical value-delta plan. Not an executable order instruction."""

    as_of: datetime
    currency: str
    current_weights: Mapping[str, Decimal]
    target_weights: Mapping[str, Decimal]
    target_values: Mapping[str, FinancialAmount]
    value_deltas: Mapping[str, FinancialAmount]

    def __post_init__(self) -> None:
        for key in ("current_weights", "target_weights", "target_values", "value_deltas"):
            object.__setattr__(self, key, freeze_mapping(getattr(self, key), key))

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


class FinancialPortfolio:
    """Long-only, source-based holdings and allocation analytics.

    A shared FinancialData store can be injected for economic/portfolio agent
    access to the same observations. The service does not authorize trades.
    """

    def __init__(self, data: FinancialData | None = None, valuation: FinancialValuation | None = None):
        self.config = load_global_config()
        self.financial_portfolio_config = get_config_section("financial_portfolio", config=self.config)
        try:
            self.max_positions = require_positive_int(self.financial_portfolio_config["max_positions"], "max_positions", maximum=4096)
        except KeyError as exc:
            raise FinancialConfigurationError("Missing financial_portfolio configuration key", context={"key": str(exc)}) from exc
        if data is not None and not isinstance(data, FinancialData):
            raise FinancialValidationError("data must be FinancialData or None")
        if valuation is not None and not isinstance(valuation, FinancialValuation):
            raise FinancialValidationError("valuation must be FinancialValuation or None")
        self.data = data if data is not None else FinancialData()
        self.valuation = valuation if valuation is not None else FinancialValuation()
        # One numerical precision for the subsystem, owned by FinancialCalculation.
        self.performance = PerformanceMeasurement(precision=self.valuation.calculation.precision)
        self.rounder = MoneyRounder()

    def value_positions(
        self, holdings: Sequence[PortfolioHolding], *, as_of: datetime | str,
        max_age: timedelta | None = None,
    ) -> PortfolioSnapshot:
        """Require a real sourced price for every position, including zero quantity.

        A *price-only* change is reported as unrealized P&L on adjusted cost
        basis; not as portfolio total return. Caller must account for cash flows.
        """
        if not isinstance(holdings, (tuple, list)) or len(holdings) > self.max_positions:
            raise FinancialValidationError("holdings must be a bounded list/tuple")
        if any(not isinstance(h, PortfolioHolding) for h in holdings):
            raise FinancialValidationError("holdings must contain PortfolioHolding values")
        identities = [h.instrument.instrument_id for h in holdings]
        if len(set(identities)) != len(identities):
            raise FinancialIntegrityError("Duplicate instrument identities; aggregate holdings upstream")
        moment = to_utc_datetime(as_of, "as_of")
        results: list[PositionValuation] = []
        totals: dict[str, FinancialAmount] = {}
        costs: dict[str, FinancialAmount] = {}
        pnls: dict[str, FinancialAmount] = {}
        with localcontext() as ctx:
            ctx.prec = self.valuation.calculation.precision
            for holding in holdings:
                code = holding.instrument.currency.code
                quote = self.data.latest_observation(
                    holding.instrument.instrument_id, holding.quote_field, as_of=moment,
                    max_age=max_age, source_id=holding.quote_source_id,
                )
                if quote.currency is None:
                    raise FinancialValidationError("Portfolio price observation must specify currency")
                require_same_currency(holding.instrument.currency, quote.currency, operation="position quote")
                if quote.value < 0:
                    raise FinancialDomainError("Unlevered long-only market quote must be nonnegative")
                value = FinancialAmount(holding.quantity * quote.value * holding.quote_multiplier, holding.instrument.currency)
                cost = FinancialAmount(holding.quantity * holding.cost_per_unit.amount, holding.instrument.currency)
                pnl = value - cost
                roi = pnl.amount / cost.amount if cost.amount else None
                results.append(PositionValuation(holding, quote, value, cost, pnl, roi))
                zero = FinancialAmount.of(0, code)
                totals[code] = totals.get(code, zero) + value
                costs[code] = costs.get(code, zero) + cost
                pnls[code] = pnls.get(code, zero) + pnl
        logger.debug("Financial portfolio valued | positions=%d | currency_sleeves=%d", len(results), len(totals))
        return PortfolioSnapshot(moment, tuple(results), totals, costs, pnls)

    def allocation_weights(self, snapshot: PortfolioSnapshot) -> Mapping[str, Decimal]:
        """Within-currency weights only; reject mixed-currency aggregation."""
        if not isinstance(snapshot, PortfolioSnapshot):
            raise FinancialValidationError("snapshot must be PortfolioSnapshot")
        if not snapshot.positions:
            raise FinancialDomainError("Cannot weight an empty portfolio")
        if len(snapshot.market_value_by_currency) != 1:
            raise FinancialDomainError("Portfolio weighting requires one currency or explicit FX conversion")
        total = next(iter(snapshot.market_value_by_currency.values())).amount
        if total <= 0:
            raise FinancialDomainError("Portfolio weights require positive aggregate market value")
        with localcontext() as ctx:
            ctx.prec = self.valuation.calculation.precision
            return freeze_mapping({p.holding.instrument.instrument_id: p.market_value.amount / total
                                   for p in snapshot.positions}, "weights")

    def rebalance_amounts(self, snapshot: PortfolioSnapshot, target_weights: Mapping[str, Any], *,
                          rounding: RoundingPolicy | None = None) -> PortfolioAllocation:
        """Compute value differences only; no trade sizing or execution.

        Does not model lot sizes, costs, taxes, settlement constraints or shorting.
        With ``rounding`` the total is quantized to the currency's minor unit,
        target values are allocated by largest remainder so they sum to that
        total exactly, and current values are quantized the same way.
        """
        weights = self.allocation_weights(snapshot)
        if not isinstance(target_weights, Mapping) or set(target_weights) != set(weights):
            raise FinancialValidationError("Target allocation must cover every holding exactly once")
        normalized = {key: to_decimal(value, "target_weight") for key, value in target_weights.items()}
        with localcontext() as ctx:
            # The check must be exact: summing at the ambient (28-digit) precision would round away the residual
            # that equal_weight_targets deliberately places on the last weight.
            ctx.prec = self.valuation.calculation.precision + 16
            weight_sum = sum(normalized.values(), Decimal(0))
        if any(w < 0 or w > 1 for w in normalized.values()) or weight_sum != 1:
            raise FinancialDomainError("Target weights must be nonnegative and sum to exactly one")
        code, total = next(iter(snapshot.market_value_by_currency.items()))
        current_values = {p.holding.instrument.instrument_id: p.market_value for p in snapshot.positions}
        targets: dict[str, FinancialAmount] = {}
        deltas: dict[str, FinancialAmount] = {}
        with localcontext() as ctx:
            ctx.prec = self.valuation.calculation.precision
            if rounding is None:
                for key, target in normalized.items():
                    targets[key] = FinancialAmount(total.amount * target, total.currency)
                    deltas[key] = targets[key] - current_values[key]
            else:
                keys = sorted(normalized)
                parts = self.rounder.allocate(self.rounder.quantize(total, rounding), [normalized[k] for k in keys])
                for key, part in zip(keys, parts):
                    targets[key] = part
                    deltas[key] = part - self.rounder.quantize(current_values[key], rounding)
        return PortfolioAllocation(snapshot.as_of, code, weights, normalized, targets, deltas)

    def equal_weight_targets(self, snapshot: PortfolioSnapshot) -> Mapping[str, Decimal]:
        """1/N weights: the naive benchmark an optimized allocation must beat out of sample
        (DeMiguel, Garlappi & Uppal, 2009). The last weight absorbs the rounding residual."""
        ids = sorted(p.holding.instrument.instrument_id for p in snapshot.positions)
        if not ids:
            raise FinancialDomainError("Cannot weight an empty portfolio")
        with localcontext() as ctx:
            ctx.prec = self.valuation.calculation.precision
            share = Decimal(1) / Decimal(len(ids))
            weights = {key: share for key in ids}
            weights[ids[-1]] = Decimal(1) - share * (len(ids) - 1)
        return freeze_mapping(weights, "weights")

    def rebalance_cost(self, plan: PortfolioAllocation, fees: Sequence[Charge], *,
                       rounding: RoundingPolicy | None = None) -> FinancialAmount:
        """Estimated trading cost of a plan: ``fees`` applied to each nonzero |delta|.

        Only FEE_* charges are meaningful here (a percentage is a proportional
        cost such as spread or commission, an amount is a fixed ticket fee).
        The result is an estimate, not a broker quote; taxes and market impact
        are not modeled.
        """
        if not isinstance(plan, PortfolioAllocation):
            raise FinancialValidationError("plan must be PortfolioAllocation")
        if any(not isinstance(c, Charge) or not str(c.kind.value).startswith("fee_") for c in fees):  # type: ignore[union-attr]
            raise FinancialValidationError("rebalance_cost accepts only fee charges")
        total = FinancialAmount.of(0, plan.currency)
        for delta in plan.value_deltas.values():
            if delta.amount == 0:
                continue
            traded = abs(delta)
            breakdown = apply_charges(traded, fees, precision=self.valuation.calculation.precision,
                                      policy=rounding, rounder=self.rounder)
            total += breakdown.total - traded
        return total

    def _fx_policy(self, policy: FxPolicy | None, max_age: timedelta | None) -> FxPolicy | None:
        if max_age is None:
            return policy
        if policy is None:
            return FxPolicy(max_age=max_age)
        return policy if policy.max_age is not None else replace(policy, max_age=max_age)

    def convert_snapshot(
        self, snapshot: PortfolioSnapshot, *, reporting_currency: str,
        max_age: timedelta | None = None, policy: FxPolicy | None = None,
    ) -> tuple[FinancialAmount, tuple[FxConversion, ...]]:
        """Total in ``reporting_currency`` and the conversion record of every sleeve.

        By default only *direct* sourced pairs are used. Reciprocals or one
        pivot currency must be requested through ``policy`` and are recorded.
        """
        if not isinstance(snapshot, PortfolioSnapshot):
            raise FinancialValidationError("snapshot must be PortfolioSnapshot")
        reporting = normalize_currency_code(reporting_currency)
        converter = self.data.fx_converter(precision=self.valuation.calculation.precision)
        sleeves = [value for _, value in sorted(snapshot.market_value_by_currency.items())]
        return converter.convert_total(sleeves, reporting, as_of=snapshot.as_of,
                                       policy=self._fx_policy(policy, max_age))

    def value_in_reporting_currency(
        self, snapshot: PortfolioSnapshot, *, reporting_currency: str,
        max_age: timedelta | None = None, policy: FxPolicy | None = None,
    ) -> FinancialAmount:
        """Convert sleeves using only *direct*, sourced FX pairs as-of snapshot.

        No reciprocal, cross-rate, price interpolation, or implied FX unless the
        caller opts in through ``policy``. Historical quote retrieval is
        constrained by source retrieval timestamp.
        """
        return self.convert_snapshot(snapshot, reporting_currency=reporting_currency,
                                     max_age=max_age, policy=policy)[0]

    # ---------------------------------------------------------- returns and moments
    def return_series(self, instrument_ids: Sequence[str], *, from_time: datetime | str, until_time: datetime | str,
                      as_of: datetime | str, field_name: str = "close", kind: str = "simple",
                      source_id: str | None = None) -> tuple[ReturnSeries, ...]:
        """Price-only return series from stored observations known at ``as_of``.

        One value per valid time (latest vintage). Use a total-return field for
        total return; dividends and splits are never added here.
        """
        if not isinstance(instrument_ids, (list, tuple)) or not 1 <= len(instrument_ids) <= self.max_positions:
            raise FinancialValidationError("instrument_ids must be a bounded nonempty list/tuple")
        moment = to_utc_datetime(as_of, "as_of")
        result: list[ReturnSeries] = []
        for instrument_id in instrument_ids:
            history = self.data.observation_history(
                instrument_id, field_name, from_time=from_time, until_time=until_time,
                as_of=moment, source_id=source_id, vintage="latest")
            result.append(price_returns(history, as_of=moment, precision=self.valuation.calculation.precision, kind=kind))
        return tuple(result)

    def return_moments(self, instrument_ids: Sequence[str], *, from_time: datetime | str, until_time: datetime | str,
                       as_of: datetime | str, periods_per_year: int, field_name: str = "close",
                       kind: str = "simple", source_id: str | None = None) -> ReturnMoments:
        """Annualized sample mean and covariance over the periods all instruments share.

        Estimates are noisy: with few observations relative to assets the
        covariance matrix is singular and optimized weights are unreliable
        (Michaud, 1989). ``covariance_full_rank_possible`` flags that case.
        """
        series = self.return_series(instrument_ids, from_time=from_time, until_time=until_time, as_of=as_of,
                                    field_name=field_name, kind=kind, source_id=source_id)
        _, columns = align_returns(series)
        return self.performance.estimate_moments(columns, periods_per_year=periods_per_year)

    # --------------------------------------------------------------- performance
    def time_weighted_return(self, valuations: Sequence[ValuationPoint], flows: Sequence[ExternalFlow] = ()) -> TimeWeightedResult:
        return self.performance.time_weighted_return(valuations, flows)

    def modified_dietz_return(self, begin: ValuationPoint, end: ValuationPoint, flows: Sequence[ExternalFlow] = ()) -> Decimal:
        return self.performance.modified_dietz(begin, end, flows)

    def money_weighted_return(self, begin: ValuationPoint, end: ValuationPoint, flows: Sequence[ExternalFlow] = ()) -> InterestRate:
        """Annualized IRR of the investor's cash flows (root finding owned by FinancialCalculation)."""
        investor_flows = self.performance.investor_cash_flows(begin, end, flows)
        return self.valuation.calculation.xirr(investor_flows)

    def annualize_return(self, total_return: Any, years: Any, *, allow_short_period: bool = False) -> Decimal:
        """Geometric annualization; periods under one year are refused unless explicitly allowed (GIPS)."""
        return self.performance.annualize_return(total_return, years, allow_short_period=allow_short_period)

    def return_statistics(self, returns: Sequence[Any], *, periods_per_year: int,
                          risk_free_per_period: Any = "0") -> ReturnStatistics:
        """Sample mean/volatility and Sharpe ratio of one period-return series (Sharpe 1966, 1994)."""
        return self.performance.statistics(returns, periods_per_year=periods_per_year,
                                           risk_free_per_period=risk_free_per_period)

    def relative_statistics(self, returns: Sequence[Any], benchmark: Sequence[Any], *, periods_per_year: int,
                            risk_free_per_period: Any = "0") -> RelativeStatistics:
        """Beta, Jensen's alpha, tracking error and information ratio against an aligned benchmark."""
        return self.performance.relative_statistics(returns, benchmark, periods_per_year=periods_per_year,
                                                    risk_free_per_period=risk_free_per_period)

    def historical_risk(self, period_returns: Sequence[Any], *, exposure: FinancialAmount,
                        confidence: Any | None = None) -> HistoricalRisk:
        """Delegate historical-risk calculations to valuation (single owner)."""
        return self.valuation.historical_var_es(period_returns, exposure=exposure, confidence=confidence)


__all__ = ["PortfolioHolding", "PositionValuation", "PortfolioSnapshot", "PortfolioAllocation", "FinancialPortfolio"]

if __name__ == "__main__":
    print("\n=== Running Financial Portfolio ===\n")
    printer.status("TEST", "Financial Portfolio initialized", "info")
    from datetime import timezone
    from .financial_types import FinancialDataSource, InstrumentType
    now = datetime(2024, 4, 2, tzinfo=timezone.utc)
    service = FinancialPortfolio()
    source = FinancialDataSource("demo-provider", now)
    instrument = FinancialInstrument("ABC.X", InstrumentType.EQUITY, "EUR")
    service.data.ingest_observations((MarketObservation("ABC.X", "close", Decimal("120"), now, source, "EUR"),))
    holding = PortfolioHolding(instrument, Decimal("2"), FinancialAmount.of(100, "EUR"))
    snap = service.value_positions((holding,), as_of=now)
    assert snap.market_value_by_currency["EUR"].amount == 240
    assert snap.unrealized_pnl_by_currency["EUR"].amount == 40
    assert service.allocation_weights(snap)["ABC.X"] == 1
    plan = service.rebalance_amounts(snap, {"ABC.X": "1"})
    assert plan.value_deltas["ABC.X"].amount == 0
    # rounding, costs, FX, returns and performance
    from .financial_types import ExchangeRate
    two = FinancialPortfolio()
    second = FinancialInstrument("DEF.X", InstrumentType.EQUITY, "EUR")
    two.data.ingest_observations((MarketObservation("ABC.X", "close", Decimal("120"), now, source, "EUR"),
                                  MarketObservation("DEF.X", "close", Decimal("33.33"), now, source, "EUR")))
    snap2 = two.value_positions((PortfolioHolding(instrument, Decimal("2"), FinancialAmount.of(100, "EUR")),
                                 PortfolioHolding(second, Decimal("3"), FinancialAmount.of(30, "EUR"))), as_of=now)
    plan2 = two.rebalance_amounts(snap2, two.equal_weight_targets(snap2), rounding=RoundingPolicy("half_even"))
    assert sum((v.amount for v in plan2.target_values.values()), Decimal(0)) == Decimal("339.99")
    assert sum((v.amount for v in plan2.value_deltas.values()), Decimal(0)) == 0
    cost = two.rebalance_cost(plan2, [Charge("commission", "fee_percent", "0.001"), Charge("ticket", "fee_amount", "1")],
                              rounding=RoundingPolicy("half_up"))
    assert cost.amount > 2 and cost.currency.code == "EUR"
    two.data.ingest_exchange_rates((ExchangeRate("EUR", "USD", Decimal("1.1"), now, source),))
    assert two.value_in_reporting_currency(snap2, reporting_currency="USD").amount == Decimal("373.989")
    try:
        two.value_in_reporting_currency(snap2.__class__(snap2.as_of, snap2.positions, {"USD": FinancialAmount.of(1, "USD")},
                                                        snap2.cost_basis_by_currency, snap2.unrealized_pnl_by_currency), reporting_currency="EUR")
    except MissingMarketDataError:
        pass
    else:
        raise AssertionError("an inverse quote must not be inferred by default")
    days = [now - timedelta(days=9 - k) for k in range(10)]
    prices = {"ABC.X": ["100", "101", "99", "102", "104", "103", "105", "104", "107", "108"],
              "DEF.X": ["50", "50.5", "50", "51", "52", "51.5", "52", "52.5", "53", "54"]}
    two.data.ingest_observations(tuple(MarketObservation(name, "px", Decimal(v), t, source, "EUR")
                                       for name, values in prices.items() for t, v in zip(days, values)))
    moments = two.return_moments(("ABC.X", "DEF.X"), from_time=days[0], until_time=now + timedelta(seconds=1), as_of=now,
                                 periods_per_year=252, field_name="px")
    assert moments.asset_ids == ("ABC.X", "DEF.X") and moments.observations == 9 and moments.covariance_full_rank_possible
    assert moments.covariance[0][1] == moments.covariance[1][0] and moments.covariance[0][0] > 0
    points = [ValuationPoint(now, FinancialAmount.of(0, "EUR")), ValuationPoint(now + timedelta(days=365), FinancialAmount.of(110, "EUR"))]
    flow = [ExternalFlow(now, FinancialAmount.of(100, "EUR"))]
    assert abs(two.time_weighted_return(points, flow).total_return - Decimal("0.1")) < Decimal("1e-25")
    assert abs(two.money_weighted_return(ValuationPoint(now, FinancialAmount.of(100, "EUR")),
                                         ValuationPoint(now + timedelta(days=365), FinancialAmount.of(110, "EUR"))).value - Decimal("0.1")) < Decimal("1e-9")
    assert PortfolioHolding.from_dict(holding.to_dict()) == holding
    print("\n=== Test ran successfully ===\n")
