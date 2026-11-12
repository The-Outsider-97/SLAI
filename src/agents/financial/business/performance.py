"""Investment performance measurement and return-moment estimation.

Return measures (single currency; ``FinancialAmount`` in, ``Decimal`` out):

* Time-weighted return (TWR) - chain-links sub-period returns between
  valuations, removing the effect of external flows; the GIPS-preferred measure
  of manager skill (GIPS Standards, CFA Institute). A valuation immediately
  BEFORE each external flow is required; flows without one are rejected rather
  than approximated.
* Modified Dietz return (Dietz, 1966; Bacon, 2008, "Practical Portfolio
  Performance Measurement and Attribution", ch. 2) - a day-weighted
  approximation of the money-weighted return that needs no intermediate
  valuations. ``investor_cash_flows`` prepares the exact money-weighted (IRR)
  problem for FinancialCalculation.xirr, which owns root finding.
* Annualization follows the GIPS rule: returns of periods shorter than one
  year are not annualized unless the caller explicitly allows it.

Risk statistics use sample (n-1) estimators in Decimal. Annualizing mean and
variance by the number of periods per year assumes i.i.d. returns; Lo (2002),
"The statistics of Sharpe ratios", Financial Analysts Journal 58(4), shows the
square-root-of-time rule can badly misstate the Sharpe ratio under serial
correlation, so treat annualized ratios as descriptive. Sharpe (1966, 1994),
Jensen (1968) define the ratio and alpha used here. ``estimate_moments`` returns
sample moments for mean-variance analysis; sample means are noisy enough that
optimizers amplify estimation error (Michaud, 1989), see DeMiguel, Garlappi &
Uppal (2009) for why 1/N is the benchmark any optimized weights must beat.
Generic statistics stay in STEM; this module needs Decimal exactness and ties
its estimators to financial conventions, which the float-based library does not
offer.
"""
from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, DecimalException, localcontext
from typing import Any

from ..financial_types import CashFlow, FinancialAmount
from ..utils.financial_errors import *
from ..utils.financial_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Financial Performance")
printer = PrettyPrinter()


@dataclass(frozen=True, slots=True)
class ValuationPoint:
    """Portfolio market value at an instant, measured BEFORE any flow at that instant."""

    at: datetime
    value: FinancialAmount

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", to_utc_datetime(self.at, "at"))
        if not isinstance(self.value, FinancialAmount) or self.value.amount < 0:
            raise FinancialDomainError("Valuation must be a nonnegative FinancialAmount")

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class ExternalFlow:
    """Contribution (positive) or withdrawal (negative) crossing the portfolio boundary."""

    at: datetime
    amount: FinancialAmount

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", to_utc_datetime(self.at, "at"))
        if not isinstance(self.amount, FinancialAmount):
            raise FinancialValidationError("Flow amount must be FinancialAmount")

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class TimeWeightedResult:
    total_return: Decimal
    period_returns: tuple[Decimal, ...]
    start: datetime
    end: datetime

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class ReturnStatistics:
    observations: int
    mean: Decimal
    volatility: Decimal
    annualized_mean: Decimal
    annualized_volatility: Decimal
    sharpe_ratio: Decimal | None

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class RelativeStatistics:
    observations: int
    beta: Decimal | None
    alpha: Decimal | None            # Jensen's alpha per period
    tracking_error: Decimal          # annualized st. dev. of active return
    information_ratio: Decimal | None

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class ReturnMoments:
    """Annualized sample moments for assets in ``asset_ids`` order."""

    asset_ids: tuple[str, ...]
    observations: int
    periods_per_year: int
    mean: tuple[Decimal, ...]
    covariance: tuple[tuple[Decimal, ...], ...]
    covariance_full_rank_possible: bool

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


class PerformanceMeasurement:
    """Return measurement and sample-moment estimation at a fixed precision."""

    def __init__(self, *, precision: int):
        self.precision = require_positive_int(precision, "precision", minimum=16, maximum=256)

    # ---------------------------------------------------------------- helpers
    def _returns(self, values: Sequence[Any], name: str, *, minimum: int) -> list[Decimal]:
        if not isinstance(values, (list, tuple)) or not minimum <= len(values) <= 1_000_000:
            raise FinancialValidationError(f"{name} must be a bounded list/tuple of at least {minimum} returns")
        out = [to_decimal(v, name) for v in values]
        if any(v <= -1 for v in out):
            raise FinancialDomainError("A period return cannot be -100% or lower")
        return out

    def _mean(self, xs: Sequence[Decimal]) -> Decimal:
        with localcontext() as ctx:
            ctx.prec = self.precision
            return sum(xs, Decimal(0)) / Decimal(len(xs))

    def _cov(self, xs: Sequence[Decimal], ys: Sequence[Decimal]) -> Decimal:
        with localcontext() as ctx:
            ctx.prec = self.precision
            mx, my = self._mean(xs), self._mean(ys)
            return sum(((x - mx) * (y - my) for x, y in zip(xs, ys)), Decimal(0)) / Decimal(len(xs) - 1)

    def _sqrt(self, value: Decimal) -> Decimal:
        with localcontext() as ctx:
            ctx.prec = self.precision
            return value.sqrt()

    @staticmethod
    def _periods(value: Any) -> int:
        return require_positive_int(value, "periods_per_year", maximum=366 * 24 * 60)

    # ----------------------------------------------------------------- returns
    def time_weighted_return(self, valuations: Sequence[ValuationPoint], flows: Sequence[ExternalFlow] = ()) -> TimeWeightedResult:
        if not isinstance(valuations, (list, tuple)) or not 2 <= len(valuations) <= 100_000:
            raise FinancialValidationError("TWR needs a bounded list of at least two valuations")
        if any(not isinstance(v, ValuationPoint) for v in valuations):
            raise FinancialValidationError("valuations must contain ValuationPoint values")
        if not isinstance(flows, (list, tuple)) or any(not isinstance(f, ExternalFlow) for f in flows):
            raise FinancialValidationError("flows must be a list/tuple of ExternalFlow values")
        code = valuations[0].value.currency
        for item in (*(v.value for v in valuations), *(f.amount for f in flows)):
            require_same_currency(code, item.currency, operation="time-weighted return")
        if any(a.at >= b.at for a, b in zip(valuations, valuations[1:])):
            raise FinancialValidationError("Valuations must have strictly increasing times")
        flow_at: dict[datetime, Decimal] = {}
        for flow in flows:
            flow_at[flow.at] = flow_at.get(flow.at, Decimal(0)) + flow.amount.amount
        known = {v.at for v in valuations}
        orphan = [t for t in flow_at if t not in known]
        if orphan:
            raise FinancialDomainError("Every external flow needs a valuation at the same instant (GIPS)",
                                       context={"unmatched_flows": len(orphan)})
        periods: list[Decimal] = []
        total = Decimal(1)
        try:
            with localcontext() as ctx:
                ctx.prec = self.precision
                base = valuations[0].value.amount + flow_at.get(valuations[0].at, Decimal(0))
                for point in valuations[1:]:
                    if base <= 0:
                        raise FinancialDomainError("Sub-period starting capital must be positive for a time-weighted return")
                    r = point.value.amount / base - 1
                    periods.append(r)
                    total *= 1 + r
                    base = point.value.amount + flow_at.get(point.at, Decimal(0))
        except DecimalException as exc:
            raise FinancialPrecisionError("Time-weighted return failed", cause=exc) from exc
        return TimeWeightedResult(total - 1, tuple(periods), valuations[0].at, valuations[-1].at)

    def modified_dietz(self, begin: ValuationPoint, end: ValuationPoint, flows: Sequence[ExternalFlow] = ()) -> Decimal:
        """(EMV - BMV - sum F) / (BMV + sum w_i F_i), w_i = (T - t_i) / T in seconds."""
        if not isinstance(begin, ValuationPoint) or not isinstance(end, ValuationPoint):
            raise FinancialValidationError("begin and end must be ValuationPoint values")
        if begin.at >= end.at:
            raise FinancialDomainError("Modified Dietz requires begin before end")
        if not isinstance(flows, (list, tuple)) or any(not isinstance(f, ExternalFlow) for f in flows):
            raise FinancialValidationError("flows must be a list/tuple of ExternalFlow values")
        for item in (end.value, *(f.amount for f in flows)):
            require_same_currency(begin.value.currency, item.currency, operation="Modified Dietz")
        if any(not begin.at < f.at < end.at for f in flows):
            raise FinancialDomainError("Flows must fall strictly inside the measurement period; "
                                       "fold boundary flows into the begin value or exclude them from the end value")
        span = Decimal(str((end.at - begin.at).total_seconds()))
        with localcontext() as ctx:
            ctx.prec = self.precision
            net = sum((f.amount.amount for f in flows), Decimal(0))
            weighted = sum((f.amount.amount * Decimal(str((end.at - f.at).total_seconds())) / span for f in flows), Decimal(0))
            denominator = begin.value.amount + weighted
            if denominator <= 0:
                raise FinancialDomainError("Modified Dietz denominator (average capital) must be positive")
            return (end.value.amount - begin.value.amount - net) / denominator

    @staticmethod
    def investor_cash_flows(begin: ValuationPoint, end: ValuationPoint, flows: Sequence[ExternalFlow] = ()) -> tuple[CashFlow, ...]:
        """Investor-perspective flows for a money-weighted return (IRR), from FinancialCalculation.xirr.

        Opening value and contributions are outflows (negative), withdrawals and
        the closing value are inflows (positive).
        """
        items = [CashFlow(begin.at.date(), FinancialAmount(-begin.value.amount, begin.value.currency), category="opening_value")]
        items += [CashFlow(f.at.date(), FinancialAmount(-f.amount.amount, f.amount.currency), category="external_flow") for f in flows]
        items.append(CashFlow(end.at.date(), end.value, category="closing_value"))
        return tuple(sorted(items, key=lambda c: c.on_date))

    def annualize_return(self, total_return: Any, years: Any, *, allow_short_period: bool = False) -> Decimal:
        r, y = to_decimal(total_return, "total_return"), to_decimal(years, "years")
        require_bool(allow_short_period, "allow_short_period")
        if r <= -1:
            raise FinancialDomainError("A return of -100% or lower cannot be annualized")
        if y <= 0:
            raise FinancialDomainError("Years must be positive")
        if y < 1 and not allow_short_period:
            raise FinancialDomainError("GIPS: returns for periods under one year are not annualized")
        try:
            with localcontext() as ctx:
                ctx.prec = self.precision
                return (1 + r) ** (1 / y) - 1
        except DecimalException as exc:
            raise FinancialPrecisionError("Annualization failed", cause=exc) from exc

    # -------------------------------------------------------------- statistics
    def statistics(self, returns: Sequence[Any], *, periods_per_year: int,
                   risk_free_per_period: Any = "0") -> ReturnStatistics:
        p = self._periods(periods_per_year)
        xs = self._returns(returns, "returns", minimum=2)
        rf = to_decimal(risk_free_per_period, "risk_free_per_period")
        with localcontext() as ctx:
            ctx.prec = self.precision
            mean = self._mean(xs)
            vol = self._sqrt(self._cov(xs, xs))
            sharpe = ((mean - rf) / vol) * self._sqrt(Decimal(p)) if vol > 0 else None
            return ReturnStatistics(len(xs), mean, vol, mean * p, vol * self._sqrt(Decimal(p)), sharpe)

    def relative_statistics(self, returns: Sequence[Any], benchmark: Sequence[Any], *, periods_per_year: int,
                            risk_free_per_period: Any = "0") -> RelativeStatistics:
        p = self._periods(periods_per_year)
        xs = self._returns(returns, "returns", minimum=3)
        bs = self._returns(benchmark, "benchmark", minimum=3)
        if len(xs) != len(bs):
            raise FinancialValidationError("returns and benchmark must have equal length")
        rf = to_decimal(risk_free_per_period, "risk_free_per_period")
        with localcontext() as ctx:
            ctx.prec = self.precision
            active = [x - b for x, b in zip(xs, bs)]
            te = self._sqrt(self._cov(active, active)) * self._sqrt(Decimal(p))
            var_b = self._cov(bs, bs)
            beta = self._cov(xs, bs) / var_b if var_b > 0 else None
            alpha = (self._mean(xs) - rf) - beta * (self._mean(bs) - rf) if beta is not None else None
            info = self._mean(active) * p / te if te > 0 else None
            return RelativeStatistics(len(xs), beta, alpha, te, info)

    def estimate_moments(self, columns: Mapping[str, Sequence[Any]], *, periods_per_year: int) -> ReturnMoments:
        """Annualized sample mean vector and covariance matrix from aligned return columns."""
        p = self._periods(periods_per_year)
        if not isinstance(columns, Mapping) or not 1 <= len(columns) <= 1024:
            raise FinancialValidationError("columns must map 1 to 1024 asset ids to return sequences")
        ids = tuple(sorted(normalize_identifier(k, "asset_id") for k in columns))
        data = {normalize_identifier(k, "asset_id"): self._returns(v, "returns", minimum=3) for k, v in columns.items()}
        n = len(data[ids[0]])
        if any(len(v) != n for v in data.values()):
            raise FinancialValidationError("All return columns must have the same length (align first)")
        with localcontext() as ctx:
            ctx.prec = self.precision
            mean = tuple(self._mean(data[a]) * p for a in ids)
            cov = tuple(tuple(self._cov(data[a], data[b]) * p for b in ids) for a in ids)
        return ReturnMoments(ids, n, p, mean, cov, n > len(ids))


__all__ = ["ValuationPoint", "ExternalFlow", "TimeWeightedResult", "ReturnStatistics", "RelativeStatistics",
           "ReturnMoments", "PerformanceMeasurement"]

if __name__ == "__main__":
    print("\n=== Running Financial Performance ===\n")
    printer.status("TEST", "Financial Performance initialized", "info")
    from datetime import timedelta, timezone
    perf = PerformanceMeasurement(precision=36)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    eur = lambda v: FinancialAmount.of(v, "EUR")  # noqa: E731
    close = lambda a, b, tol="1e-25": abs(a - Decimal(b)) < Decimal(tol)  # noqa: E731
    # TWR: 100 -> 110 (+10%), add 100, 210 -> 231 (+10%): TWR 21%, though money-weighted differs.
    points = [ValuationPoint(t0, eur(0)), ValuationPoint(t0 + timedelta(days=30), eur(110)),
              ValuationPoint(t0 + timedelta(days=60), eur(231))]
    flows = [ExternalFlow(t0, eur(100)), ExternalFlow(t0 + timedelta(days=30), eur(100))]
    twr = perf.time_weighted_return(points, flows)
    assert close(twr.total_return, "0.21") and [close(r, "0.1") for r in twr.period_returns] == [True, True]
    try:
        perf.time_weighted_return(points, [ExternalFlow(t0 + timedelta(days=1), eur(5))])
    except FinancialDomainError:
        pass
    else:
        raise AssertionError("flows without a same-instant valuation must be rejected")
    # Modified Dietz: BMV 100, EMV 130, flow +10 at the midpoint: 20 / 105.
    dietz = perf.modified_dietz(ValuationPoint(t0, eur(100)), ValuationPoint(t0 + timedelta(days=10), eur(130)),
                                [ExternalFlow(t0 + timedelta(days=5), eur(10))])
    assert close(dietz, Decimal(20) / Decimal(105), "1e-25")
    mw = perf.investor_cash_flows(ValuationPoint(t0, eur(100)), ValuationPoint(t0 + timedelta(days=365), eur(110)), [])
    assert [c.amount.amount for c in mw] == [-100, 110]
    assert close(perf.annualize_return("0.21", 2), "0.1")
    try:
        perf.annualize_return("0.05", "0.5")
    except FinancialDomainError:
        pass
    else:
        raise AssertionError("GIPS forbids annualizing periods under one year")
    assert close(perf.annualize_return("0.05", "0.5", allow_short_period=True), "0.1025")
    rs = [Decimal(x) for x in ("0.01", "0.02", "-0.01", "0.03", "0.00")]
    stats = perf.statistics(rs, periods_per_year=12, risk_free_per_period="0.001")
    assert close(stats.mean, "0.01") and close(stats.volatility, Decimal("0.0158113883008418966599944677221635241"), "1e-25")
    assert close(stats.sharpe_ratio, (Decimal("0.009") / Decimal("0.0158113883008418966599944677221635241")) * Decimal(12).sqrt(), "1e-20")
    rel = perf.relative_statistics(rs, rs, periods_per_year=12)
    assert close(rel.beta, 1) and close(rel.alpha, 0) and close(rel.tracking_error, 0) and rel.information_ratio is None
    scaled = [Decimal(2) * x for x in rs]
    assert close(perf.relative_statistics(scaled, rs, periods_per_year=12).beta, 2)
    moments = perf.estimate_moments({"B": scaled, "A": rs}, periods_per_year=12)
    assert moments.asset_ids == ("A", "B") and moments.covariance_full_rank_possible
    assert close(moments.covariance[0][1] * 2, moments.covariance[1][1], "1e-25") and close(moments.mean[1], moments.mean[0] * 2)
    print("\n=== Test ran successfully ===\n")
