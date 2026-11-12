"""Validation, return construction and alignment of observation series.

Single owner of the rules that make a list of ``MarketObservation`` usable as a
time series: homogeneity (one instrument, field, currency and unit), no
look-ahead against an as-of instant, strictly increasing valid time, and
bounded length. Used by FinancialIntelligence (forecast history), by
FinancialEconomics (economic series and price returns) and by
FinancialPortfolio (return histories feeding risk and allocation analysis).

Returns are price-only: they exclude dividends, splits, taxes and costs unless
the provider pre-adjusted the series (a total-return index is the right input
for total return). Missing data are never interpolated or filled.
"""
from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, DecimalException, localcontext
from typing import Any

from ..financial_types import MarketObservation
from ..utils.financial_errors import *
from ..utils.financial_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Financial Series")
printer = PrettyPrinter()

RETURN_KINDS = ("simple", "log")


def validate_series(
    observations: Sequence[MarketObservation], *, as_of: datetime | str,
    max_points: int, min_points: int = 1,
) -> tuple[tuple[MarketObservation, ...], datetime]:
    """Return the validated series and the normalized as-of instant.

    Rejects mixed instrument/field/currency/unit, facts unknown at ``as_of``
    (valid time or knowledge time in the future) and non-increasing valid time.
    """
    require_positive_int(max_points, "max_points", maximum=10_000_000)
    require_positive_int(min_points, "min_points", maximum=max_points)
    if not isinstance(observations, (list, tuple)) or not min_points <= len(observations) <= max_points:
        raise FinancialValidationError(
            "history must be a bounded list/tuple within the permitted length",
            context={"min_points": min_points, "max_points": max_points})
    if any(not isinstance(x, MarketObservation) for x in observations):
        raise FinancialValidationError("history must contain MarketObservation values")
    now = to_utc_datetime(as_of, "as_of")
    first = observations[0]
    if any((x.instrument_id, x.field_name, x.currency, x.unit) !=
           (first.instrument_id, first.field_name, first.currency, first.unit) for x in observations):
        raise InvalidMarketDataError("Forecast history mixes instrument, field, currency or unit")
    if any(x.observed_at > now or x.source.retrieved_at > now for x in observations):
        raise InvalidMarketDataError("Forecast history includes data unavailable at as_of")
    if any(left.observed_at >= right.observed_at for left, right in zip(observations, observations[1:])):
        raise InvalidMarketDataError("Forecast history requires unique increasing observed_at timestamps")
    return tuple(observations), now


@dataclass(frozen=True, slots=True)
class ReturnSeries:
    """Period returns; ``times[i]`` is the end of the period earning ``values[i]``."""

    instrument_id: str
    field_name: str
    kind: str
    times: tuple[datetime, ...]
    values: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", normalize_identifier(self.instrument_id, "instrument_id"))
        object.__setattr__(self, "field_name", normalize_identifier(self.field_name, "field_name"))
        if self.kind not in RETURN_KINDS:
            raise UnsupportedConventionError("Return kind must be simple or log")
        times = tuple(to_utc_datetime(t, "time") for t in self.times)
        values = tuple(to_decimal(v, "return") for v in self.values)
        if len(times) != len(values):
            raise FinancialValidationError("times and values must have equal length")
        if any(a >= b for a, b in zip(times, times[1:])):
            raise FinancialValidationError("Return times must be strictly increasing")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)

    def __len__(self) -> int:
        return len(self.values)

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


def price_returns(
    observations: Sequence[MarketObservation], *, as_of: datetime | str, precision: int,
    kind: str = "simple", max_points: int = 1_000_000,
) -> ReturnSeries:
    """Period-over-period price returns of a validated, currency-qualified series.

    simple: P_t / P_(t-1) - 1; log: ln(P_t / P_(t-1)). Both need strictly
    positive prices. Observation spacing is not assumed regular; annualization
    requires an explicit period count at the consumer.
    """
    if kind not in RETURN_KINDS:
        raise UnsupportedConventionError("Return kind must be simple or log", context={"kind": str(kind)[:16]})
    require_positive_int(precision, "precision", minimum=16, maximum=256)
    series, _ = validate_series(observations, as_of=as_of, max_points=max_points, min_points=2)
    if series[0].currency is None:
        raise FinancialValidationError("Price returns require currency-qualified observations")
    if any(x.value <= 0 for x in series):
        raise FinancialDomainError("Return construction requires strictly positive prices")
    values: list[Decimal] = []
    try:
        with localcontext() as ctx:
            ctx.prec = precision
            for prev, cur in zip(series, series[1:]):
                ratio = cur.value / prev.value
                values.append(ratio - 1 if kind == "simple" else ratio.ln())
    except DecimalException as exc:
        raise FinancialPrecisionError("Return computation failed", cause=exc) from exc
    return ReturnSeries(series[0].instrument_id, series[0].field_name, kind,
                        tuple(x.observed_at for x in series[1:]), tuple(values))


def align_returns(series: Sequence[ReturnSeries]) -> tuple[tuple[datetime, ...], dict[str, tuple[Decimal, ...]]]:
    """Restrict to the return periods shared by every series (inner join).

    The key of each column is the ``instrument_id``. No observation is filled,
    interpolated or carried forward; identical identifiers are rejected.
    """
    if not isinstance(series, (list, tuple)) or not series or any(not isinstance(s, ReturnSeries) for s in series):
        raise FinancialValidationError("align_returns requires a nonempty list of ReturnSeries")
    if len({s.instrument_id for s in series}) != len(series):
        raise FinancialIntegrityError("Duplicate instrument in return alignment")
    if len({s.kind for s in series}) != 1:
        raise FinancialValidationError("Cannot align simple and log returns together")
    shared = sorted(set.intersection(*(set(s.times) for s in series)))
    if not shared:
        raise InvalidMarketDataError("Return series share no common period end")
    columns: dict[str, tuple[Decimal, ...]] = {}
    for s in series:
        lookup = dict(zip(s.times, s.values))
        columns[s.instrument_id] = tuple(lookup[t] for t in shared)
    return tuple(shared), columns


def observation_as_of(
    observations: Sequence[MarketObservation], when: datetime | str, *, max_age: timedelta,
    known_by: datetime | str | None = None,
) -> MarketObservation:
    """Last observation valid at or before ``when`` (last-value-carried-forward, bounded).

    The carried value must not be older than ``max_age`` at ``when``. Knowledge
    time is checked against ``known_by`` (default: ``when``, i.e. a live read).
    Pass a later ``known_by`` to ask "what was the level at ``when``, as known
    by ``known_by``" (a revised or late-published series).
    """
    moment = to_utc_datetime(when, "when")
    cutoff = moment if known_by is None else to_utc_datetime(known_by, "known_by")
    if not isinstance(max_age, timedelta) or max_age <= timedelta(0):
        raise FinancialValidationError("max_age must be a positive timedelta")
    candidates = [x for x in observations if x.observed_at <= moment and x.source.retrieved_at <= cutoff]
    if not candidates:
        raise MissingMarketDataError("No observation available at the requested time")
    latest = max(candidates, key=lambda x: (x.observed_at, x.source.retrieved_at))
    latest.assert_fresh(as_of=moment, max_age=max_age)
    return latest


__all__ = ["RETURN_KINDS", "ReturnSeries", "validate_series", "price_returns", "align_returns", "observation_as_of"]

if __name__ == "__main__":
    print("\n=== Running Financial Series ===\n")
    printer.status("TEST", "Financial Series initialized", "info")
    from datetime import timezone
    from ..financial_types import FinancialDataSource
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    src = FinancialDataSource("p", base + timedelta(days=30))

    def px(name: str, day: int, value: str) -> MarketObservation:
        return MarketObservation(name, "close", Decimal(value), base + timedelta(days=day), src, currency="EUR")

    a = [px("A", 0, "100"), px("A", 1, "110"), px("A", 2, "99")]
    b = [px("B", 1, "50"), px("B", 2, "55"), px("B", 3, "55")]
    ra = price_returns(a, as_of=base + timedelta(days=30), precision=36)
    assert ra.values == (Decimal("0.1"), Decimal("-0.1"))
    rl = price_returns(a, as_of=base + timedelta(days=30), precision=36, kind="log")
    assert abs(rl.values[0] - Decimal("0.0953101798043248600439521232807650922")) < Decimal("1e-30")
    times, cols = align_returns([ra, price_returns(b, as_of=base + timedelta(days=30), precision=36)])
    assert len(times) == 1 and cols["A"] == (Decimal("-0.1"),) and cols["B"] == (Decimal("0.1"),)
    try:
        validate_series(a, as_of=base + timedelta(days=1), max_points=10)
    except InvalidMarketDataError:
        pass
    else:
        raise AssertionError("retrieval after as_of must be rejected")
    try:
        validate_series(list(reversed(a)), as_of=base + timedelta(days=30), max_points=10)
    except InvalidMarketDataError:
        pass
    else:
        raise AssertionError("non-increasing series must be rejected")
    assert observation_as_of(a, base + timedelta(days=30), max_age=timedelta(days=40)).value == Decimal("99")
    assert observation_as_of(a, base + timedelta(days=2), max_age=timedelta(days=5), known_by=base + timedelta(days=30)).value == Decimal("99")
    try:
        observation_as_of(a, base + timedelta(days=2), max_age=timedelta(days=5))
    except MissingMarketDataError:
        pass
    else:
        raise AssertionError("late-published data must be invisible to a live read")
    try:
        observation_as_of(a, base + timedelta(days=30), max_age=timedelta(days=1))
    except StaleMarketDataError:
        pass
    else:
        raise AssertionError("stale carry-forward must be rejected")
    print("\n=== Test ran successfully ===\n")
