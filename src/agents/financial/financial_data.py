"""Provider-neutral, revision-aware financial observation store for SLAI v2.4.

Stores supplied validated MarketObservation/ExchangeRate facts only. No HTTP,
SDKs, guessed prices, implicit FX arithmetic, brokerage operations or auth.
As-of reads enforce both observed_at and retrieved_at to prevent look-ahead.
Use an authorized, durable ingestion backend outside this module in production
when market data must survive process restarts.
"""
from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from .data.fx import FxConverter
from .data.observation_store import BitemporalStore
from .financial_types import ExchangeRate, MarketObservation
from .utils.config_loader import load_global_config, get_config_section
from .utils.financial_errors import *
from .utils.financial_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Financial Data")
printer = PrettyPrinter()


class FinancialData:
    """Thread-safe point-in-time observations with immutable value contracts.

    Identical keys with conflicting payloads are rejected transactionally.
    A revision identifier belongs to the source; an ingest with a new revision
    is a distinct observation and remains available for audit/as-of queries.
    Visibility, conflict and retention rules live in ``data.observation_store``;
    this class adds configuration, freshness policy and the public contract.
    """

    def __init__(self):
        self.config = load_global_config()
        self.financial_data_config = get_config_section("financial_data", config=self.config)
        try:
            self.max_observations_per_series = require_positive_int(
                self.financial_data_config["max_observations_per_series"], "max_observations_per_series", maximum=1000000)
            self.max_age_seconds = require_positive_int(
                self.financial_data_config["max_age_seconds"], "max_age_seconds", maximum=315360000)
        except KeyError as exc:
            raise FinancialConfigurationError("Missing financial_data configuration key", context={"key": str(exc)}) from exc
        self._observations: BitemporalStore[MarketObservation] = BitemporalStore(
            MarketObservation, lambda v: (v.instrument_id, v.field_name),
            max_per_bucket=self.max_observations_per_series)
        self._fx: BitemporalStore[ExchangeRate] = BitemporalStore(
            ExchangeRate, lambda v: (v.base_currency.code, v.quote_currency.code),  # type: ignore[union-attr]
            max_per_bucket=self.max_observations_per_series)

    def ingest_observations(self, observations: Sequence[MarketObservation]) -> int:
        """Atomically ingest market observations; exact replay is idempotent."""
        return self._observations.ingest(observations)  # type: ignore[arg-type]

    def ingest_exchange_rates(self, rates: Sequence[ExchangeRate]) -> int:
        """Atomically ingest positive, sourced FX quotes (no inversion)."""
        return self._fx.ingest(rates)  # type: ignore[arg-type]

    @staticmethod
    def _as_of(as_of: datetime | str | None) -> datetime:
        return datetime.now(timezone.utc) if as_of is None else to_utc_datetime(as_of, "as_of")

    def _freshness(self, value: Any, as_of: datetime, max_age: timedelta | None) -> None:
        duration = timedelta(seconds=self.max_age_seconds) if max_age is None else max_age
        if not isinstance(duration, timedelta) or duration <= timedelta(0):
            raise FinancialValidationError("max_age must be positive timedelta")
        if value.observed_at > as_of or value.source.retrieved_at > as_of:
            raise InvalidMarketDataError("Observation is not available at the requested as-of time")
        if as_of - value.observed_at > duration:
            raise StaleMarketDataError("Financial observation exceeds freshness window", context={"max_age_seconds": duration.total_seconds()})

    def latest_observation(self, instrument_id: str, field_name: str, *,
                           as_of: datetime | str | None = None, max_age: timedelta | None = None,
                           source_id: str | None = None) -> MarketObservation:
        inst = normalize_identifier(instrument_id, "instrument_id")
        field = normalize_identifier(field_name, "field_name")
        src = None if source_id is None else normalize_identifier(source_id, "source_id")
        time = self._as_of(as_of)
        latest = self._observations.latest((inst, field), as_of=time, source_id=src)
        if latest is None:
            raise MissingMarketDataError("No financial observation available at requested point in time", context={"field": field})
        self._freshness(latest, time, max_age)
        return latest

    def latest_exchange_rate(self, base_currency: str, quote_currency: str, *,
                             as_of: datetime | str | None = None, max_age: timedelta | None = None,
                             source_id: str | None = None) -> ExchangeRate:
        base, quote = normalize_currency_code(base_currency), normalize_currency_code(quote_currency)
        if base == quote:
            raise FinancialValidationError("FX pair requires distinct currencies")
        src = None if source_id is None else normalize_identifier(source_id, "source_id")
        time = self._as_of(as_of)
        latest = self._fx.latest((base, quote), as_of=time, source_id=src)
        if latest is None:
            raise MissingMarketDataError("No sourced FX quote available at requested point in time")
        self._freshness(latest, time, max_age)
        return latest

    def observation_history(self, instrument_id: str, field_name: str, *,
                            from_time: datetime | str, until_time: datetime | str,
                            as_of: datetime | str | None = None, source_id: str | None = None,
                            vintage: str = "all") -> tuple[MarketObservation, ...]:
        """Observations with ``from_time <= observed_at < until_time`` known at ``as_of``.

        ``vintage="all"`` (default) returns every revision for audit;
        ``vintage="latest"`` returns one value per valid time, the newest
        revision known at ``as_of`` (real-time-data semantics).
        """
        bucket = (normalize_identifier(instrument_id, "instrument_id"), normalize_identifier(field_name, "field_name"))
        src = None if source_id is None else normalize_identifier(source_id, "source_id")
        return self._observations.history(
            bucket, start=to_utc_datetime(from_time, "from_time"), end=to_utc_datetime(until_time, "until_time"),
            as_of=self._as_of(as_of), source_id=src, vintage=vintage)

    def fx_converter(self, *, precision: int) -> FxConverter:
        """Converter bound to this store; precision is owned by FinancialCalculation."""
        return FxConverter(self.latest_exchange_rate, precision=precision)

    def stored_observations(self) -> tuple[MarketObservation, ...]:
        """All stored observations (audit and checkpoint export)."""
        return self._observations.items()

    def stored_exchange_rates(self) -> tuple[ExchangeRate, ...]:
        """All stored FX quotes (audit and checkpoint export)."""
        return self._fx.items()

    def counts(self) -> dict[str, int]:
        series, observations = self._observations.counts()
        pairs, quotes = self._fx.counts()
        return {"market_series": series, "market_observations": observations,
                "fx_pairs": pairs, "fx_observations": quotes}

    def clear(self) -> None:
        self._observations.clear()
        self._fx.clear()


__all__ = ["FinancialData"]

if __name__ == "__main__":
    print("\n=== Running Financial Data ===\n")
    printer.status("TEST", "Financial Data initialized", "info")
    from datetime import timedelta
    from .financial_types import FinancialDataSource
    from decimal import Decimal
    engine = FinancialData()
    now = datetime.now(timezone.utc)
    source = FinancialDataSource("provider", now)
    sample = MarketObservation("ASSET", "close", Decimal("25.60"), now - timedelta(seconds=1), source, currency="EUR")
    assert engine.ingest_observations([sample]) == 1
    assert engine.ingest_observations([sample]) == 0
    assert engine.latest_observation("ASSET", "close", as_of=now).value == Decimal("25.60")
    assert engine.counts()["market_observations"] == 1
    assert len(engine.observation_history("ASSET", "close", from_time=now-timedelta(days=1), until_time=now+timedelta(seconds=1))) == 1
    revised = MarketObservation("ASSET", "close", Decimal("25.70"), now - timedelta(seconds=1),
                                FinancialDataSource("provider", now + timedelta(seconds=5), revision="r2"), currency="EUR")
    assert engine.ingest_observations([revised]) == 1
    window = dict(from_time=now - timedelta(days=1), until_time=now + timedelta(seconds=1))
    assert len(engine.observation_history("ASSET", "close", as_of=now + timedelta(seconds=9), **window)) == 2
    latest_vintage = engine.observation_history("ASSET", "close", as_of=now + timedelta(seconds=9), vintage="latest", **window)
    assert [x.value for x in latest_vintage] == [Decimal("25.70")]
    assert engine.latest_observation("ASSET", "close", as_of=now).value == Decimal("25.60"), "revision unknown at as_of"
    engine.clear()
    assert engine.counts()["market_observations"] == 0
    print("\n=== Test ran successfully ===\n")
