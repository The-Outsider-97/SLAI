"""Explicit, auditable currency conversion over sourced exchange-rate quotes.

Conversion is never implicit. By default only a *direct* sourced quote
``base -> quote`` is used. Reciprocals and a single pivot (triangulation) are
available only when the caller opts in through ``FxPolicy``, and every
conversion records the legs and the data sources it relied on. Quotes are
treated as mid rates: no bid/ask spread, fee or settlement lag is modeled, and
results are not rounded (see business.rounding for explicit quantization).

The converter is storage-agnostic: it is given a ``lookup`` callable with the
signature of ``FinancialData.latest_exchange_rate`` so that point-in-time
visibility (observed_at and retrieved_at) stays owned by the observation store.
"""
from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, DecimalException, localcontext
from typing import Any

from ..financial_types import ExchangeRate, FinancialAmount, FinancialDataSource
from ..utils.financial_errors import *
from ..utils.financial_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Financial FX")
printer = PrettyPrinter()

RateLookup = Callable[..., ExchangeRate]


@dataclass(frozen=True, slots=True)
class FxPolicy:
    """Opt-in conventions for resolving a rate when no direct quote exists."""

    allow_inverse: bool = False
    pivot_currency: str | None = None
    max_age: timedelta | None = None
    source_id: str | None = None

    def __post_init__(self) -> None:
        require_bool(self.allow_inverse, "allow_inverse")
        if self.pivot_currency is not None:
            object.__setattr__(self, "pivot_currency", normalize_currency_code(self.pivot_currency))
        if self.max_age is not None and (not isinstance(self.max_age, timedelta) or self.max_age <= timedelta(0)):
            raise FinancialValidationError("max_age must be a positive timedelta")
        if self.source_id is not None:
            object.__setattr__(self, "source_id", normalize_identifier(self.source_id, "source_id"))


@dataclass(frozen=True, slots=True)
class FxLeg:
    """One quote used in a conversion; ``inverted`` marks a reciprocal use."""

    base: str
    quote: str
    quoted_rate: Decimal
    inverted: bool
    observed_at: datetime
    source: FinancialDataSource

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class FxConversion:
    original: FinancialAmount
    converted: FinancialAmount
    rate: Decimal
    as_of: datetime
    legs: tuple[FxLeg, ...]

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)

    @property
    def sources(self) -> tuple[FinancialDataSource, ...]:
        return tuple(leg.source for leg in self.legs)


class FxConverter:
    """Resolve and apply exchange rates as of an instant."""

    def __init__(self, lookup: RateLookup, *, precision: int):
        if not callable(lookup):
            raise FinancialValidationError("lookup must be callable")
        self._lookup = lookup
        self.precision = require_positive_int(precision, "precision", minimum=16, maximum=256)

    def _leg(self, base: str, quote: str, as_of: datetime, policy: FxPolicy, *, allow_inverse: bool) -> FxLeg:
        failure: FinancialError | None = None
        attempts = [(base, quote, False)] + ([(quote, base, True)] if allow_inverse else [])
        for b, q, inverted in attempts:
            try:
                fx = self._lookup(b, q, as_of=as_of, max_age=policy.max_age, source_id=policy.source_id)
            except (MissingMarketDataError, StaleMarketDataError) as exc:
                if failure is None or isinstance(exc, StaleMarketDataError):
                    failure = exc
                continue
            return FxLeg(b, q, fx.rate, inverted, fx.observed_at, fx.source)
        raise failure if failure is not None else MissingMarketDataError("No FX quote available")

    def _chain(self, base: str, quote: str, as_of: datetime, policy: FxPolicy) -> tuple[FxLeg, ...]:
        try:
            return (self._leg(base, quote, as_of, policy, allow_inverse=policy.allow_inverse),)
        except (MissingMarketDataError, StaleMarketDataError) as direct_failure:
            pivot = policy.pivot_currency
            if pivot is None or pivot in (base, quote):
                raise direct_failure
            try:
                return (self._leg(base, pivot, as_of, policy, allow_inverse=policy.allow_inverse),
                        self._leg(pivot, quote, as_of, policy, allow_inverse=policy.allow_inverse))
            except (MissingMarketDataError, StaleMarketDataError) as exc:
                raise exc from direct_failure

    def rate(self, base: str, quote: str, *, as_of: datetime | str, policy: FxPolicy | None = None) -> tuple[Decimal, tuple[FxLeg, ...]]:
        """Units of ``quote`` per one unit of ``base`` and the legs that justify it."""
        b, q = normalize_currency_code(base), normalize_currency_code(quote)
        moment = to_utc_datetime(as_of, "as_of")
        if policy is not None and not isinstance(policy, FxPolicy):
            raise FinancialValidationError("policy must be FxPolicy or None")
        if b == q:
            return Decimal(1), ()
        legs = self._chain(b, q, moment, policy or FxPolicy())
        try:
            with localcontext() as ctx:
                ctx.prec = self.precision
                result = Decimal(1)
                for leg in legs:
                    result *= (1 / leg.quoted_rate) if leg.inverted else leg.quoted_rate
        except DecimalException as exc:
            raise FinancialPrecisionError("FX rate composition failed", cause=exc) from exc
        return result, legs

    def convert(self, amount: FinancialAmount, to_currency: str, *, as_of: datetime | str,
                policy: FxPolicy | None = None) -> FxConversion:
        if not isinstance(amount, FinancialAmount):
            raise FinancialValidationError("amount must be FinancialAmount")
        target = normalize_currency_code(to_currency)
        moment = to_utc_datetime(as_of, "as_of")
        rate, legs = self.rate(amount.currency.code, target, as_of=moment, policy=policy)  # type: ignore[union-attr]
        with localcontext() as ctx:
            ctx.prec = self.precision
            converted = FinancialAmount(amount.amount * rate, target)
        return FxConversion(amount, converted, rate, moment, legs)

    def convert_total(self, amounts: Sequence[FinancialAmount], to_currency: str, *, as_of: datetime | str,
                      policy: FxPolicy | None = None) -> tuple[FinancialAmount, tuple[FxConversion, ...]]:
        """Sum amounts after explicit conversion; one conversion record per input."""
        if not isinstance(amounts, (list, tuple)):
            raise FinancialValidationError("amounts must be a list/tuple")
        target = normalize_currency_code(to_currency)
        total = FinancialAmount.of(0, target)
        records: list[FxConversion] = []
        for amount in amounts:
            record = self.convert(amount, target, as_of=as_of, policy=policy)
            total += record.converted
            records.append(record)
        return total, tuple(records)


__all__ = ["FxPolicy", "FxLeg", "FxConversion", "FxConverter"]

if __name__ == "__main__":
    print("\n=== Running Financial FX ===\n")
    printer.status("TEST", "Financial FX initialized", "info")
    from datetime import timezone
    now = datetime(2024, 3, 1, tzinfo=timezone.utc)
    src = FinancialDataSource("ecb-like", now)
    quotes = {("EUR", "USD"): Decimal("1.25"), ("EUR", "GBP"): Decimal("0.8")}

    def lookup(base: str, quote: str, *, as_of: datetime, max_age: timedelta | None = None, source_id: str | None = None) -> ExchangeRate:
        if (base, quote) not in quotes:
            raise MissingMarketDataError("No sourced FX quote available at requested point in time")
        return ExchangeRate(base, quote, quotes[(base, quote)], now - timedelta(hours=1), src)

    conv = FxConverter(lookup, precision=36)
    eur = FinancialAmount.of("100", "EUR")
    assert conv.convert(eur, "USD", as_of=now).converted.amount == Decimal("125.00")
    assert conv.convert(eur, "EUR", as_of=now).legs == ()
    for policy, message in ((None, "reciprocal"), (FxPolicy(), "cross")):
        try:
            conv.convert(FinancialAmount.of("125", "USD"), "EUR", as_of=now, policy=policy)
        except MissingMarketDataError:
            pass
        else:
            raise AssertionError(f"{message} rate must not be inferred by default")
    back = conv.convert(FinancialAmount.of("125", "USD"), "EUR", as_of=now, policy=FxPolicy(allow_inverse=True))
    assert back.converted.amount == Decimal("100") and back.legs[0].inverted
    cross = conv.convert(FinancialAmount.of("125", "USD"), "GBP", as_of=now, policy=FxPolicy(allow_inverse=True, pivot_currency="EUR"))
    assert cross.converted.amount == Decimal("80") and len(cross.legs) == 2
    total, records = conv.convert_total([eur, FinancialAmount.of("125", "USD")], "EUR", as_of=now, policy=FxPolicy(allow_inverse=True))
    assert total.amount == Decimal("200") and len(records) == 2
    print("\n=== Test ran successfully ===\n")
