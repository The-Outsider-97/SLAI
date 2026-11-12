"""Canonical financial value objects and request/result contracts for SLAI v2.4.

Contracts, not financial algorithms or payment processing. All monetary values
use Decimal and an explicit, validated currency. Cash flows can be signed;
SLAI's separate ``src.functions.payment.Money`` deliberately accepts only
positive payment amounts. Cross-currency operations require an explicit FX
workflow; no exchange rate is silently inferred.

Financial input data is immutable by construction. Time stamps must include an
explicit UTC offset. Currency codes are verified against the subsystem's
versioned ISO-4217 reference snapshot (see financial_helpers). They are not a
promise of historical legal tender or up-to-date regulatory status.

No live market data, trained models, external calls, or side effects are
performed here. Non-sensitive diagnostic logging uses SLAI's canonical logger.
"""
from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, localcontext, getcontext
from enum import Enum
from typing import Any

from .utils.financial_errors import *
from .utils.financial_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Financial Types")
printer = PrettyPrinter()


def _coerce_enum(value: Any, enum_type: type[Enum], name: str) -> Enum:
    """Strictly coerce domain enum string values, never implicit numbers."""
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise FinancialValidationError(
            f"{name} must be a supported string value",
            context={"field": name, "type": type(value).__name__},
        )
    try:
        return enum_type(value.strip().lower())
    except ValueError as exc:
        raise UnsupportedConventionError(
            f"Unsupported {name}", context={"field": name, "value": value[:80]}
        ) from exc


class DayCountConvention(str, Enum):
    """Declared conventions only; day-count algorithms belong in calculation."""

    ACT_365F = "act/365f"
    ACT_360 = "act/360"
    ACT_ACT_ISDA = "act/act-isda"
    THIRTY_360_US = "30/360-us"
    THIRTYE_360 = "30e/360"


class RateConvention(str, Enum):
    """How an annualized rate is stated (not how it is computed)."""

    NOMINAL = "nominal"
    EFFECTIVE = "effective"
    CONTINUOUS = "continuous"


class InstrumentType(str, Enum):
    EQUITY = "equity"
    BOND = "bond"
    FUND = "fund"
    ETF = "etf"
    CASH = "cash"
    LOAN = "loan"
    OPTION = "option"
    FUTURE = "future"
    INDEX = "index"
    OTHER = "other"


class FinancialResultStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    INSUFFICIENT_DATA = "insufficient_data"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class Currency:
    """ISO-4217 currency designator and optional authoritative minor units.

    A code is validated against a bounded, documented ISO reference snapshot.
    ``minor_units`` is NEVER guessed from the code; pass it only when sourced
    from a trusted, versioned catalogue. Special non-settlement codes XXX/XTS
    are rejected by the default registry policy.
    """

    code: str
    minor_units: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", normalize_currency_code(self.code))
        if self.minor_units is not None:
            require_positive_int(self.minor_units, "minor_units", minimum=0, maximum=9)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Currency":
        if not isinstance(data, Mapping):
            raise FinancialValidationError("Currency must be a mapping")
        if "code" not in data:
            raise FinancialValidationError("Currency requires code")
        return cls(code=data["code"], minor_units=data.get("minor_units"))

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "minor_units": self.minor_units}

    def __str__(self) -> str:
        return self.code



def _require_fields(data: Any, *keys: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise FinancialValidationError("Financial type payload must be a mapping")
    missing = [key for key in keys if key not in data]
    if missing:
        raise FinancialValidationError(
            "Financial type payload is missing required fields",
            context={"missing_fields": missing},
        )
    return data


def _currency_from_payload(value: Any) -> Currency | str:
    return Currency.from_dict(value) if isinstance(value, Mapping) else value


def _as_currency(value: Currency | str, field_name: str = "currency") -> Currency:
    if isinstance(value, Currency):
        return value
    if not isinstance(value, str):
        raise FinancialValidationError(f"{field_name} must be Currency or currency code", context={"field": field_name})
    return Currency(value)


def _exact_amount_sum(left: Decimal, right: Decimal) -> Decimal:
    """Add Decimal amounts without implicit default-context precision loss.

    The decimal context is enlarged for this bounded operation only; global
    context and financial rounding policies are not modified.
    """
    left_exponent = left.as_tuple().exponent
    right_exponent = right.as_tuple().exponent
    if not isinstance(left_exponent, int) or not isinstance(right_exponent, int):
        raise FinancialValidationError("Financial amounts must be finite")
    exponent = min(left_exponent, right_exponent)
    integer_digits = max(left.adjusted() + 1, right.adjusted() + 1, 1)
    required = max(1, integer_digits) + max(0, -exponent) + 2
    with localcontext() as ctx:
        ctx.prec = max(getcontext().prec, required)
        return left + right


@dataclass(frozen=True, slots=True)
class FinancialAmount:
    """Exact, signed, currency-qualified amount; not a payment authorization.

    Values such as negative expenses, credits, losses, and cash outflows are
    permitted. Arithmetic requires identical currency codes. Rounding and FX
    conversion belong to explicit higher-level policies, not this type.
    """

    amount: Decimal
    currency: Currency | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount", to_decimal(self.amount, "amount"))
        object.__setattr__(self, "currency", _as_currency(self.currency))

    @classmethod
    def of(cls, amount: Decimal | int | str, currency: Currency | str) -> "FinancialAmount":
        return cls(amount=to_decimal(amount, "amount"), currency=currency)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FinancialAmount":
        if not isinstance(data, Mapping) or "amount" not in data or "currency" not in data:
            raise FinancialValidationError("FinancialAmount requires amount and currency")
        currency = data["currency"]
        if isinstance(currency, Mapping):
            currency = Currency.from_dict(currency)
        return cls(data["amount"], currency)

    def __add__(self, other: Any) -> "FinancialAmount":
        if not isinstance(other, FinancialAmount):
            return NotImplemented
        require_same_currency(self.currency, other.currency, operation="addition")
        return FinancialAmount(_exact_amount_sum(self.amount, other.amount), self.currency)

    def __sub__(self, other: Any) -> "FinancialAmount":
        if not isinstance(other, FinancialAmount):
            return NotImplemented
        require_same_currency(self.currency, other.currency, operation="subtraction")
        return FinancialAmount(_exact_amount_sum(self.amount, -other.amount), self.currency)

    def __neg__(self) -> "FinancialAmount":
        return FinancialAmount(-self.amount, self.currency)

    def __abs__(self) -> "FinancialAmount":
        return FinancialAmount(abs(self.amount), self.currency)

    def to_dict(self) -> dict[str, Any]:
        return {"amount": format(self.amount, "f"), "currency": _as_currency(self.currency).to_dict()}


@dataclass(frozen=True, slots=True)
class InterestRate:
    """Signed annualized rate with explicit quotation/compounding convention.

    For nominal rates the integer ``periods_per_year`` must be supplied.
    Effective annual or continuously compounded annual rates do not accept
    an additional compounding frequency. Negative rates are not categorically
    rejected: validity depends on the consuming financial formula.
    """

    value: Decimal
    convention: RateConvention | str
    periods_per_year: int | None = None
    day_count: DayCountConvention | str = DayCountConvention.ACT_365F

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", to_decimal(self.value, "rate"))
        c = _coerce_enum(self.convention, RateConvention, "rate convention")
        dc = _coerce_enum(self.day_count, DayCountConvention, "day-count convention")
        object.__setattr__(self, "convention", c)
        object.__setattr__(self, "day_count", dc)
        if c == RateConvention.NOMINAL:
            require_positive_int(self.periods_per_year, "periods_per_year", maximum=366)
        elif self.periods_per_year is not None:
            raise UnsupportedConventionError(
                "Compounding frequency is only applicable to nominal rates",
                context={"convention": c.value},
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InterestRate":
        data = _require_fields(data, "value", "convention")
        return cls(data["value"], data["convention"], data.get("periods_per_year"),
                   data.get("day_count", DayCountConvention.ACT_365F))

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": format(self.value, "f"),
            "convention": (
                self.convention.value
                if isinstance(self.convention, RateConvention)
                else self.convention
            ),
            "periods_per_year": self.periods_per_year,
            "day_count": (
                self.day_count.value
                if isinstance(self.day_count, DayCountConvention)
                else self.day_count
            ),
        }


@dataclass(frozen=True, slots=True)
class FinancialPeriod:
    """Half-open [start, end) financial interval; requires start < end."""

    start: date
    end: date

    def __post_init__(self) -> None:
        start, end = to_date(self.start, "start"), to_date(self.end, "end")
        if start >= end:
            raise InvalidFinancialPeriodError("Financial period requires start < end")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def contains(self, day: date | str) -> bool:
        candidate = to_date(day, "day")
        return self.start <= candidate < self.end

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FinancialPeriod":
        data = _require_fields(data, "start", "end")
        return cls(data["start"], data["end"])

    def to_dict(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


@dataclass(frozen=True, slots=True)
class CashFlow:
    """One explicitly dated signed cash movement.

    Positive amounts mean cash received; negative amounts mean cash paid.
    Accounting interpretation and discounting belong to specialized modules.
    """

    on_date: date
    amount: FinancialAmount
    reference: str | None = None
    category: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "on_date", to_date(self.on_date, "on_date"))
        if not isinstance(self.amount, FinancialAmount):
            raise FinancialValidationError("CashFlow.amount must be FinancialAmount")
        object.__setattr__(self, "reference", _optional_text(self.reference, "reference", 128))
        object.__setattr__(self, "category", _optional_text(self.category, "category", 128))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata, "metadata"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CashFlow":
        data = _require_fields(data, "on_date", "amount")
        amount = data["amount"]
        if isinstance(amount, Mapping):
            amount = FinancialAmount.from_dict(amount)
        return cls(data["on_date"], amount, data.get("reference"),
                   data.get("category"), data.get("metadata", {}))

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class FinancialInstrument:
    """Instrument identity and classification only (no guessed market data).

    ``instrument_id`` should be the authoritative identifier used by the
    caller/data source. Symbols need not be globally unique; the venue and
    identifiers may be supplied to disambiguate instruments.
    """

    instrument_id: str
    instrument_type: InstrumentType | str
    currency: Currency | str
    symbol: str | None = None
    name: str | None = None
    venue: str | None = None
    identifiers: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", normalize_identifier(self.instrument_id, "instrument_id"))
        object.__setattr__(self, "instrument_type", _coerce_enum(self.instrument_type, InstrumentType, "instrument type"))
        object.__setattr__(self, "currency", _as_currency(self.currency))
        for field_name in ("symbol", "name", "venue"):
            object.__setattr__(self, field_name, _optional_text(getattr(self, field_name), field_name, 256))
        object.__setattr__(self, "identifiers", freeze_mapping(self.identifiers, "identifiers"))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata, "metadata"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FinancialInstrument":
        data = _require_fields(data, "instrument_id", "instrument_type", "currency")
        return cls(data["instrument_id"], data["instrument_type"],
                   _currency_from_payload(data["currency"]),
                   symbol=data.get("symbol"), name=data.get("name"),
                   venue=data.get("venue"), identifiers=data.get("identifiers", {}),
                   metadata=data.get("metadata", {}))

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class FinancialDataSource:
    """Provenance of an externally obtained financial observation.

    This describes evidence only; it does not imply a network query was made.
    ``retrieved_at`` is the instant data was observed by the ingest process.
    """

    source_id: str
    retrieved_at: datetime
    reference: str | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", normalize_identifier(self.source_id, "source_id"))
        object.__setattr__(self, "retrieved_at", to_utc_datetime(self.retrieved_at, "retrieved_at"))
        object.__setattr__(self, "reference", _optional_text(self.reference, "reference", 1024))
        object.__setattr__(self, "revision", _optional_text(self.revision, "revision", 128))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FinancialDataSource":
        data = _require_fields(data, "source_id", "retrieved_at")
        return cls(data["source_id"], data["retrieved_at"],
                   reference=data.get("reference"), revision=data.get("revision"))

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class MarketObservation:
    """Sourced point observation, not inherently a tradeable current quote.

    ``value`` is unit-specific and can be negative (e.g. interest rates).
    FinancialAmount is not forced onto quantities such as volume and yield.
    ``assert_fresh`` is opt-in and must use an explicit caller-defined horizon.
    """

    instrument_id: str
    field_name: str
    value: Decimal
    observed_at: datetime
    source: FinancialDataSource
    currency: Currency | str | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", normalize_identifier(self.instrument_id, "instrument_id"))
        object.__setattr__(self, "field_name", normalize_identifier(self.field_name, "field_name"))
        object.__setattr__(self, "value", to_decimal(self.value, "value"))
        object.__setattr__(self, "observed_at", to_utc_datetime(self.observed_at, "observed_at"))
        if not isinstance(self.source, FinancialDataSource):
            raise InvalidMarketDataError("MarketObservation.source must be FinancialDataSource")
        if self.observed_at > self.source.retrieved_at:
            raise InvalidMarketDataError("Observation time must not exceed source retrieval time")
        if self.currency is not None:
            object.__setattr__(self, "currency", _as_currency(self.currency))
        object.__setattr__(self, "unit", _optional_text(self.unit, "unit", 64))
        if self.currency is None and self.unit is None:
            raise InvalidMarketDataError("Market observation requires a currency or unit")

    def assert_fresh(self, *, as_of: datetime, max_age: timedelta) -> None:
        now = to_utc_datetime(as_of, "as_of")
        if not isinstance(max_age, timedelta) or max_age <= timedelta(0):
            raise FinancialValidationError("max_age must be a positive timedelta")
        if now < self.observed_at:
            raise InvalidMarketDataError("Observation timestamp is later than evaluation time")
        if now - self.observed_at > max_age:
            logger.warning("Financial market observation rejected as stale (field=%s)", self.field_name)
            raise StaleMarketDataError(
                "Market observation exceeds permitted age",
                context={"field": self.field_name, "max_age_seconds": max_age.total_seconds()},
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MarketObservation":
        data = _require_fields(data, "instrument_id", "field_name", "value", "observed_at", "source")
        source = data["source"]
        if isinstance(source, Mapping):
            source = FinancialDataSource.from_dict(source)
        currency = data.get("currency")
        return cls(data["instrument_id"], data["field_name"], data["value"],
                   data["observed_at"], source,
                   currency=None if currency is None else _currency_from_payload(currency),
                   unit=data.get("unit"))

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class ExchangeRate:
    """Sourced quote: one unit of base currency equals `rate` quote units."""

    base_currency: Currency | str
    quote_currency: Currency | str
    rate: Decimal
    observed_at: datetime
    source: FinancialDataSource

    def __post_init__(self) -> None:
        base, quote = _as_currency(self.base_currency, "base_currency"), _as_currency(self.quote_currency, "quote_currency")
        if base.code == quote.code:
            raise CurrencyMismatchError("ExchangeRate requires two different currencies")
        object.__setattr__(self, "base_currency", base)
        object.__setattr__(self, "quote_currency", quote)
        rate = to_decimal(self.rate, "rate")
        if rate <= 0:
            raise FinancialDomainError("Exchange rate must be positive")
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "observed_at", to_utc_datetime(self.observed_at, "observed_at"))
        if not isinstance(self.source, FinancialDataSource):
            raise FinancialValidationError("ExchangeRate.source must be FinancialDataSource")
        if self.observed_at > self.source.retrieved_at:
            raise InvalidMarketDataError("FX rate observation cannot be later than retrieval")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExchangeRate":
        data = _require_fields(data, "base_currency", "quote_currency", "rate", "observed_at", "source")
        source = data["source"]
        if isinstance(source, Mapping):
            source = FinancialDataSource.from_dict(source)
        return cls(_currency_from_payload(data["base_currency"]),
                   _currency_from_payload(data["quote_currency"]),
                   data["rate"], data["observed_at"], source)

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class FinancialRequest:
    """Immutable, validated and schema-stable financial operation request.

    This type records operation intent; it does NOT confer permissions. External
    state changes require separate authorization in SLAI's execution layer.
    """

    request_id: str
    domain: str
    operation: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", normalize_identifier(self.request_id, "request_id"))
        object.__setattr__(self, "domain", normalize_identifier(self.domain, "domain"))
        object.__setattr__(self, "operation", normalize_identifier(self.operation, "operation"))
        object.__setattr__(self, "created_at", to_utc_datetime(self.created_at, "created_at"))
        object.__setattr__(self, "parameters", freeze_mapping(self.parameters, "parameters"))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata, "metadata"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FinancialRequest":
        if not isinstance(data, Mapping):
            raise FinancialValidationError("FinancialRequest must be a mapping")
        for key in ("request_id", "domain", "operation"):
            if key not in data:
                raise FinancialValidationError(f"FinancialRequest requires {key}")
        return cls(
            request_id=data["request_id"], domain=data["domain"], operation=data["operation"],
            parameters=data.get("parameters", {}),
            created_at=data["created_at"] if "created_at" in data else _utc_now(),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class FinancialResult:
    """Immutable auditable result; never upgrades model predictions to facts.

    The ``result`` value must be JSON-compatible and is deeply frozen if a
    mapping/sequence. Structured error codes may be placed under ``diagnostics``;
    secrets must never be put into result metadata or reporting fields.
    """

    request_id: str
    domain: str
    operation: str
    status: FinancialResultStatus | str
    result: Any = None
    completed_at: datetime = field(default_factory=_utc_now)
    assumptions: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    sources: tuple[FinancialDataSource, ...] = ()
    warnings: tuple[str, ...] = ()
    methodology: str | None = None
    model_version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in ("request_id", "domain", "operation"):
            object.__setattr__(self, key, normalize_identifier(getattr(self, key), key))
        status = _coerce_enum(self.status, FinancialResultStatus, "result status")
        object.__setattr__(self, "status", status)
        if status == FinancialResultStatus.SUCCESS and self.result is None:
            raise FinancialValidationError("Successful FinancialResult requires a result")
        object.__setattr__(self, "completed_at", to_utc_datetime(self.completed_at, "completed_at"))
        if isinstance(self.result, Mapping):
            object.__setattr__(self, "result", freeze_mapping(self.result, "result"))
        elif isinstance(self.result, (tuple, list)):
            # Defensively freeze mixed nested mappings as well.
            object.__setattr__(self, "result", freeze_mapping({"value": self.result}, "result")["value"])
        else:
            to_json_compatible(self.result)
        for key in ("assumptions", "diagnostics", "metadata"):
            object.__setattr__(self, key, freeze_mapping(getattr(self, key), key))
        if not isinstance(self.sources, (list, tuple)) or any(not isinstance(s, FinancialDataSource) for s in self.sources):
            raise FinancialValidationError("sources must contain FinancialDataSource values")
        object.__setattr__(self, "sources", tuple(self.sources))
        if not isinstance(self.warnings, (list, tuple)):
            raise FinancialValidationError("warnings must be a list/tuple of strings")
        object.__setattr__(self, "warnings", tuple(require_text(w, "warning", max_length=1024) for w in self.warnings))
        object.__setattr__(self, "methodology", _optional_text(self.methodology, "methodology", 1024))
        object.__setattr__(self, "model_version", _optional_text(self.model_version, "model_version", 128))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FinancialResult":
        data = _require_fields(data, "request_id", "domain", "operation", "status")
        sources = data.get("sources", ())
        if not isinstance(sources, (list, tuple)):
            raise FinancialValidationError("sources must be a sequence")
        parsed_sources = tuple(
            FinancialDataSource.from_dict(s) if isinstance(s, Mapping) else s
            for s in sources
        )
        return cls(
            request_id=data["request_id"], domain=data["domain"],
            operation=data["operation"], status=data["status"],
            result=data.get("result"),
            completed_at=data["completed_at"] if "completed_at" in data else _utc_now(),
            assumptions=data.get("assumptions", {}),
            diagnostics=data.get("diagnostics", {}),
            sources=parsed_sources,
            warnings=data.get("warnings", ()),
            methodology=data.get("methodology"),
            model_version=data.get("model_version"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


__all__ = [
    "DayCountConvention", "RateConvention", "InstrumentType", "FinancialResultStatus",
    "Currency", "FinancialAmount", "InterestRate", "FinancialPeriod", "CashFlow",
    "FinancialInstrument", "FinancialDataSource", "MarketObservation", "ExchangeRate",
    "FinancialRequest", "FinancialResult",
]
