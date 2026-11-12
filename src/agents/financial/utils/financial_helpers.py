"""SLAI Financial subsystem: validated, deterministic domain helpers.

Pure normalization and serialization helpers only. Financial formulas, payment
orchestration, account access, generic statistics, HTTP, and model inference
must remain in their respective owners. Importing this module has no I/O or
model-loading side effects.

Currency catalog: bundled ISO 4217 *code registry snapshot* (not a promise that
an instrument or currency was valid on a specific historical date). Refresh the
catalog as standards evolve; callers may provide a versioned, authoritative
allowed-code set. Currency minor units are intentionally NOT inferred here.
"""

from __future__ import annotations

__version__ = "2.4.0"

import hashlib
import json
import math
import re

from collections.abc import Collection, Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeVar
from uuid import UUID

from logs.logger import get_logger # pyright: ignore[reportMissingImports]
from .financial_errors import (
    FinancialSerializationError,
    FinancialValidationError,
    InvalidCurrencyError,
)

logger = get_logger("Financial Helpers")

# ISO 4217 currency-designator reference set; includes special X-codes and
# legacy codes used in financial historical datasets. It is NOT a date-aware
# legal-tender registry. Unknown future codes must be explicitly admitted by
# caller-supplied updated registries, not inferred from syntax alone.
ISO_4217_CODE_SNAPSHOT = frozenset("""
AED AFN ALL AMD ANG AOA ARS AUD AWG AZN BAM BBD BDT BGN BHD BIF BMD
BND BOB BOV BRL BSD BTN BWP BYN BZD CAD CDF CHE CHF CHW CLF CLP CNY
COP COU CRC CUC CUP CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD FKP
GBP GEL GHS GIP GMD GNF GTQ GYD HKD HNL HRK HTG HUF IDR ILS INR IQD
IRR ISK JMD JOD JPY KES KGS KHR KMF KPW KRW KWD KYD KZT LAK LBP LKR
LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP MRU MUR MVR MWK MXN MXV MYR
MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN PGK PHP PKR PLN PYG QAR RON
RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP SLE SLL SOS SRD SSP STN SVC
SYP SZL THB TJS TMT TND TOP TRY TTD TWD TZS UAH UGX USD USN UYI UYU
UYW UZS VED VES VND VUV WST XAF XAG XAU XBA XBB XBC XBD XCD XCG XDR XOF
XPD XPF XPT XSU XTS XUA XXX YER ZAR ZMW ZWG ZWL
""".split())

_NON_TRANSACTION_CODES = frozenset({"XXX", "XTS"})
_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$", re.ASCII)
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$", re.ASCII)
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$", re.ASCII)
_MAX_DEPTH = 14
_MAX_ITEMS = 4096
_MAX_TEXT_LENGTH = 16384
_T = TypeVar("_T")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _optional_text(value: Any, name: str, limit: int = 256) -> str | None:
    return None if value is None else require_text(value, name, max_length=limit)


def require_text(value: Any, name: str, *, max_length: int = 256) -> str:
    """Return stripped nonblank text, rejecting implicit object coercion."""
    if not isinstance(name, str) or not name:
        raise ValueError("name must be non-empty text")
    if not isinstance(value, str):
        raise FinancialValidationError(f"{name} must be text", context={"field": name})
    result = value.strip()
    if not result or len(result) > max_length or "\x00" in result:
        raise FinancialValidationError(
            f"{name} must contain 1 to {max_length} non-NUL characters",
            context={"field": name},
        )
    return result


def normalize_identifier(value: Any, name: str = "identifier", *, max_length: int = 128) -> str:
    """Normalize nonsecret public identifiers (not personal account numbers)."""
    result = require_text(value, name, max_length=max_length)
    if not _IDENTIFIER_PATTERN.fullmatch(result):
        raise FinancialValidationError(
            f"{name} contains unsupported identifier characters",
            context={"field": name},
        )
    return result


def normalize_currency_code(
    value: Any,
    *,
    allowed_codes: Collection[str] | None = None,
    allow_non_transaction_codes: bool = False,
) -> str:
    """Validate currency designator against a known ISO-style registry.

    ``allowed_codes`` is an authoritative override for legitimate newer codes
    or an application-specific supported-currency subset.  No exchange rates
    or fraction digits are guessed from currency code.  XXX and XTS are not
    valid settlement currencies by default.
    """
    if not isinstance(value, str):
        raise InvalidCurrencyError("Currency code must be a string")
    code = value.strip().upper()
    if not _CURRENCY_PATTERN.fullmatch(code):
        raise InvalidCurrencyError("Currency code must contain exactly three ASCII letters")
    registry = ISO_4217_CODE_SNAPSHOT if allowed_codes is None else allowed_codes
    if isinstance(registry, (str, bytes)) or not isinstance(registry, Collection):
        raise FinancialValidationError("allowed_codes must be a collection of currency codes")
    if code not in registry:
        raise InvalidCurrencyError("Unrecognized or disallowed currency code", context={"currency": code})
    if not allow_non_transaction_codes and code in _NON_TRANSACTION_CODES:
        raise InvalidCurrencyError("Non-transaction currency code is not permitted", context={"currency": code})
    return code


def to_decimal(value: Any, name: str = "value", *, allow_float: bool = False) -> Decimal:
    """Parse a finite Decimal without implicit binary-floating-point rounding.

    Accepts Decimal, int (except bool), or plain base-10 string. For floats,
    callers must explicitly opt in to ``Decimal(str(float))`` conversion.
    Rejects NaN, infinities and arbitrary object coercion.
    """
    if isinstance(value, bool):
        raise FinancialValidationError(f"{name} must not be boolean", context={"field": name})
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, int):
        result = Decimal(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or len(text) > 1024:
            raise FinancialValidationError(f"{name} must be a valid finite decimal string", context={"field": name})
        try:
            result = Decimal(text)
        except InvalidOperation as exc:
            raise FinancialValidationError(f"{name} must be a valid decimal", context={"field": name}, cause=exc) from exc
    elif isinstance(value, float) and allow_float:
        if not math.isfinite(value):
            raise FinancialValidationError(f"{name} must be finite", context={"field": name})
        result = Decimal(str(value))
    else:
        raise FinancialValidationError(
            f"{name} must be Decimal, int or a decimal string",
            context={"field": name, "type": type(value).__name__},
        )
    if not result.is_finite():
        raise FinancialValidationError(f"{name} must be finite", context={"field": name})
    tuple_info = result.as_tuple()
    exponent = tuple_info.exponent
    if result.adjusted() > 60 or (isinstance(exponent, int) and exponent < -60) or len(tuple_info.digits) > 1024:
        raise FinancialValidationError(f"{name} exceeds supported decimal magnitude or precision", context={"field": name})
    return result


def to_date(value: Any, name: str = "date") -> date:
    """Parse strict YYYY-MM-DD ISO dates without accepting timestamps."""
    if isinstance(value, datetime):
        raise FinancialValidationError(f"{name} must be a date, not datetime", context={"field": name})
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not _DATE_PATTERN.fullmatch(value):
        raise FinancialValidationError(f"{name} must use YYYY-MM-DD", context={"field": name})
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise FinancialValidationError(f"{name} is not a valid calendar date", context={"field": name}, cause=exc) from exc


def to_utc_datetime(value: Any, name: str = "timestamp") -> datetime:
    """Parse/normalize aware timestamps to UTC; reject naive date-times."""
    if isinstance(value, str):
        text = value.strip()
        if not text or "T" not in text:
            raise FinancialValidationError(f"{name} must be an ISO-8601 timestamp", context={"field": name})
        try:
            candidate = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise FinancialValidationError(f"{name} is not a valid timestamp", context={"field": name}, cause=exc) from exc
    elif isinstance(value, datetime):
        candidate = value
    else:
        raise FinancialValidationError(f"{name} must be an aware datetime or ISO timestamp", context={"field": name})
    if candidate.tzinfo is None or candidate.utcoffset() is None:
        raise FinancialValidationError(f"{name} must include a timezone offset", context={"field": name})
    return candidate.astimezone(timezone.utc)


def iso_utc(value: Any, name: str = "timestamp") -> str:
    """Canonical UTC timestamp with Z suffix."""
    return to_utc_datetime(value, name).isoformat().replace("+00:00", "Z")


def require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise FinancialValidationError(f"{name} must be boolean", context={"field": name})
    return value


def require_positive_int(value: Any, name: str, *, minimum: int = 1, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise FinancialValidationError(
            f"{name} must be an integer in the permitted range",
            context={"field": name, "minimum": minimum, "maximum": maximum},
        )
    return value


def _safe_value(value: Any, *, depth: int, seen: set[int], canonical: bool) -> Any:
    """Internal strict JSON serializer; no repr/eval/pickle fallbacks."""
    if depth > _MAX_DEPTH:
        raise FinancialSerializationError("Financial structure exceeds maximum nesting depth")
    if value is None or type(value) in (str, bool, int):
        if isinstance(value, str) and len(value) > _MAX_TEXT_LENGTH:
            raise FinancialSerializationError("Financial text exceeds serialization limit")
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise FinancialSerializationError("Cannot serialize non-finite Decimal")
        text = format(value, "f")
        return {"__decimal__": text} if canonical else text
    if type(value) is float:
        if not math.isfinite(value):
            raise FinancialSerializationError("Cannot serialize non-finite float")
        return {"__float__": value.hex()} if canonical else value
    if isinstance(value, datetime):
        text = iso_utc(value)
        return {"__datetime__": text} if canonical else text
    if isinstance(value, date):
        text = value.isoformat()
        return {"__date__": text} if canonical else text
    if isinstance(value, UUID):
        return {"__uuid__": str(value)} if canonical else str(value)
    if isinstance(value, Enum):
        inner = _safe_value(value.value, depth=depth + 1, seen=seen, canonical=canonical)
        return {"__enum__": type(value).__qualname__, "value": inner} if canonical else inner

    oid = id(value)
    if oid in seen:
        raise FinancialSerializationError("Cyclic financial structures cannot be serialized")
    seen.add(oid)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            if len(fields(value)) > _MAX_ITEMS:
                raise FinancialSerializationError("Financial dataclass has too many fields")
            data = {f.name: _safe_value(getattr(value, f.name), depth=depth + 1, seen=seen, canonical=canonical) for f in fields(value)}
            return {"__type__": f"{type(value).__module__}.{type(value).__qualname__}", "fields": data} if canonical else data
        if isinstance(value, Mapping):
            if len(value) > _MAX_ITEMS:
                raise FinancialSerializationError("Financial mapping exceeds item limit")
            if any(not isinstance(key, str) for key in value):
                raise FinancialSerializationError("Financial mapping keys must be strings")
            return {key: _safe_value(item, depth=depth + 1, seen=seen, canonical=canonical) for key, item in sorted(value.items())}
        if isinstance(value, (tuple, list)):
            if len(value) > _MAX_ITEMS:
                raise FinancialSerializationError("Financial sequence exceeds item limit")
            data = [_safe_value(item, depth=depth + 1, seen=seen, canonical=canonical) for item in value]
            return {"__tuple__": data} if canonical and isinstance(value, tuple) else data
    finally:
        seen.discard(oid)
    raise FinancialSerializationError(
        "Unsupported financial serialization type",
        context={"type": type(value).__name__},
    )


def to_json_compatible(value: Any) -> Any:
    """Convert validated financial values to JSON primitives without guessing."""
    return _safe_value(value, depth=0, seen=set(), canonical=False)


def canonical_json(value: Any) -> str:
    """Type-preserving, stable JSON for fingerprints and cache identities."""
    normalized = _safe_value(value, depth=0, seen=set(), canonical=True)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def stable_fingerprint(value: Any) -> str:
    """SHA-256 digest with distinct representations for Decimal/date/types."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def freeze_mapping(value: Mapping[str, Any] | None, name: str = "mapping") -> Mapping[str, Any]:
    """Defensively deep-freeze JSON-compatible input mappings.

    Finance dataclass values are preserved as independently validated objects;
    mutable mappings/lists are recursively frozen. Rejects circular objects,
    unknown types, non-finite numbers, non-string keys, and excessive nesting.
    """
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise FinancialValidationError(f"{name} must be a mapping", context={"field": name})
    # Perform strict validation before freezing; this handles unsupported types.
    to_json_compatible(value)

    def freeze(x: Any, depth: int) -> Any:
        if depth > _MAX_DEPTH:
            raise FinancialSerializationError("Financial structure exceeds maximum nesting depth")
        if isinstance(x, Mapping):
            return MappingProxyType({key: freeze(v, depth + 1) for key, v in x.items()})
        if isinstance(x, (list, tuple)):
            return tuple(freeze(v, depth + 1) for v in x)
        return x

    return freeze(value, 0)


def require_same_currency(left: Any, right: Any, *, operation: str = "combine") -> None:
    """Require matching currency codes without performing implicit FX conversion."""
    from .financial_errors import CurrencyMismatchError
    def get_code(value: Any) -> Any:
        if hasattr(value, "currency"):
            return get_code(value.currency)
        return value.code if hasattr(value, "code") else value
    left_code = normalize_currency_code(get_code(left))
    right_code = normalize_currency_code(get_code(right))
    if left_code != right_code:
        raise CurrencyMismatchError(
            "Amounts use different currencies; explicit conversion required",
            context={"left_currency": left_code, "right_currency": right_code, "operation": operation},
        )


def _status_text(value: Any) -> str:
    """Normalize status-like values to a string without requiring Enum access."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    status_value = getattr(value, "value", value)
    if isinstance(status_value, str):
        return status_value
    return str(status_value)


__all__ = [
    "ISO_4217_CODE_SNAPSHOT",
    "_utc_now", "_optional_text",
    "require_text", "normalize_identifier", "normalize_currency_code",
    "to_decimal", "to_date", "to_utc_datetime", "iso_utc",
    "require_bool", "require_positive_int", "to_json_compatible",
    "canonical_json", "stable_fingerprint", "freeze_mapping", "require_same_currency", "_status_text",
]
