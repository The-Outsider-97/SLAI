"""ISO 4217 minor-unit reference data for the SLAI Financial subsystem.

``Currency.minor_units`` is deliberately never guessed from a code (see
``financial_types``). This module is the single, versioned source from which a
caller may obtain authoritative minor units, for example to quantize a charge
or a rebalance amount to a currency's smallest settlement unit.

The table was checked against the SIX/ISO 4217 "List One" (current currency and
funds codes) as mirrored by Wikipedia (https://en.wikipedia.org/wiki/ISO_4217,
checked 2026-11-12). Every code in ``ISO_4217_CODE_SNAPSHOT`` has an entry; codes
without a decimal relationship (precious metals, SDR, test/no-currency codes)
carry ``minor_units=None`` and refuse quantization. Withdrawn codes remain
resolvable because historical datasets still reference them, but are flagged
``active=False``. It is a reference snapshot, not a date-aware legal-tender
registry: refresh it when SIX publishes an amendment.
"""
from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..financial_types import Currency
from ..utils.financial_errors import *
from ..utils.financial_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Financial Currency Catalog")
printer = PrettyPrinter()

CATALOG_VERSION = "ISO-4217 list-one, checked 2026-11-12"

# Codes whose minor-unit exponent differs from 2 (SIX list one).
_MINOR_UNIT_EXCEPTIONS: Mapping[str, int] = MappingProxyType({
    "BHD": 3, "IQD": 3, "JOD": 3, "KWD": 3, "LYD": 3, "OMR": 3, "TND": 3,
    "BIF": 0, "CLP": 0, "DJF": 0, "GNF": 0, "ISK": 0, "JPY": 0, "KMF": 0,
    "KRW": 0, "PYG": 0, "RWF": 0, "UGX": 0, "UYI": 0, "VND": 0, "VUV": 0,
    "XAF": 0, "XOF": 0, "XPF": 0,
    "CLF": 4, "UYW": 4,
})

# No decimal relationship: metals, SDR, bond-market units, test and no-currency codes.
_NO_MINOR_UNIT = frozenset({
    "XAG", "XAU", "XBA", "XBB", "XBC", "XBD", "XDR", "XPD", "XPT",
    "XSU", "XTS", "XUA", "XXX",
})

# Codes withdrawn from list one but still present in historical datasets.
_WITHDRAWN = frozenset({"ANG", "BGN", "CUC", "HRK", "SLL", "ZWL"})


@dataclass(frozen=True, slots=True)
class CurrencyRecord:
    """Reference record for one currency designator."""

    code: str
    minor_units: int | None
    active: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", normalize_currency_code(
            self.code, allow_non_transaction_codes=True))
        if self.minor_units is not None:
            require_positive_int(self.minor_units, "minor_units", minimum=0, maximum=9)
        if type(self.active) is not bool:
            raise FinancialValidationError("active must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "minor_units": self.minor_units, "active": self.active}


def _default_records() -> dict[str, CurrencyRecord]:
    records: dict[str, CurrencyRecord] = {}
    for code in sorted(ISO_4217_CODE_SNAPSHOT):
        if code in _NO_MINOR_UNIT:
            minor: int | None = None
        else:
            minor = _MINOR_UNIT_EXCEPTIONS.get(code, 2)
        records[code] = CurrencyRecord(code, minor, code not in _WITHDRAWN)
    return records


class CurrencyCatalog:
    """Immutable lookup of ISO 4217 minor units.

    A custom ``records`` mapping may replace the bundled snapshot with a newer
    authoritative catalogue. The catalogue only answers questions about a
    currency; it never converts, rounds or infers an exchange rate.
    """

    def __init__(self, records: Mapping[str, CurrencyRecord] | None = None):
        source = _default_records() if records is None else dict(records)
        if not source:
            raise FinancialConfigurationError("Currency catalog requires at least one record")
        checked: dict[str, CurrencyRecord] = {}
        for code, record in source.items():
            if not isinstance(record, CurrencyRecord):
                raise FinancialValidationError("Catalog records must be CurrencyRecord values")
            if normalize_currency_code(code, allow_non_transaction_codes=True) != record.code:
                raise FinancialIntegrityError("Catalog key does not match its record code")
            checked[record.code] = record
        self._records: Mapping[str, CurrencyRecord] = MappingProxyType(checked)

    def record(self, code: Any) -> CurrencyRecord:
        key = normalize_currency_code(code, allow_non_transaction_codes=True)
        try:
            return self._records[key]
        except KeyError as exc:
            raise InvalidCurrencyError("Currency is not in the catalog", context={"currency": key}) from exc

    def codes(self) -> frozenset[str]:
        return frozenset(self._records)

    def has_minor_units(self, code: Any) -> bool:
        return self.record(code).minor_units is not None

    def minor_units(self, code: Any) -> int:
        """Exponent such that one major unit equals 10**exponent minor units."""
        record = self.record(code)
        if record.minor_units is None:
            raise FinancialDomainError(
                "Currency has no decimal minor unit and cannot be quantized",
                context={"currency": record.code})
        return record.minor_units

    def currency(self, code: Any) -> Currency:
        """Validated ``Currency`` carrying authoritative minor units."""
        record = self.record(code)
        return Currency(record.code, record.minor_units)

    def resolve_minor_units(self, currency: Currency | str) -> int:
        """Minor units of a Currency, preferring an explicitly supplied value.

        An explicit ``Currency.minor_units`` always wins; a conflict with the
        catalogue is reported rather than silently overridden.
        """
        if isinstance(currency, Currency):
            reference = self.record(currency.code).minor_units
            if currency.minor_units is None:
                return self.minor_units(currency.code)
            if reference is not None and reference != currency.minor_units:
                raise FinancialIntegrityError(
                    "Explicit minor units conflict with the catalog",
                    context={"currency": currency.code})
            return currency.minor_units
        return self.minor_units(currency)

    def __contains__(self, code: object) -> bool:
        try:
            self.record(code)
        except FinancialError:
            return False
        return True

    def __len__(self) -> int:
        return len(self._records)


DEFAULT_CATALOG = CurrencyCatalog()


def minor_units(code: Any) -> int:
    """Convenience lookup in the bundled catalogue."""
    return DEFAULT_CATALOG.minor_units(code)


__all__ = ["CATALOG_VERSION", "CurrencyRecord", "CurrencyCatalog", "DEFAULT_CATALOG", "minor_units"]

if __name__ == "__main__":
    print("\n=== Running Financial Currency Catalog ===\n")
    printer.status("TEST", "Financial Currency Catalog initialized", "info")
    catalog = CurrencyCatalog()
    assert catalog.codes() >= {"EUR", "USD", "JPY", "KWD", "CLF", "ZWG", "XCG"}
    assert len(catalog) == len(ISO_4217_CODE_SNAPSHOT)
    assert [catalog.minor_units(c) for c in ("EUR", "JPY", "KWD", "CLF")] == [2, 0, 3, 4]
    assert not catalog.has_minor_units("XAU") and not catalog.record("HRK").active
    assert catalog.currency("usd") == Currency("USD", 2)
    assert catalog.resolve_minor_units(Currency("EUR")) == 2
    try:
        catalog.minor_units("XAU")
    except FinancialDomainError:
        pass
    else:
        raise AssertionError("metal codes must refuse quantization")
    try:
        catalog.resolve_minor_units(Currency("JPY", 2))
    except FinancialIntegrityError:
        pass
    else:
        raise AssertionError("conflicting explicit minor units must be reported")
    assert "ABC" not in catalog and "eur" in catalog
    print("\n=== Test ran successfully ===\n")
