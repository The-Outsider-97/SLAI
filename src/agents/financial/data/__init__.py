"""Reusable financial data-layer modules for the SLAI Financial subsystem.

Consumed by ``financial_data`` (storage), ``financial_economics`` (series and
indices), ``financial_portfolio`` (valuation currency and return histories) and
``financial_subscription`` (conversion of projected flows).
"""

from __future__ import annotations

__version__ = "2.4.0"

from .currency_catalog import *
from .fx import *
from .observation_store import *
from .series import *


from .currency_catalog import __all__ as _currency_catalog_exports
from .fx import __all__ as _fx_exports
from .observation_store import __all__ as _observation_store_exports
from .series import __all__ as _series_exports


__all__ = [
    *_currency_catalog_exports,
    *_fx_exports,
    *_observation_store_exports,
    *_series_exports,
] # type: ignore
