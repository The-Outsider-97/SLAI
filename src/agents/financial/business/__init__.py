"""Reusable financial business-rule modules for the SLAI Financial subsystem.

Consumed by ``financial_subscription`` (itemized charges, proration, recurring
revenue metrics) and ``financial_portfolio`` (rounding, cost estimation,
performance measurement and return moments).
"""

from __future__ import annotations

__version__ = "2.4.0"

from .charges import *
from .performance import *
from .proration import *
from .recurring_metrics import *
from .rounding import *


from .charges import __all__ as _charges_exports
from .performance import __all__ as _performance_exports
from .proration import __all__ as _proration_exports
from .recurring_metrics import __all__ as _recurring_metrics_exports
from .rounding import __all__ as _rounding_exports


__all__ = [
    *_charges_exports,
    *_performance_exports,
    *_proration_exports,
    *_recurring_metrics_exports,
    *_rounding_exports,
] # type: ignore
