"""SLAI v2.4: DCF valuation, sensitivity, deterministic stress and tail risk.

Time-value arithmetic lives in FinancialCalculation. This module owns economic
interpretation of valuation assumptions and historical-risk conventions, not a
second solver, optimization engine, Monte Carlo engine, trading system, or
regulatory-capital certification.

References: Gordon (1959) dividend growth/perpetuity; Rockafellar & Uryasev
(2000) CVaR/Expected Shortfall. Historical risk uses a *declared* empirical
quantile and fractional tail average, not a predictive loss bound.
"""
from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from typing import Any

from .financial_calculation import FinancialCalculation
from .financial_types import CashFlow, FinancialAmount, InterestRate, RateConvention
from .utils.config_loader import get_config_section, load_global_config
from .utils.financial_errors import *
from .utils.financial_helpers import *
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Financial Valuation")
printer = PrettyPrinter()


@dataclass(frozen=True, slots=True)
class DCFValuation:
    valuation_date: date
    cashflow_present_value: FinancialAmount
    terminal_present_value: FinancialAmount | None
    total_value: FinancialAmount
    method: str
    discount_rate: InterestRate
    terminal_growth_rate: Decimal | None

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class ValuationScenario:
    """Deterministic scenario only: no assumed probability or distribution."""

    name: str
    discount_rate: InterestRate
    cashflow_multiplier: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_text(self.name, "name", max_length=128))
        if not isinstance(self.discount_rate, InterestRate):
            raise FinancialValidationError("Scenario discount_rate must be InterestRate")
        factor = to_decimal(self.cashflow_multiplier, "cashflow_multiplier")
        if factor < 0:
            raise FinancialDomainError("Scenario cashflow_multiplier must be nonnegative")
        object.__setattr__(self, "cashflow_multiplier", factor)


@dataclass(frozen=True, slots=True)
class ScenarioValuation:
    scenario: str
    value: DCFValuation

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class HistoricalRisk:
    currency: str
    confidence: Decimal
    sample_count: int
    value_at_risk: FinancialAmount
    expected_shortfall: FinancialAmount
    method: str = "historical_nearest_rank_fractional_tail"

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class DrawdownResult:
    maximum_drawdown: Decimal
    peak_index: int
    trough_index: int
    ending_growth_factor: Decimal

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


class FinancialValuation:
    """Financial valuation and empirical risk with explicit assumptions.

    No implicit FX conversion or implicit market-data sourcing. All valuation
    cash flows must be in the same currency and occur on/after valuation date.
    """

    def __init__(self):
        self.config = load_global_config()
        self.financial_valuation_config = get_config_section("financial_valuation", config=self.config)
        cfg = self.financial_valuation_config
        try:
            self.max_scenarios = require_positive_int(cfg["max_scenarios"], "max_scenarios", maximum=4096)
            self.min_risk_observations = require_positive_int(cfg["min_risk_observations"], "min_risk_observations", minimum=2, maximum=1000000)
            self.max_risk_observations = require_positive_int(cfg["max_risk_observations"], "max_risk_observations", maximum=1000000)
            self.default_confidence = to_decimal(cfg["default_confidence"], "default_confidence")
        except KeyError as exc:
            raise FinancialConfigurationError("Missing financial_valuation configuration key", context={"key": str(exc)}) from exc
        if self.min_risk_observations > self.max_risk_observations or not (0 < self.default_confidence < 1):
            raise FinancialConfigurationError("Invalid financial_valuation risk limits or confidence")
        self.calculation = FinancialCalculation()

    def gordon_terminal_value(
        self, next_annual_cashflow: FinancialAmount, *, discount_rate: InterestRate,
        annual_growth: Any, terminal_date: date | str, valuation_date: date | str,
    ) -> FinancialAmount:
        """PV of a growing perpetuity with first flow one year after terminal.

        ``next_annual_cashflow`` is *not* the terminal-year flow. r and g are
        effective annual rates; r > g > -1. This method does not model debt,
        reinvestment, finite business life, or terminal capital structure.
        """
        if not isinstance(next_annual_cashflow, FinancialAmount):
            raise FinancialValidationError("next_annual_cashflow must be FinancialAmount")
        if not isinstance(discount_rate, InterestRate) or discount_rate.convention is not RateConvention.EFFECTIVE:
            raise UnsupportedConventionError("Growing perpetuity requires effective annual discount rate")
        growth = to_decimal(annual_growth, "annual_growth")
        if growth <= -1 or discount_rate.value <= growth:
            raise FinancialDomainError("Perpetuity requires -100% < growth < discount rate")
        terminal, valuation = to_date(terminal_date), to_date(valuation_date)
        if terminal < valuation:
            raise FinancialDomainError("terminal_date must not precede valuation_date")
        with localcontext() as ctx:
            ctx.prec = self.calculation.precision
            terminal_value = FinancialAmount(next_annual_cashflow.amount / (discount_rate.value - growth), next_annual_cashflow.currency)
        return self.calculation.present_value(terminal_value, discount_rate, valuation, terminal)

    def discounted_cashflow(
        self, flows: Sequence[CashFlow], *, discount_rate: InterestRate,
        valuation_date: date | str, terminal_next_annual_cashflow: FinancialAmount | None = None,
        terminal_date: date | str | None = None, terminal_growth_rate: Any | None = None,
    ) -> DCFValuation:
        """DCF from dated signed cashflows; optional explicit Gordon terminal.

        For enterprise DCF, callers supply enterprise free cash flow; for equity
        DCF, supply equity cash flow. Mixing bases is an input-modeling error.
        Terminal flows must be excluded from the explicit cashflow sequence.
        """
        if not isinstance(flows, (list, tuple)) or not flows or not all(isinstance(f, CashFlow) for f in flows):
            raise FinancialValidationError("DCF flows must be a nonempty CashFlow sequence")
        if not isinstance(discount_rate, InterestRate):
            raise FinancialValidationError("discount_rate must be InterestRate")
        valuation = to_date(valuation_date, "valuation_date")
        if any(f.on_date < valuation for f in flows):
            raise FinancialDomainError("DCF contains flows preceding valuation date")
        pv = self.calculation.net_present_value(flows, discount_rate, valuation)
        terminal_set = (terminal_next_annual_cashflow is not None, terminal_date is not None, terminal_growth_rate is not None)
        if any(terminal_set) and not all(terminal_set):
            raise FinancialValidationError("Terminal value requires cashflow, date, and growth together")
        terminal_pv = None
        growth = None
        if all(terminal_set):
            assert terminal_next_annual_cashflow is not None and terminal_date is not None and terminal_growth_rate is not None
            require_same_currency(pv.currency, terminal_next_annual_cashflow.currency, operation="terminal valuation")
            terminal = to_date(terminal_date, "terminal_date")
            if terminal < max(f.on_date for f in flows):
                raise FinancialDomainError("Terminal date precedes the last explicit cash flow")
            growth = to_decimal(terminal_growth_rate, "terminal_growth_rate")
            terminal_pv = self.gordon_terminal_value(terminal_next_annual_cashflow, discount_rate=discount_rate,
                                                     annual_growth=growth, terminal_date=terminal,
                                                     valuation_date=valuation)
        total = pv if terminal_pv is None else pv + terminal_pv
        logger.debug("Financial DCF evaluated | flows=%d | terminal=%s", len(flows), terminal_pv is not None)
        return DCFValuation(valuation, pv, terminal_pv, total,
                            "dated_discounted_cashflow_gordon_terminal" if terminal_pv is not None else "dated_discounted_cashflow",
                            discount_rate, growth)

    def scenario_valuations(
        self, flows: Sequence[CashFlow], *, valuation_date: date | str,
        scenarios: Sequence[ValuationScenario],
    ) -> tuple[ScenarioValuation, ...]:
        if not isinstance(scenarios, (tuple, list)) or not scenarios or len(scenarios) > self.max_scenarios:
            raise FinancialValidationError("scenarios must be a nonempty bounded list/tuple")
        if any(not isinstance(s, ValuationScenario) for s in scenarios):
            raise FinancialValidationError("scenarios must contain ValuationScenario values")
        names = [s.name for s in scenarios]
        if len(set(names)) != len(names):
            raise FinancialValidationError("Scenario names must be unique")
        if not isinstance(flows, (list, tuple)) or not all(isinstance(x, CashFlow) for x in flows):
            raise FinancialValidationError("flows must contain CashFlow values")
        results: list[ScenarioValuation] = []
        with localcontext() as ctx:
            ctx.prec = self.calculation.precision
            for scenario in scenarios:
                scaled = tuple(CashFlow(f.on_date, FinancialAmount(f.amount.amount * scenario.cashflow_multiplier, f.amount.currency),
                                        reference=f.reference, category=f.category, metadata=f.metadata) for f in flows)
                results.append(ScenarioValuation(scenario.name, self.discounted_cashflow(
                    scaled, discount_rate=scenario.discount_rate, valuation_date=valuation_date)))
        return tuple(results)

    def historical_var_es(
        self, returns: Sequence[Any], *, exposure: FinancialAmount, confidence: Any | None = None,
    ) -> HistoricalRisk:
        """Historical loss VaR + Expected Shortfall from simple fractional returns.

        Loss = -(exposure × return), VaR is empirical nearest-rank loss quantile;
        ES is average worst tail (fractional boundary included). Results are
        floored at zero when all relevant outcomes are gains. Neither estimates
        maximum possible future losses or guarantees portfolio coverage.
        """
        if not isinstance(exposure, FinancialAmount) or exposure.amount <= 0:
            raise FinancialDomainError("Historical risk exposure must be positive FinancialAmount")
        if not isinstance(returns, (list, tuple)):
            raise FinancialValidationError("returns must be a list/tuple")
        if not self.min_risk_observations <= len(returns) <= self.max_risk_observations:
            raise FinancialDomainError("Historical risk sample count outside configured limits")
        conf = self.default_confidence if confidence is None else to_decimal(confidence, "confidence")
        if not 0 < conf < 1:
            raise FinancialDomainError("Risk confidence must be between zero and one")
        observations = tuple(to_decimal(r, "return") for r in returns)
        with localcontext() as ctx:
            ctx.prec = self.calculation.precision
            losses_asc = sorted(-exposure.amount * r for r in observations)
            n = len(losses_asc)
            rank = max(1, int((conf * n).to_integral_value(rounding=ROUND_CEILING)))
            var = max(Decimal(0), losses_asc[rank - 1])
            tail_mass = (Decimal(1) - conf) * n
            count = int(tail_mass.to_integral_value(rounding=ROUND_FLOOR))
            remainder = tail_mass - count
            ordered = list(reversed(losses_asc))
            tail_sum = sum(ordered[:count], Decimal(0))
            if remainder:
                tail_sum += remainder * ordered[count]
            es = max(Decimal(0), tail_sum / tail_mass)
        logger.debug("Historical financial risk evaluated | observations=%d", n)
        return HistoricalRisk(exposure.currency.code, conf, n, FinancialAmount(var, exposure.currency), # type: ignore
                              FinancialAmount(es, exposure.currency))

    def maximum_drawdown(self, returns: Sequence[Any]) -> DrawdownResult:
        """Historical compounded drawdown from simple period returns.

        Starting wealth = 1, no flows or leverage, sequential equal-unit periods.
        Return < -100% is unsupported because this normalized wealth can turn
        negative and makes ordinary percentage drawdowns ambiguous.
        """
        if not isinstance(returns, (list, tuple)) or not returns or len(returns) > self.max_risk_observations:
            raise FinancialValidationError("drawdown returns must be bounded nonempty list/tuple")
        values = tuple(to_decimal(r, "return") for r in returns)
        if any(r < -1 for r in values):
            raise FinancialDomainError("Unlevered drawdown requires returns >= -100%")
        equity = peak = Decimal(1)
        peak_index = -1
        biggest = Decimal(0)
        best_peak = -1
        best_trough = -1
        with localcontext() as ctx:
            ctx.prec = self.calculation.precision
            for index, value in enumerate(values):
                equity *= 1 + value
                if equity > peak:
                    peak, peak_index = equity, index
                drawdown = Decimal(1) - (equity / peak)
                if drawdown > biggest:
                    biggest, best_peak, best_trough = drawdown, peak_index, index
        return DrawdownResult(biggest, best_peak, best_trough, equity)

    def stress_value(self, value: FinancialAmount, *, fractional_price_shock: Any) -> FinancialAmount:
        """Reprice one positive asset value; no probability or price forecast."""
        if not isinstance(value, FinancialAmount) or value.amount < 0:
            raise FinancialDomainError("Stress value requires nonnegative FinancialAmount")
        shock = to_decimal(fractional_price_shock, "fractional_price_shock")
        if shock < -1:
            raise FinancialDomainError("Unlevered price shock cannot be less than -100%")
        with localcontext() as ctx:
            ctx.prec = self.calculation.precision
            return FinancialAmount(value.amount * (1 + shock), value.currency)


__all__ = ["DCFValuation", "ValuationScenario", "ScenarioValuation", "HistoricalRisk",
           "DrawdownResult", "FinancialValuation"]

if __name__ == "__main__":
    print("\n=== Running Financial Valuation ===\n")
    printer.status("TEST", "Financial Valuation initialized", "info")
    service = FinancialValuation()
    r = InterestRate(Decimal("0.10"), RateConvention.EFFECTIVE)
    flows = (CashFlow(date(2024, 1, 1), FinancialAmount.of("-100", "EUR")),
             CashFlow(date(2025, 1, 1), FinancialAmount.of("120", "EUR")))
    pv = service.discounted_cashflow(flows, discount_rate=r, valuation_date="2024-01-01")
    assert pv.total_value.amount > 0
    assert len(service.scenario_valuations(flows, valuation_date="2024-01-01", scenarios=(ValuationScenario("base", r),))) == 1
    risk = service.historical_var_es(["-0.1", "0.2", "-0.05", "0", "0.01"], exposure=FinancialAmount.of(100, "EUR"))
    assert risk.expected_shortfall.amount >= risk.value_at_risk.amount
    assert service.maximum_drawdown(["0.1", "-0.2", "0.05"]).maximum_drawdown > 0
    print("\n=== Test ran successfully ===\n")
