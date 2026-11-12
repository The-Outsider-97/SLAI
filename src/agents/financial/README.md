# SLAI v2.4 Financial Subsystem

`src/agents/financial/` is the financial-domain substrate behind `FinancialAgent`. It owns financial semantics and reusable financial analysis capabilities — interpreting, modeling, calculating, analyzing and evaluating financial information and decisions — so that applications reuse one set of calculations, forecasting baselines and risk measures instead of each implementing their own. It produces **sourced, auditable analysis results**; it never moves money, places orders or contacts a provider.

General computation, learning, optimization, simulation, scheduling, authorization and external execution stay in the existing SLAI components. The subsystem only holds the financial meaning of the numbers.

## Scope

1. **Calculation** – time value of money (present/future value, NPV, XIRR, loan payments, amortization), day-count year fractions, periodic rates. Every convention is passed explicitly; nothing is rounded implicitly.
2. **Cash flows** – recurrence schedules with month-end handling, running balances, grouped totals, discounting.
3. **Valuation and risk** – discounted cash flow, Gordon terminal value, scenario and stress valuation, historical VaR / expected shortfall, maximum drawdown.
4. **Economics** – inflation and purchasing power, real vs. nominal rates, price-only returns, source-bound economic series (by vintage), descriptive accounting ratios and balance-sheet reconciliation.
5. **Market data** (`financial_data` + `data/`) – bitemporal, point-in-time observations and exchange-rate quotes, a currency catalog, explicit FX conversion with an audit trail, and return series.
6. **Subscription** (`financial_subscription` + `business/`) – billing periods, invoices with ordered charges, proration, cancellation credit and plan-change adjustment, forecasts (also in a reporting currency), MRR movement and retention, customer lifetime value.
7. **Portfolio** (`financial_portfolio` + `business/`) – position valuation from sourced prices, weights, rebalance value differences and costs, time-weighted / money-weighted / modified-Dietz return, return statistics, estimated moments for allocation.
8. **Intelligence** – deterministic baseline forecasts, leak-free walk-forward backtests, valuation classification, a manifest-checked contract for externally trained models.
9. **Memory and reports** – a bounded, checkpoint-safe analysis history and presentation-neutral reports (JSON, Markdown, HTML, SLDC).

It does **not** train models, solve optimization problems, simulate scenarios, authorize actions, fetch prices, hold balances, execute payments or trades, or keep an accounting ledger. Real payment actions stay in `src.functions.payment`.

## Layout

| Path | Role |
| --- | --- |
| `financial_types.py` | Immutable contracts: `FinancialAmount`, `Currency`, `CashFlow`, `InterestRate`, `FinancialPeriod`, `MarketObservation`, `ExchangeRate`, `FinancialResult` and friends |
| `data/` | Consumed by `financial_data`, `financial_subscription`, `financial_portfolio`, `financial_economics`: `currency_catalog` (ISO 4217 minor units), `observation_store` (bitemporal store), `series` (validation, returns, alignment), `fx` (explicit FX resolution) |
| `business/` | Consumed by `financial_subscription` and `financial_portfolio`: `rounding` (minor-unit quantization, exact allocation), `charges` (ordered discounts/fees/tax), `proration`, `recurring_metrics` (MRR, retention, CLV), `performance` (TWR, Dietz, statistics, moments) |
| `financial_calculation.py` … `financial_report.py` | The central handlers, one per domain (below) |
| `financial_memory.py` | Bounded, integrity-checked local analysis history |
| `configs/financial_config.yaml` | Subsystem precision, freshness, limits and memory sizing |
| `utils/` | `financial_errors` (`FIN-xxxx`, inherits `BaseError`), `financial_helpers`, `config_loader` |

Handlers compose bottom-up; each reuses the one below instead of re-implementing it:

```
FinancialCalculation → FinancialValuation / FinancialCashflow → FinancialEconomics / FinancialSubscription
FinancialData → FinancialPortfolio / FinancialIntelligence → FinancialReport / FinancialMemory
```

## Agent and configuration

`src/agents/financial_agent.py` is the facade (`FinancialAgent(BaseAgent)`; registered as `financial` / `financial_agent` in `AgentFactory`). Configuration is split on purpose:

| Owner | File | Holds |
| --- | --- | --- |
| Subsystem | `financial/configs/financial_config.yaml`, read via `utils/config_loader.py` | Decimal precision, freshness limits, record and position caps, memory size |
| Agent | `base/configs/agents_config.yaml` → `financial_agent` | Authorization mode, optimization on/off, batch size, remember/checkpoint/publish switches, SharedMemory keys and bounds, `allowed_operations` |

**The agent never reads `financial_config.yaml`.** It only builds the subsystem objects; limits stay the subsystem's concern.

### Two memories

- **Financial memory** (local, `FinancialMemory`) keeps the agent's own analysis history (`memory.recall`, `memory.recent`, `report.render`). It is included in checkpoints, together with the ingested market data, and restored atomically: state is validated and decoded first and rolled back on failure.
- **SharedMemory** carries coordination data: per-request records (money and holdings are replaced by item counts), result envelopes (long lists cut, the cut recorded in `truncated_items`), summaries, errors, the latest result and an event channel. Other agents can hand data in with `data.ingest_shared`, by publishing `{"observations": [...], "exchange_rates": [...]}` under `<input_key_prefix>:<name>`.

### Delegation

| Need | Delegated to | Notes |
| --- | --- | --- |
| Mean-variance allocation (`portfolio.optimize_allocation`) | `OptimizationAgent` | The agent builds the quadratic program from estimated moments; returned weights are re-quantized to Decimal, bounds-checked and renormalized to sum exactly 1. A missing or failing solver gives `unavailable`; a result is never fabricated |
| Approval of state-changing operations (`data.ingest_*`) | `SafetyAgent.validate_action` | `authorization_mode`: `disabled`, `if_available`, `required`. A reachable safety agent that errors or declines always blocks. Calculations and reads never ask |

Simulation, Learning, Planning and Execution agents are deliberately **not** connected: nothing in this subsystem needs them, and connecting them would imply payment or trade execution.

## Minimal usage

```python
from src.agents.financial import FinancialAmount, FinancialCalculation, InterestRate

calc = FinancialCalculation()
rate = InterestRate("0.06", "nominal", periods_per_year=12)
print(calc.loan_payment(FinancialAmount.of("10000", "EUR"), rate, periods=12, periods_per_year=12))   # 860.66… EUR
```

Through the agent:

```python
agent = factory.create("financial", shared_memory)
agent.perform_task({
    "operation": "calculation.present_value",
    "parameters": {
        "future": {"amount": "1000", "currency": "EUR"},
        "rate": {"value": "0.05", "convention": "effective"},
        "start": "2023-01-01", "end": "2024-01-01",
    },
})
```

### Request shape

| Field | Meaning |
| --- | --- |
| `operation` | `"domain.operation"`, or `domain` + `operation` separately |
| `parameters` | Operation arguments. Unknown keys are rejected so a typo can never silently change a result or mutate state |
| `request_id`, `metadata`, `created_at` | Correlation; generated when absent |
| `tasks` | A batch of independent requests (bounded by `max_batch_size`); the batch status is `success`, `partial` or `failed` |

Results are envelopes with `success`, `status`, `result`, `assumptions`, `sources`, `warnings`, `methodology`, `diagnostics` and a `fingerprint`. Expected domain failures are **failure envelopes** (`failed`; `insufficient_data` for missing or stale market data; `unavailable` for a peer or provider that is not there). Unexpected defects are logged, published and re-raised.

### Operations

| Domain | Operations |
| --- | --- |
| `calculation` | `year_fraction`, `periodic_rate`, `discount_factor`, `present_value`, `future_value`, `net_present_value`, `xirr`, `loan_payment`, `amortization_schedule` |
| `cashflow` | `recurring_schedule`, `summarize`, `monthly_totals`, `running_balances`, `discounted_value` |
| `valuation` | `discounted_cashflow`, `gordon_terminal_value`, `scenario_valuations`, `stress_value`, `historical_var_es`, `maximum_drawdown` |
| `economics` | `inflation_rate`, `cumulative_inflation`, `annualized_inflation_rate`, `adjust_purchasing_power`, `index_level`, `real_effective_rate`, `price_return`, `economic_series`, `accounting_ratios`, `reconcile_balance_sheet` |
| `data` | `ingest_observations`*, `ingest_exchange_rates`*, `ingest_shared`*, `latest_observation`, `observation_history`, `latest_exchange_rate`, `convert`, `counts` |
| `subscription` | `billing_period`, `invoice`, `prorated_charge`, `cancellation_credit`, `projected_charges`, `forecast`, `forecast_in_currency`, `forecast_by_currency`, `monthly_recurring_value`, `annual_recurring_value`, `recurring_value_movement`, `customer_lifetime_value` |
| `portfolio` | `value_positions`, `value_in_currency`, `allocation_weights`, `rebalance_amounts`, `rebalance_cost`, `time_weighted_return`, `money_weighted_return`, `modified_dietz_return`, `annualize_return`, `return_statistics`, `relative_statistics`, `return_moments`, `optimize_allocation` |
| `intelligence` | `forecast`, `backtest`, `assess_valuation`, `models` |
| `report` / `memory` | `render` / `recall`, `recent` |

\* mutating; subject to authorization. `register_forecast_model(manifest, predictor)` is Python-only because code cannot travel in a request, and `reset_local_state()` is the only way to clear stored data.

## Design rules

- **Decimal only.** Money and rates are `Decimal` with an explicit currency. JSON floats are rejected (`to_decimal` is strict); send strings or integers.
- **Nothing implicit.** Day count, rate convention, rounding mode and FX policy are always declared by the caller. Rounding is never applied unless asked, and allocations sum exactly (largest remainder).
- **Point-in-time.** Every data-dependent operation takes an explicit `as_of`; there is no implicit "now". Observations are bitemporal (`observed_at` and `retrieved_at`), so a read can never see data that was not yet known, and revisions are vintages (`economic_series` defaults to the latest vintage known at `as_of`).
- **No implicit FX.** A conversion needs a sourced quote; a reciprocal or a pivot currency must be opted into and every leg is reported.
- **Undefined is not zero.** Ratios with a zero denominator are `None`; periods under a year are not annualized unless `allow_short_period` is set (GIPS).
- **Honest uncertainty.** Estimated moments, Sharpe ratios and backtests carry warnings; a backtest refuses data that could not have been known at the evaluation origin rather than leaking it.

## Limits

- Stdlib only (`decimal`, `datetime`, `math`, `hashlib`); no numpy/scipy under `financial/`. The optimization peer needs its own dependencies.
- Market data lives in process memory and is checkpointed with the agent; use an authorized durable backend for production ingestion.
- Long-only unit-price positions (equity, ETF, fund, bond with an explicit quote multiplier). Dividends, splits, taxes, costs and shorting are not modelled; a quote series is not a total-return series.
- Baseline forecasts are real but simple models (naive, drift, seasonal); they are benchmarks, not validated market forecasts. Valuation signals are research classifications, not investment advice.
- Subscription results are projections, not invoices, revenue recognition or proof of entitlement.
- Not implemented: tax engines, accrual accounting, bond pricing conventions beyond supplied quotes, options and derivatives.

## Academic design basis

Markowitz (1952) portfolio selection; Sharpe (1964, 1966, 1994) and Jensen (1968) performance measures; Fisher (1930) real vs. nominal rates; Dietz (1966) and Bacon (2008) return measurement; CFA Institute GIPS; Berger & Nasr (1998) customer lifetime value; Gupta & Lehmann (2003) customer value; Croushore & Stark (2001) real-time data; Lo (2002) Sharpe ratio estimation error; Michaud (1989) estimation error in optimization; DeMiguel, Garlappi & Uppal (2009) 1/N benchmark; Rockafellar & Uryasev (2002) and Acerbi & Tasche (2002) expected shortfall; Gordon & Shapiro (1956) terminal value; ISO 4217 currency codes.

## Testing

```bash
pytest tests/agents/test_financial_subsystem.py tests/agents/test_financial_agent.py
```

Each module also has a runnable self-test: `python -m src.agents.financial.financial_portfolio`, `python -m src.agents.financial_agent`, and so on.
