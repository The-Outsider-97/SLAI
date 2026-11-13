"""
SLAI Financial Agent. Top-level orchestration facade for the Financial subsystem.

FinancialAgent is SLAI's specialized financial intelligence component. It interprets, models, calculates, analyzes
and evaluates financial information and financial decisions. It owns financial-domain semantics and reusable
financial analysis capabilities (time value of money, cash flows, valuation and historical risk, subscription and
recurring-revenue economics, portfolio valuation and performance measurement, macro-economic and accounting
analysis, point-in-time market data, forecasting and research signals) and delegates everything that is not
financial-domain semantics to existing SLAI components:

- mean-variance optimization            -> OptimizationAgent (optional, ``portfolio.optimize_allocation`` only)
- authorization of state mutations      -> SafetyAgent (optional, governed by ``authorization_mode``)
- coordination and result hand-off      -> SharedMemory (BaseAgent) ; analysis history -> FinancialMemory
- scheduling, simulation, learning, payment or trade execution are deliberately NOT performed here.

Academic basis of the capabilities reached through this facade (the subsystem modules cite them in detail):

- Fisher (1930), The Theory of Interest.                                   (real vs nominal rates)
- Markowitz (1952), Portfolio Selection; Michaud (1989), The Markowitz Optimization Enigma;
  DeMiguel, Garlappi & Uppal (2009), Optimal Versus Naive Diversification.  (mean-variance, estimation error)
- Sharpe (1964, 1966, 1994); Jensen (1968).                                (risk-adjusted performance)
- Dietz (1966); Bacon (2008); GIPS.                                        (time-/money-weighted returns)
- Berger & Nasr (1998); Gupta & Lehmann (2003).                            (customer lifetime value)
- Croushore & Stark (2001).                                                (real-time data vintages)
- Lo (2002), The Statistics of Sharpe Ratios.

Contract summary
- Money is Decimal-only with explicit currency. JSON floats are rejected: send decimal strings or integers.
- Every data-dependent operation takes an explicit ``as_of`` (point in time). Nothing defaults to "now".
- Expected domain failures (bad input, missing/stale data, denied authorization, unavailable peer) are returned as
  a failure-status envelope, never raised; unexpected defects propagate to BaseAgent's recovery path.
- The agent never reads the subsystem's financial_config.yaml; subsystem limits belong to the subsystem and
  agent-level policy lives in ``financial_agent`` of ``agents_config.yaml`` only.
"""

from __future__ import annotations

__version__ = "2.4.0"

import threading
import uuid

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass, fields
from datetime import date, datetime, timedelta
from decimal import Decimal, localcontext
from typing import Any, Callable, ClassVar, overload

from .base_agent import BaseAgent
from .runtime_contracts import RuntimeLifecycle
from .base.failure_taxonomy import DEFAULT_FAILURE_TAXONOMY
from .base.utils.base_errors import BaseConfigurationError, BaseStateError
from .base.utils.base_helpers import coerce_bool, coerce_int, monotonic_seconds, redact_mapping, to_json_safe
from .base.utils.config_contract import assert_valid_config_contract
from .base.utils.main_config_loader import get_config_section
from .financial import (
    AccountingSnapshot, BillingInterval, CashFlow, Charge, DayCountConvention, ExchangeRate, ExternalFlow,
    FinancialAmount, FinancialPeriod, FinancialCalculation, FinancialCashflow, FinancialData, FinancialDataSource, FinancialEconomics,
    FinancialIntelligence, FinancialMemory, FinancialModelManifest, FinancialPortfolio, FinancialPredictor,
    FinancialReport, FinancialRequest, FinancialResult, FinancialResultStatus, FinancialSubscription,
    FinancialValuation, FxPolicy, InterestRate, MarketObservation, PortfolioHolding, RoundingPolicy,
    SubscriptionContract, ValuationPoint, ValuationScenario,
)
from .financial.utils.financial_errors import (
    FinancialDataError, FinancialDelegationError, FinancialError, FinancialModelUnavailableError,
    FinancialProviderError, FinancialValidationError, MissingMarketDataError, StaleMarketDataError,
    UnauthorizedFinancialOperationError, UnsupportedFinancialOperationError,
)
from .financial.utils.financial_helpers import (
    normalize_identifier, require_text, stable_fingerprint, to_date, to_decimal, to_json_compatible, to_utc_datetime,
)
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Financial Agent")
printer = PrettyPrinter()

_AUTHORIZATION_MODES = ("disabled", "if_available", "required")
_ONE = Decimal(1)
# Structural cap for to_json_safe: it counts mapping keys too, so it must never be the (small) list limit or it would
# cut the envelope itself. Long result lists are trimmed, and the cut recorded, by FinancialAgent._bounded.
_SHARED_STRUCTURE_ITEMS = 1024


# ----------------------------------------------------------------------------------------------------------------------
# Request decoding
# ----------------------------------------------------------------------------------------------------------------------

def _decode(factory: Callable[[Any], Any], value: Any, label: str) -> Any:
    """Build a subsystem value from request data; malformed input is a validation error, never a bare KeyError."""

    try:
        return factory(value)
    except FinancialError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise FinancialValidationError(
            f"{label} is malformed", context={"parameter": label, "error_type": type(exc).__name__}, cause=exc
        ) from exc


class _Params:
    """Typed, strict view of one request's ``parameters``.

    Every accessor records the key it consumed; ``unused()`` then exposes misspelled or unsupported parameters so a
    typo can never silently change an analysis. Numbers must be decimal strings or integers (no binary floats).
    """

    __slots__ = ("_data", "_used", "_item_limit", "request_id")

    def __init__(self, data: Mapping[str, Any], *, item_limit: int, request_id: str = "") -> None:
        self._data = data
        self._used: set[str] = set()
        self._item_limit = item_limit
        self.request_id = request_id

    # -- presence -----------------------------------------------------------------------------------------------

    def has(self, key: str) -> bool:
        self._used.add(key)
        return self._data.get(key) is not None

    def unused(self) -> list[str]:
        return sorted(str(key) for key in self._data if key not in self._used)

    def _take(self, key: str, *, required: bool) -> Any:
        self._used.add(key)
        value = self._data.get(key)
        if value is None and required:
            raise FinancialValidationError(f"parameter '{key}' is required", context={"parameter": key})
        return value

    # -- scalars ------------------------------------------------------------------------------------------------

    def text(self, key: str) -> str:
        return require_text(self._take(key, required=True), key)

    @overload
    def opt_text(self, key: str, default: str) -> str: ...

    @overload
    def opt_text(self, key: str, default: None = None) -> str | None: ...

    def opt_text(self, key: str, default: str | None = None) -> str | None:
        value = self._take(key, required=False)
        return default if value is None else require_text(value, key)

    def decimal(self, key: str) -> Decimal:
        return to_decimal(self._take(key, required=True), key)

    @overload
    def opt_decimal(self, key: str, default: Decimal) -> Decimal: ...

    @overload
    def opt_decimal(self, key: str, default: None = None) -> Decimal | None: ...

    def opt_decimal(self, key: str, default: Decimal | None = None) -> Decimal | None:
        value = self._take(key, required=False)
        return default if value is None else to_decimal(value, key)

    def integer(self, key: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
        return self._int(key, self._take(key, required=True), minimum, maximum)

    @overload
    def opt_integer(self, key: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int: ...

    @overload
    def opt_integer(self, key: str, default: None = None, *, minimum: int | None = None, maximum: int | None = None) -> int | None: ...

    def opt_integer(self, key: str, default: int | None = None, *, minimum: int | None = None, maximum: int | None = None) -> int | None:
        value = self._take(key, required=False)
        return default if value is None else self._int(key, value, minimum, maximum)

    @staticmethod
    def _int(key: str, value: Any, minimum: int | None, maximum: int | None) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
                value = int(value.strip())
            else:
                raise FinancialValidationError(f"parameter '{key}' must be an integer", context={"parameter": key})
        if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
            raise FinancialValidationError(
                f"parameter '{key}' is outside its permitted range",
                context={"parameter": key, "minimum": minimum, "maximum": maximum},
            )
        return value

    def boolean(self, key: str, default: bool = False) -> bool:
        value = self._take(key, required=False)
        if value is None:
            return default
        if not isinstance(value, bool):
            raise FinancialValidationError(f"parameter '{key}' must be a boolean", context={"parameter": key})
        return value

    def opt_boolean(self, key: str) -> bool | None:
        value = self._take(key, required=False)
        if value is not None and not isinstance(value, bool):
            raise FinancialValidationError(f"parameter '{key}' must be a boolean", context={"parameter": key})
        return value

    def date(self, key: str) -> date:
        return to_date(self._take(key, required=True), key)

    def opt_date(self, key: str) -> date | None:
        value = self._take(key, required=False)
        return None if value is None else to_date(value, key)

    def moment(self, key: str) -> datetime:
        return to_utc_datetime(self._take(key, required=True), key)

    def opt_moment(self, key: str) -> datetime | None:
        value = self._take(key, required=False)
        return None if value is None else to_utc_datetime(value, key)

    def max_age(self, key: str = "max_age_seconds") -> timedelta | None:
        seconds = self.opt_integer(key, minimum=1, maximum=315_360_000)
        return None if seconds is None else timedelta(seconds=seconds)

    # -- containers ---------------------------------------------------------------------------------------------

    def mapping(self, key: str) -> dict[str, Any]:
        return self._as_mapping(key, self._take(key, required=True))

    def opt_mapping(self, key: str) -> dict[str, Any] | None:
        value = self._take(key, required=False)
        return None if value is None else self._as_mapping(key, value)

    @staticmethod
    def _as_mapping(key: str, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise FinancialValidationError(f"parameter '{key}' must be a mapping", context={"parameter": key})
        return dict(value)

    def items(self, key: str, *, minimum: int = 1) -> list[Any]:
        value = self._take(key, required=True)
        return self._as_items(key, value, minimum)

    def opt_items(self, key: str) -> list[Any] | None:
        value = self._take(key, required=False)
        return None if value is None else self._as_items(key, value, 0)

    def _as_items(self, key: str, value: Any, minimum: int) -> list[Any]:
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            raise FinancialValidationError(f"parameter '{key}' must be a sequence", context={"parameter": key})
        if len(value) < minimum or len(value) > self._item_limit:
            raise FinancialValidationError(
                f"parameter '{key}' must hold between {minimum} and {self._item_limit} items",
                context={"parameter": key, "size": len(value)},
            )
        return list(value)

    def strings(self, key: str, *, minimum: int = 1) -> list[str]:
        values = [require_text(item, key) for item in self.items(key, minimum=minimum)]
        if len(set(values)) != len(values):
            raise FinancialValidationError(f"parameter '{key}' must not contain duplicates", context={"parameter": key})
        return values

    def decimals(self, key: str, *, minimum: int = 1) -> list[Decimal]:
        return [to_decimal(item, key) for item in self.items(key, minimum=minimum)]

    # -- domain values ------------------------------------------------------------------------------------------

    def build(self, key: str, factory: Callable[[Any], Any]) -> Any:
        return _decode(factory, self._take(key, required=True), key)

    def opt_build(self, key: str, factory: Callable[[Any], Any]) -> Any:
        value = self._take(key, required=False)
        return None if value is None else _decode(factory, value, key)

    def build_many(self, key: str, factory: Callable[[Any], Any], *, minimum: int = 1) -> tuple[Any, ...]:
        return tuple(_decode(factory, item, f"{key}[{index}]") for index, item in enumerate(self.items(key, minimum=minimum)))

    def opt_build_many(self, key: str, factory: Callable[[Any], Any]) -> tuple[Any, ...]:
        items = self.opt_items(key)
        return () if not items else tuple(_decode(factory, item, f"{key}[{index}]") for index, item in enumerate(items))

    def amount(self, key: str) -> FinancialAmount:
        return self.build(key, _amount)

    def opt_amount(self, key: str) -> FinancialAmount | None:
        return self.opt_build(key, _amount)

    def rate(self, key: str) -> InterestRate:
        return self.build(key, _rate)

    def opt_rate(self, key: str) -> InterestRate | None:
        return self.opt_build(key, _rate)

    def flows(self, key: str = "flows") -> tuple[CashFlow, ...]:
        return self.build_many(key, _cash_flow)

    def observations(self, key: str = "observations") -> tuple[MarketObservation, ...]:
        return self.build_many(key, _observation)

    def fx_policy(self, key: str = "policy") -> FxPolicy | None:
        spec = self.opt_mapping(key)
        if spec is None:
            return None
        unknown = sorted(set(spec) - {"allow_inverse", "pivot_currency", "max_age_seconds", "source_id"})
        if unknown:
            raise FinancialValidationError("unknown FX policy fields", context={"parameter": key, "unknown": unknown})
        seconds = spec.get("max_age_seconds")
        return _decode(
            lambda _: FxPolicy(
                allow_inverse=_policy_bool(spec.get("allow_inverse", False)),
                pivot_currency=spec.get("pivot_currency"),
                max_age=None if seconds is None else timedelta(seconds=_whole_seconds(seconds)),
                source_id=spec.get("source_id"),
            ),
            spec,
            key,
        )

    def rounding(self, key: str = "rounding") -> RoundingPolicy | None:
        spec = self.opt_mapping(key)
        if spec is None:
            return None
        if set(spec) != {"mode"}:
            raise FinancialValidationError("rounding must be a mapping with exactly one key: mode", context={"parameter": key})
        return _decode(lambda _: RoundingPolicy(spec["mode"]), spec, key)


def _policy_bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise FinancialValidationError("FX policy allow_inverse must be a boolean")
    return value


def _whole_seconds(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FinancialValidationError("seconds must be a positive integer")
    return value


def _mapping_of(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinancialValidationError("expected a mapping")
    return value


def _amount(value: Any) -> FinancialAmount:
    return FinancialAmount.from_dict(_mapping_of(value))


def _rate(value: Any) -> InterestRate:
    return InterestRate.from_dict(_mapping_of(value))


def _cash_flow(value: Any) -> CashFlow:
    return CashFlow.from_dict(_mapping_of(value))


def _observation(value: Any) -> MarketObservation:
    return MarketObservation.from_dict(_mapping_of(value))


def _exchange_rate(value: Any) -> ExchangeRate:
    return ExchangeRate.from_dict(_mapping_of(value))


def _holding(value: Any) -> PortfolioHolding:
    return PortfolioHolding.from_dict(_mapping_of(value))


def _contract(value: Any) -> SubscriptionContract:
    return SubscriptionContract.from_dict(_mapping_of(value))


def _charge(value: Any) -> Charge:
    return Charge.from_dict(_mapping_of(value))


def _external_flow(value: Any) -> ExternalFlow:
    spec = _mapping_of(value)
    return ExternalFlow(to_utc_datetime(spec["at"], "at"), _amount(spec["amount"]))


def _valuation_point(value: Any) -> ValuationPoint:
    spec = _mapping_of(value)
    return ValuationPoint(to_utc_datetime(spec["at"], "at"), _amount(spec["value"]))


def _scenario(value: Any) -> ValuationScenario:
    spec = _mapping_of(value)
    return ValuationScenario(
        require_text(spec["name"], "name"),
        _rate(spec["discount_rate"]),
        to_decimal(spec.get("cashflow_multiplier", 1), "cashflow_multiplier"),
    )


def _required_age(p: _Params) -> timedelta:
    age = p.max_age()
    if age is None:
        raise FinancialValidationError("parameter 'max_age_seconds' is required", context={"parameter": "max_age_seconds"})
    return age


def _period(value: Any) -> FinancialPeriod:
    return FinancialPeriod.from_dict(_mapping_of(value))


# ----------------------------------------------------------------------------------------------------------------------
# Operation model
# ----------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FinancialOperation:
    """One reusable financial capability: ``domain.operation``, its handler and its provenance."""

    name: str
    handler: Callable[[_Params], Any]
    summary: str
    basis: str | None = None
    mutating: bool = False

    @property
    def domain(self) -> str:
        return self.name.split(".", 1)[0]

    def describe(self) -> dict[str, object]:
        return {"operation": self.name, "summary": self.summary, "basis": self.basis, "mutating": self.mutating}


@dataclass(frozen=True, slots=True)
class _Outcome:
    """A handler's value plus the provenance it wants recorded on the FinancialResult."""

    value: Any
    assumptions: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    methodology: str | None = None
    model_version: str | None = None
    status: FinancialResultStatus = FinancialResultStatus.SUCCESS


def _failure_status(error: FinancialError) -> FinancialResultStatus:
    """Missing/stale evidence and unreachable peers are distinct from invalid requests and failed calculations."""

    if isinstance(error, (MissingMarketDataError, StaleMarketDataError)):
        return FinancialResultStatus.INSUFFICIENT_DATA
    if isinstance(error, (FinancialModelUnavailableError, FinancialProviderError, FinancialDelegationError)):
        return FinancialResultStatus.UNAVAILABLE
    return FinancialResultStatus.FAILED


def _gather_sources(value: Any, *, _depth: int = 0, _found: dict[tuple[Any, ...], FinancialDataSource] | None = None) -> tuple[FinancialDataSource, ...]:
    """Data sources that evidence a value, found structurally (observations, quotes, conversions, positions)."""

    found = {} if _found is None else _found
    if _depth > 6 or value is None or isinstance(value, (str, bytes, int, float, Decimal, date)):
        return tuple(found.values())
    if isinstance(value, FinancialDataSource):
        found.setdefault((value.source_id, value.retrieved_at, value.revision, value.reference), value)
    elif isinstance(value, Mapping):
        for item in value.values():
            _gather_sources(item, _depth=_depth + 1, _found=found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _gather_sources(item, _depth=_depth + 1, _found=found)
    elif is_dataclass(value) and not isinstance(value, type):
        for spec in fields(value):
            _gather_sources(getattr(value, spec.name), _depth=_depth + 1, _found=found)
    return tuple(found.values())


# ----------------------------------------------------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------------------------------------------------

class FinancialAgent(BaseAgent):
    """Production orchestration facade over ``src.agents.financial``.

    The subsystem owns financial semantics, numerics and its own limits (via its private configuration). The agent
    owns SLAI integration: request normalization, operation routing, agent-level policy from ``agents_config.yaml``,
    authorization and optimization delegation, shared-memory coordination, local analysis memory, metrics and
    checkpoint participation. Analyses never execute payments, trades or any external side effect.
    """

    AGENT_KEY: ClassVar[str] = "financial_agent"
    CHECKPOINTING_SUPPORTED = True
    CHECKPOINT_SCHEMA: ClassVar[str] = "slai.financial_agent.state.v1"
    # "insufficient_data" is a FinancialResultStatus the default taxonomy does not know; it is not a success.
    FAILURE_TAXONOMY = DEFAULT_FAILURE_TAXONOMY.with_statuses("insufficient_data")

    _ALLOWED_CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset({
        "enabled", "allowed_operations", "authorization_mode", "optimization_enabled",
        "max_batch_size", "max_parameter_items", "metrics_enabled",
        "remember_results", "remember_failures",
        "checkpoint_local_memory", "checkpoint_market_data", "restore_local_memory",
        "publish_shared_memory", "fail_on_shared_memory_error", "shared_memory_ttl_seconds",
        "max_shared_list_items", "max_shared_string_chars",
        "request_key_prefix", "result_key_prefix", "summary_key_prefix", "error_key_prefix", "latest_result_key",
        "input_key_prefix", "event_channel",
    })

    _TASK_KEYS: ClassVar[frozenset[str]] = frozenset({"request_id", "domain", "operation", "parameters", "metadata", "created_at"})

    def __init__(
        self,
        shared_memory: object,
        agent_factory: object,
        config: Mapping[str, object] | None = None,
        *,
        checkpoint_manager: object | None = None,
        data: FinancialData | None = None,
        memory: FinancialMemory | None = None,
    ) -> None:
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, checkpoint_manager=checkpoint_manager)

        # Agent-level policy comes from agents_config.yaml only. The subsystem's configuration is the subsystem's.
        self.agent_config: dict[str, object] = dict(get_config_section(self.AGENT_KEY) or {})
        if config is not None:
            if not isinstance(config, Mapping):
                raise BaseConfigurationError("FinancialAgent config override must be a mapping", component=self.name)
            self.agent_config.update(dict(config))
        assert_valid_config_contract(
            global_config=self.config,
            agent_key=self.AGENT_KEY,
            agent_config=self.agent_config,
            agent_allowed_keys=self._ALLOWED_CONFIG_KEYS,
            require_global_keys=False,
            require_agent_section=False,
            warn_unknown_global_keys=False,
            logger=logger,
        )
        self._load_agent_config()
        self._operations: dict[str, FinancialOperation] = self._build_registry()
        self._validate_agent_config()

        # One market-data store is shared by every analytical component so that all of them see the same evidence.
        if data is not None and not isinstance(data, FinancialData):
            raise BaseConfigurationError("data must be a FinancialData instance", component=self.name)
        if memory is not None and not isinstance(memory, FinancialMemory):
            raise BaseConfigurationError("memory must be a FinancialMemory instance", component=self.name)
        self.data = data or FinancialData()
        self.valuation = FinancialValuation()
        self.calculation: FinancialCalculation = self.valuation.calculation
        self.cashflow = FinancialCashflow()
        self.subscription = FinancialSubscription()
        self.economics = FinancialEconomics(self.data)
        self.portfolio = FinancialPortfolio(self.data, self.valuation)
        self.intelligence = FinancialIntelligence(self.data, self.valuation)
        self.report = FinancialReport()
        self.memory = memory or FinancialMemory()  # financial (local) memory; SharedMemory is BaseAgent's

        self._peers: dict[str, Any] = {}
        self._peer_lock = threading.RLock()
        self._metrics_lock = threading.RLock()
        self._requests_total = 0
        self._requests_success = 0
        self._requests_failed = 0
        self._elapsed_total = 0.0
        self._by_operation: Counter[str] = Counter()
        self._by_status: Counter[str] = Counter()
        self._error_types: Counter[str] = Counter()
        self._shared_memory_errors = 0

        logger.info(
            "Financial Agent initialized | operations=%d | authorization=%s | optimization=%s | shared_memory=%s",
            len(self.allowed_operations), self.authorization_mode, self.optimization_enabled, self.publish_shared_memory,
        )
        self._publish_event("initialized", {"agent_id": self.agent_id})

    # ------------------------------------------------------------------------------------------------------------
    # Agent-level configuration (agents_config.yaml -> financial_agent)
    # ------------------------------------------------------------------------------------------------------------

    def _cfg(self, key: str, default: object) -> object:
        return self.agent_config.get(key, default)

    def _load_agent_config(self) -> None:
        self.enabled = coerce_bool(self._cfg("enabled", True), True)
        self.authorization_mode = str(self._cfg("authorization_mode", "if_available")).strip().lower()
        self.optimization_enabled = coerce_bool(self._cfg("optimization_enabled", True), True)
        self.metrics_enabled = coerce_bool(self._cfg("metrics_enabled", True), True)
        self.max_batch_size = coerce_int(self._cfg("max_batch_size", 100), 100, minimum=1, maximum=10_000)
        self.max_parameter_items = coerce_int(self._cfg("max_parameter_items", 10_000), 10_000, minimum=1, maximum=1_000_000)

        self.remember_results = coerce_bool(self._cfg("remember_results", True), True)
        self.remember_failures = coerce_bool(self._cfg("remember_failures", False), False)
        self.checkpoint_local_memory = coerce_bool(self._cfg("checkpoint_local_memory", True), True)
        self.checkpoint_market_data = coerce_bool(self._cfg("checkpoint_market_data", True), True)
        self.restore_local_memory = coerce_bool(self._cfg("restore_local_memory", True), True)

        self.publish_shared_memory = coerce_bool(self._cfg("publish_shared_memory", True), True)
        self.fail_on_shared_memory_error = coerce_bool(self._cfg("fail_on_shared_memory_error", False), False)
        self.shared_memory_ttl_seconds = coerce_int(self._cfg("shared_memory_ttl_seconds", 604800), 604800, minimum=0, maximum=31_536_000)
        self.max_shared_list_items = coerce_int(self._cfg("max_shared_list_items", 128), 128, minimum=1, maximum=100_000)
        self.max_shared_string_chars = coerce_int(self._cfg("max_shared_string_chars", 8192), 8192, minimum=64, maximum=16_777_216)

        for attribute, default in (
            ("request_key_prefix", "financial_agent.request"),
            ("result_key_prefix", "financial_agent.result"),
            ("summary_key_prefix", "financial_agent.summary"),
            ("error_key_prefix", "financial_agent.error"),
            ("latest_result_key", "financial_agent.latest"),
            ("input_key_prefix", "financial_agent.input"),
            ("event_channel", "financial_agent.events"),
        ):
            setattr(self, attribute, str(self._cfg(attribute, default) or "").strip())

        configured = self._cfg("allowed_operations", None)
        if configured is None:
            self._configured_operations: set[str] | None = None
        elif isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
            self._configured_operations = {str(item).strip().lower() for item in configured if str(item).strip()}
        else:
            raise BaseConfigurationError("financial_agent.allowed_operations must be a sequence of operation names", component=self.name)

    def _validate_agent_config(self) -> None:
        if self.authorization_mode not in _AUTHORIZATION_MODES:
            raise BaseConfigurationError(
                "financial_agent.authorization_mode is not supported",
                component=self.name,
                context={"actual": self.authorization_mode, "supported": list(_AUTHORIZATION_MODES)},
            )
        for attribute in ("request_key_prefix", "result_key_prefix", "summary_key_prefix", "error_key_prefix",
                          "latest_result_key", "input_key_prefix", "event_channel"):
            if not getattr(self, attribute):
                raise BaseConfigurationError(
                    f"financial_agent.{attribute} must be a non-empty string", component=self.name, context={"field": attribute}
                )
        if self._configured_operations is None:
            self.allowed_operations = frozenset(self._operations)
        else:
            unknown = self._configured_operations - set(self._operations)
            if unknown:
                raise BaseConfigurationError(
                    "financial_agent.allowed_operations contains unsupported operations",
                    component=self.name,
                    context={"unknown": sorted(unknown)},
                )
            self.allowed_operations = frozenset(self._configured_operations)
        if self.publish_shared_memory:
            missing = [name for name in ("get", "set") if not callable(getattr(self.shared_memory, name, None))]
            if missing:
                raise BaseConfigurationError(
                    "SharedMemory does not satisfy FinancialAgent's get/set contract",
                    component=self.name,
                    context={"missing_methods": missing},
                )

    # ------------------------------------------------------------------------------------------------------------
    # Operation registry
    # ------------------------------------------------------------------------------------------------------------

    def _build_registry(self) -> dict[str, FinancialOperation]:
        registry: dict[str, FinancialOperation] = {}

        def add(name: str, handler: Callable[[_Params], Any], summary: str, basis: str | None = None, *, mutating: bool = False) -> None:
            registry[name] = FinancialOperation(name, handler, summary, basis, mutating)

        # time value of money
        add("calculation.year_fraction", self._calc_year_fraction, "Year fraction between two dates under a day-count convention")
        add("calculation.discount_factor", self._calc_discount_factor, "Discount factor of a rate between two dates")
        add("calculation.present_value", self._calc_present_value, "Present value of a future amount")
        add("calculation.future_value", self._calc_future_value, "Future value of a present amount")
        add("calculation.net_present_value", self._calc_net_present_value, "Net present value of dated cash flows")
        add("calculation.periodic_rate", self._calc_periodic_rate, "Periodic rate equivalent to an annual rate")
        add("calculation.loan_payment", self._calc_loan_payment, "Level annuity payment of an amortizing loan")
        add("calculation.amortization_schedule", self._calc_amortization_schedule, "Full amortization schedule of a loan")
        add("calculation.xirr", self._calc_xirr, "Annualized internal rate of return of dated cash flows (XIRR)")
        # cash flows
        add("cashflow.recurring_schedule", self._cf_recurring_schedule, "Calendar-aware recurring cash-flow schedule")
        add("cashflow.summarize", self._cf_summarize, "Inflow/outflow/net summary of cash flows over a window")
        add("cashflow.monthly_totals", self._cf_monthly_totals, "Net cash flow per calendar month")
        add("cashflow.running_balances", self._cf_running_balances, "Running balance after each cash-flow date")
        add("cashflow.discounted_value", self._cf_discounted_value, "Present value of a cash-flow schedule")
        # valuation and historical risk
        add("valuation.gordon_terminal_value", self._val_gordon, "Gordon-growth terminal value", "Gordon & Shapiro (1956)")
        add("valuation.discounted_cashflow", self._val_dcf, "Discounted cash-flow valuation with optional terminal value")
        add("valuation.scenario_valuations", self._val_scenarios, "Valuation under named discount-rate/cash-flow scenarios")
        add("valuation.historical_var_es", self._val_var_es, "Historical Value-at-Risk and Expected Shortfall",
            "Rockafellar & Uryasev (2002); Acerbi & Tasche (2002)")
        add("valuation.maximum_drawdown", self._val_drawdown, "Maximum drawdown of a return series")
        add("valuation.stress_value", self._val_stress, "Value after an explicit fractional price shock")
        # macro-economic and accounting analysis
        add("economics.inflation_rate", self._eco_inflation_rate, "Inflation between two index levels")
        add("economics.annualized_inflation_rate", self._eco_annualized_inflation, "Annualized inflation between two dated index levels")
        add("economics.adjust_purchasing_power", self._eco_adjust, "Restate an amount at another price level")
        add("economics.real_effective_rate", self._eco_real_rate, "Real rate from nominal rate and inflation", "Fisher (1930)")
        add("economics.price_return", self._eco_price_return, "Simple price return between two observations")
        add("economics.accounting_ratios", self._eco_ratios, "Standard accounting ratios from a statement snapshot")
        add("economics.reconcile_balance_sheet", self._eco_reconcile, "Assets = liabilities + equity reconciliation")
        add("economics.economic_series", self._eco_series, "Point-in-time economic indicator series (latest or all vintages)",
            "Croushore & Stark (2001)")
        add("economics.index_level", self._eco_index_level, "Point-in-time index level on a date")
        add("economics.cumulative_inflation", self._eco_cumulative_inflation, "Point-in-time cumulative inflation between dates")
        # market data
        add("data.ingest_observations", self._data_ingest_observations, "Ingest sourced market observations (bitemporal)", mutating=True)
        add("data.ingest_exchange_rates", self._data_ingest_rates, "Ingest sourced exchange-rate quotes (bitemporal)", mutating=True)
        add("data.ingest_shared", self._data_ingest_shared, "Ingest observations/quotes another agent published to SharedMemory", mutating=True)
        add("data.latest_observation", self._data_latest_observation, "Latest observation known at a point in time")
        add("data.latest_exchange_rate", self._data_latest_rate, "Latest exchange-rate quote known at a point in time")
        add("data.observation_history", self._data_history, "Observation history known at a point in time")
        add("data.convert", self._data_convert, "Explicit, auditable currency conversion")
        add("data.counts", self._data_counts, "Sizes of the local market-data store")
        # subscriptions and recurring revenue
        add("subscription.invoice", self._sub_invoice, "Charge breakdown of one billing date")
        add("subscription.projected_charges", self._sub_projected, "Projected charges of one contract")
        add("subscription.forecast", self._sub_forecast, "Projected charges of several contracts")
        add("subscription.forecast_by_currency", self._sub_forecast_by_currency, "Forecast summarized per currency")
        add("subscription.forecast_in_currency", self._sub_forecast_in_currency, "Forecast converted explicitly into one currency")
        add("subscription.monthly_recurring_value", self._sub_mrr, "Monthly recurring value of a contract")
        add("subscription.annual_recurring_value", self._sub_arr, "Annual recurring value of a contract")
        add("subscription.billing_period", self._sub_billing_period, "Billing period containing a date")
        add("subscription.prorated_charge", self._sub_prorated, "Prorated charge for part of a billing period")
        add("subscription.cancellation_credit", self._sub_cancellation, "Credit for the unused part of a paid period")
        add("subscription.recurring_value_movement", self._sub_movement, "MRR movement and retention between two dates")
        add("subscription.customer_lifetime_value", self._sub_clv, "Expected discounted margin of one customer",
            "Berger & Nasr (1998); Gupta & Lehmann (2003)")
        # portfolio valuation and performance
        add("portfolio.value_positions", self._pf_value_positions, "Mark holdings to point-in-time quotes")
        add("portfolio.allocation_weights", self._pf_weights, "Within-currency portfolio weights")
        add("portfolio.rebalance_amounts", self._pf_rebalance, "Value differences to reach target weights (no trading)")
        add("portfolio.rebalance_cost", self._pf_rebalance_cost, "Explicit fee cost of a rebalance plan")
        add("portfolio.value_in_currency", self._pf_value_in_currency, "Portfolio value in one reporting currency via explicit FX")
        add("portfolio.return_moments", self._pf_moments, "Annualized mean vector and covariance of instrument returns", "Markowitz (1952)")
        add("portfolio.return_statistics", self._pf_return_statistics, "Mean, volatility and Sharpe ratio of a return series",
            "Sharpe (1966, 1994); Lo (2002)")
        add("portfolio.relative_statistics", self._pf_relative_statistics, "Beta, alpha, tracking error, information ratio",
            "Sharpe (1964); Jensen (1968)")
        add("portfolio.annualize_return", self._pf_annualize, "Geometric annualization of a total return", "GIPS")
        add("portfolio.time_weighted_return", self._pf_twr, "Time-weighted return, valuing at each external flow", "GIPS; Bacon (2008)")
        add("portfolio.modified_dietz_return", self._pf_dietz, "Modified Dietz return", "Dietz (1966)")
        add("portfolio.money_weighted_return", self._pf_mwr, "Money-weighted return (investor IRR)", "Bacon (2008)")
        if self.optimization_enabled:
            add("portfolio.optimize_allocation", self._pf_optimize, "Mean-variance target weights, solved by OptimizationAgent",
                "Markowitz (1952); Michaud (1989); DeMiguel et al. (2009)")
        # forecasting and research
        add("intelligence.forecast", self._intel_forecast, "Point forecast by a validated baseline or registered model")
        add("intelligence.backtest", self._intel_backtest, "Walk-forward backtest of a forecasting method")
        add("intelligence.assess_valuation", self._intel_assess, "Research classification of a price against a fair value")
        add("intelligence.models", self._intel_models, "Registered forecasting models and their validation manifests")
        # reporting and local memory
        add("report.render", self._report_render, "Render a remembered or supplied result as json/markdown/html/sldc")
        add("memory.recall", self._mem_recall, "Recall one remembered analysis result")
        add("memory.recent", self._mem_recent, "Most recent remembered analysis results")
        return registry

    def available_operations(self) -> tuple[dict[str, object], ...]:
        """Enabled operations with their summaries and academic basis, for applications and orchestrators."""

        return tuple(self._operations[name].describe() for name in sorted(self.allowed_operations))

    # ------------------------------------------------------------------------------------------------------------
    # Request normalization
    # ------------------------------------------------------------------------------------------------------------

    def _normalize_request(self, request: FinancialRequest | Mapping[str, object]) -> FinancialRequest:
        if isinstance(request, FinancialRequest):
            return request
        if not isinstance(request, Mapping):
            raise FinancialValidationError("FinancialAgent request must be a mapping or FinancialRequest")
        unknown = sorted(str(key) for key in request if key not in self._TASK_KEYS)
        if unknown:
            raise FinancialValidationError(
                "unknown FinancialAgent request fields",
                context={"unknown": unknown, "allowed": sorted(self._TASK_KEYS)},
            )
        operation = request.get("operation")
        domain = request.get("domain")
        if operation is None:
            raise FinancialValidationError("FinancialAgent request requires operation")
        operation_text = require_text(operation, "operation")
        if "." in operation_text:
            qualified_domain, operation_text = operation_text.split(".", 1)
            if domain is not None and str(domain).strip().lower() != qualified_domain.strip().lower():
                raise FinancialValidationError(
                    "request domain contradicts the qualified operation name",
                    context={"domain": str(domain), "operation": str(operation)},
                )
            domain = qualified_domain
        if domain is None:
            raise FinancialValidationError("FinancialAgent request requires domain (or a qualified 'domain.operation')")
        payload: dict[str, Any] = {
            "request_id": request.get("request_id") or f"fin-{uuid.uuid4().hex[:20]}",
            "domain": str(domain).strip().lower(),
            "operation": operation_text.strip().lower(),
            "parameters": request.get("parameters") or {},
            "metadata": request.get("metadata") or {},
        }
        if request.get("created_at") is not None:
            payload["created_at"] = request["created_at"]
        if not isinstance(payload["parameters"], Mapping) or not isinstance(payload["metadata"], Mapping):
            raise FinancialValidationError("parameters and metadata must be mappings")
        return FinancialRequest.from_dict(payload)

    def _resolve(self, request: FinancialRequest) -> FinancialOperation:
        name = f"{request.domain}.{request.operation}"
        spec = self._operations.get(name)
        if spec is None or name not in self.allowed_operations:
            raise UnsupportedFinancialOperationError(
                "operation is not available on this FinancialAgent",
                context={"operation": name, "available": sorted(self.allowed_operations)},
            )
        return spec

    # ------------------------------------------------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------------------------------------------------

    def analyze(self, request: FinancialRequest | Mapping[str, object]) -> FinancialResult:
        """Run one request. Expected domain failures yield a non-success ``FinancialResult``; defects propagate."""

        if not self.enabled:
            raise BaseConfigurationError("FinancialAgent is disabled by configuration", component=self.name, operation="analyze")
        started = monotonic_seconds()
        request_id: str | None = None
        try:
            normalized = self._normalize_request(request)
            request_id = normalized.request_id
        except FinancialError as exc:
            # Not even a request: there is nothing to attribute the failure to.
            placeholder = self._placeholder_request(request)
            rejected = self._failure_result(placeholder, exc, monotonic_seconds() - started)
            self._record_metrics(placeholder, None, rejected, monotonic_seconds() - started)
            return rejected
        spec: FinancialOperation | None = None
        authorization: dict[str, object] = {"checked": False}
        logger.info("Financial request received | id=%s | operation=%s.%s", normalized.request_id, normalized.domain, normalized.operation)
        try:
            spec = self._resolve(normalized)
            self._shared_set(
                self._key(self.request_key_prefix, normalized.request_id),
                self._request_record(normalized),
                tags=("financial", "request", spec.name),
            )
            authorization = self._authorize(spec, normalized)
            params = _Params(normalized.parameters, item_limit=self.max_parameter_items, request_id=normalized.request_id)
            produced = spec.handler(params)
            leftover = params.unused()
            if leftover:
                raise FinancialValidationError(
                    "unknown parameters for this operation",
                    context={"operation": spec.name, "unknown": leftover},
                )
            outcome = produced if isinstance(produced, _Outcome) else _Outcome(produced)
            result = self._success_result(normalized, spec, outcome, authorization)
        except FinancialError as exc:
            result = self._failure_result(normalized, exc, monotonic_seconds() - started, authorization=authorization)
        except Exception as exc:
            self._record_metrics(normalized, spec, None, monotonic_seconds() - started, error=exc)
            self._publish_defect(normalized, exc)
            logger.exception("FinancialAgent defect | id=%s | operation=%s.%s", normalized.request_id, normalized.domain, normalized.operation)
            raise
        elapsed = monotonic_seconds() - started
        self._after_analysis(normalized, spec, result, elapsed)
        return result

    @staticmethod
    def _placeholder_request(raw: object) -> FinancialRequest:
        request_id = f"fin-{uuid.uuid4().hex[:20]}"
        if isinstance(raw, Mapping):
            candidate = raw.get("request_id")
            if isinstance(candidate, str) and candidate.strip():
                try:
                    request_id = normalize_identifier(candidate, "request_id")
                except FinancialError:
                    pass
        return FinancialRequest(request_id, "agent", "invalid_request")

    def _success_result(self, request: FinancialRequest, spec: FinancialOperation, outcome: _Outcome,
                        authorization: Mapping[str, object]) -> FinancialResult:
        diagnostics: dict[str, Any] = {
            "agent": self.name,
            "agent_version": __version__,
            "operation": spec.name,
            "authorization": dict(authorization),
        }
        diagnostics.update(outcome.diagnostics)
        return FinancialResult(
            request_id=request.request_id,
            domain=request.domain,
            operation=request.operation,
            status=outcome.status,
            result=to_json_compatible(outcome.value),
            assumptions=to_json_compatible(dict(outcome.assumptions)),
            diagnostics=to_json_compatible(diagnostics),
            sources=_gather_sources(outcome.value),
            warnings=tuple(outcome.warnings),
            methodology=outcome.methodology or spec.basis,
            model_version=outcome.model_version,
            metadata=to_json_compatible(dict(request.metadata)),
        )

    def _failure_result(self, request: FinancialRequest, error: FinancialError, elapsed: float, *,
                        authorization: Mapping[str, object] | None = None) -> FinancialResult:
        code = getattr(error, "code", None)
        diagnostics: dict[str, Any] = {
            "agent": self.name,
            "agent_version": __version__,
            "error_type": type(error).__name__,
            "error_code": str(code) if code else None,
            "error_category": str(getattr(error, "category", "") or "") or None,
            "retryable": bool(getattr(error, "retryable", False)),
            "message": str(getattr(error, "message", None) or error)[:1000],
        }
        if authorization:
            diagnostics["authorization"] = dict(authorization)
        context = getattr(error, "context", None)
        if isinstance(context, Mapping) and context:
            diagnostics["error_context"] = to_json_safe(redact_mapping(dict(context)), max_depth=6, max_items=64, max_string_length=512)
        try:
            error.log()
        except Exception:  # logging must never mask the outcome
            pass
        result = FinancialResult(
            request_id=request.request_id,
            domain=request.domain,
            operation=request.operation,
            status=_failure_status(error),
            result=None,
            diagnostics=to_json_compatible(diagnostics),
            warnings=(),
            metadata=to_json_compatible(dict(request.metadata)),
        )
        if request.domain != "agent":  # a request that never parsed has no operation to count
            logger.warning("Financial request failed | id=%s | operation=%s.%s | error=%s", request.request_id,
                           request.domain, request.operation, type(error).__name__)
        return result

    def _after_analysis(self, request: FinancialRequest, spec: FinancialOperation | None, result: FinancialResult, elapsed: float) -> None:
        self._record_metrics(request, spec, result, elapsed)
        succeeded = result.status is FinancialResultStatus.SUCCESS
        if self.remember_results and (succeeded or self.remember_failures) and spec is not None and spec.domain not in ("memory", "report"):
            try:
                self.memory.remember(result)
            except FinancialError as exc:  # local memory is an aid; it must not turn a valid analysis into a failure
                logger.warning("Financial memory could not retain a result | id=%s | error=%s", request.request_id, type(exc).__name__)
        envelope = self._envelope(result, elapsed)
        self._publish_result(request, result, envelope, elapsed)

    # ------------------------------------------------------------------------------------------------------------
    # BaseAgent entry points
    # ------------------------------------------------------------------------------------------------------------

    def _envelope(self, result: FinancialResult, elapsed: float) -> dict[str, object]:
        data = result.to_dict()
        succeeded = result.status is FinancialResultStatus.SUCCESS
        data["success"] = succeeded
        data["agent"] = self.name
        data["elapsed_seconds"] = round(elapsed, 6)
        data["fingerprint"] = stable_fingerprint(
            {"operation": f"{result.domain}.{result.operation}", "status": data["status"], "result": data["result"]}
        )
        if not succeeded:
            diagnostics = data.get("diagnostics")
            diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
            data["error"] = {
                key: diagnostics.get(key)
                for key in ("error_type", "error_code", "error_category", "retryable", "message")
            }
        return data

    def execute_request(self, request: FinancialRequest | Mapping[str, object]) -> dict[str, object]:
        """Run one request and return the JSON envelope (the result plus ``success``, timing and fingerprint)."""

        started = monotonic_seconds()
        result = self.analyze(request)
        return self._envelope(result, monotonic_seconds() - started)

    def perform_task(self, task_data: object) -> dict[str, object]:
        """BaseAgent task entry point: one request, or ``{"tasks": [...]}`` for an independent batch."""

        started = monotonic_seconds()
        if isinstance(task_data, Mapping) and "tasks" in task_data:
            try:
                return self._perform_batch(task_data)
            except FinancialError as exc:  # a malformed batch is an expected failure, like a malformed request
                rejected = self._failure_result(self._placeholder_request(task_data), exc, monotonic_seconds() - started)
                return self._envelope(rejected, monotonic_seconds() - started)
        result = self.analyze(task_data)  # type: ignore[arg-type]
        return self._envelope(result, monotonic_seconds() - started)

    def _perform_batch(self, task_data: Mapping[str, object]) -> dict[str, object]:
        extra = sorted(str(key) for key in task_data if key != "tasks")
        if extra:
            raise FinancialValidationError("a batch request holds only 'tasks'", context={"unexpected": extra})
        tasks = task_data["tasks"]
        if isinstance(tasks, (str, bytes, bytearray)) or not isinstance(tasks, Sequence) or not 1 <= len(tasks) <= self.max_batch_size:
            raise FinancialValidationError(
                "tasks must be a sequence within max_batch_size",
                context={"max_batch_size": self.max_batch_size, "size": len(tasks) if isinstance(tasks, Sequence) else None},
            )
        started = monotonic_seconds()
        envelopes = [self._batch_item(item) for item in tasks]
        successes = sum(1 for item in envelopes if item.get("success") is True)
        status = "success" if successes == len(envelopes) else ("partial" if successes else "failed")
        return {
            "status": status,
            "success": status == "success",
            "agent": self.name,
            "batch_size": len(envelopes),
            "succeeded": successes,
            "results": envelopes,
            "elapsed_seconds": round(monotonic_seconds() - started, 6),
        }

    def _batch_item(self, item: object) -> dict[str, object]:
        """One independent batch member; a member's failure never aborts its siblings."""

        started = monotonic_seconds()
        try:
            if isinstance(item, Mapping) and "tasks" in item:
                raise FinancialValidationError("batches cannot be nested")
            return self.perform_task(item)
        except FinancialError as exc:
            result = self._failure_result(self._placeholder_request(item), exc, monotonic_seconds() - started)
            return self._envelope(result, monotonic_seconds() - started)
        except Exception as exc:  # the defect was already logged and published by analyze(); report it per member
            return {
                "status": "failed",
                "success": False,
                "agent": self.name,
                "error": {"error_type": type(exc).__name__, "message": str(exc)[:1000]},
                "elapsed_seconds": round(monotonic_seconds() - started, 6),
            }

    def predict(self, state: object, context: object = None) -> dict[str, object]:
        """Compatibility route for predict-style dispatch."""

        if not isinstance(state, Mapping):
            raise FinancialValidationError("FinancialAgent.predict requires a request mapping")
        return self.perform_task(dict(state))

    def act(self, task_data: object, context: object = None) -> dict[str, object]:
        """Compatibility route for action-style dispatch."""

        return self.predict(task_data, context=context)

    def alternative_execute(self, task_data: object, original_error: BaseException | None = None) -> dict[str, object]:
        """Decline BaseAgent's generic text fallback: a financial result must never be improvised or echoed."""

        return {
            "status": "failed",
            "success": False,
            "reason": "FinancialAgent does not synthesize fallback financial results.",
            "error_type": type(original_error).__name__ if original_error is not None else None,
            "error": str(original_error) if original_error is not None else None,
        }

    # ------------------------------------------------------------------------------------------------------------
    # Handlers: time value of money
    # ------------------------------------------------------------------------------------------------------------

    def _calc_year_fraction(self, p: _Params) -> Any:
        return {"year_fraction": self.calculation.year_fraction(p.date("start"), p.date("end"), p.text("convention"))}

    def _calc_discount_factor(self, p: _Params) -> Any:
        return {"discount_factor": self.calculation.discount_factor(p.rate("rate"), p.date("start"), p.date("end"))}

    def _calc_present_value(self, p: _Params) -> Any:
        return {"present_value": self.calculation.present_value(p.amount("future"), p.rate("rate"), p.date("start"), p.date("end"))}

    def _calc_future_value(self, p: _Params) -> Any:
        return {"future_value": self.calculation.future_value(p.amount("present"), p.rate("rate"), p.date("start"), p.date("end"))}

    def _calc_net_present_value(self, p: _Params) -> Any:
        return {"net_present_value": self.calculation.net_present_value(p.flows(), p.rate("rate"), p.date("valuation_date"))}

    def _calc_periodic_rate(self, p: _Params) -> Any:
        return {"periodic_rate": self.calculation.periodic_rate(p.rate("rate"), p.integer("periods_per_year", minimum=1))}

    def _calc_loan_payment(self, p: _Params) -> Any:
        payment = self.calculation.loan_payment(
            p.amount("principal"), p.rate("rate"), p.integer("periods", minimum=1), p.integer("periods_per_year", minimum=1)
        )
        return {"payment": payment}

    def _calc_amortization_schedule(self, p: _Params) -> Any:
        schedule = self.calculation.amortization_schedule(
            p.amount("principal"), p.rate("rate"), p.integer("periods", minimum=1), p.integer("periods_per_year", minimum=1)
        )
        return {"installments": schedule}

    def _calc_xirr(self, p: _Params) -> Any:
        return {"xirr": self.calculation.xirr(p.flows(), valuation_date=p.opt_date("valuation_date"))}

    # ------------------------------------------------------------------------------------------------------------
    # Handlers: cash flows
    # ------------------------------------------------------------------------------------------------------------

    def _cf_recurring_schedule(self, p: _Params) -> Any:
        schedule = self.cashflow.recurring_schedule(
            p.amount("amount"),
            p.date("first_date"),
            interval_unit=p.text("interval_unit"),
            interval_count=p.opt_integer("interval_count", minimum=1) or 1,
            end_exclusive=p.date("end_exclusive"),
            start_inclusive=p.opt_date("start_inclusive"),
            reference=p.opt_text("reference"),
            category=p.opt_text("category"),
            preserve_end_of_month=p.opt_boolean("preserve_end_of_month"),
        )
        return {"cash_flows": schedule}

    def _cf_summarize(self, p: _Params) -> Any:
        summary = self.cashflow.summarize(p.flows(), start=p.date("start"), end_exclusive=p.date("end_exclusive"), currency=p.opt_text("currency"))
        return {"summary": summary}

    def _cf_monthly_totals(self, p: _Params) -> Any:
        return {"monthly_totals": self.cashflow.monthly_totals(p.flows(), currency=p.opt_text("currency"))}

    def _cf_running_balances(self, p: _Params) -> Any:
        balances = self.cashflow.running_balances(p.flows(), p.amount("opening_balance"), as_of=p.date("as_of"))
        return {"balances": [{"date": day, "balance": balance} for day, balance in balances]}

    def _cf_discounted_value(self, p: _Params) -> Any:
        return {"present_value": self.cashflow.discounted_value(p.flows(), p.rate("rate"), valuation_date=p.date("valuation_date"))}

    # ------------------------------------------------------------------------------------------------------------
    # Handlers: valuation and historical risk
    # ------------------------------------------------------------------------------------------------------------

    def _val_gordon(self, p: _Params) -> Any:
        value = self.valuation.gordon_terminal_value(
            p.amount("next_annual_cashflow"),
            discount_rate=p.rate("discount_rate"),
            annual_growth=p.decimal("annual_growth"),
            terminal_date=p.date("terminal_date"),
            valuation_date=p.date("valuation_date"),
        )
        return {"terminal_value": value}

    def _val_dcf(self, p: _Params) -> Any:
        valuation = self.valuation.discounted_cashflow(
            p.flows(),
            discount_rate=p.rate("discount_rate"),
            valuation_date=p.date("valuation_date"),
            terminal_next_annual_cashflow=p.opt_amount("terminal_next_annual_cashflow"),
            terminal_date=p.opt_date("terminal_date"),
            terminal_growth_rate=p.opt_decimal("terminal_growth_rate"),
        )
        return {"valuation": valuation}

    def _val_scenarios(self, p: _Params) -> Any:
        scenarios = self.valuation.scenario_valuations(
            p.flows(), valuation_date=p.date("valuation_date"), scenarios=p.build_many("scenarios", _scenario)
        )
        return {"scenarios": scenarios}

    def _val_var_es(self, p: _Params) -> Any:
        risk = self.valuation.historical_var_es(p.decimals("returns"), exposure=p.amount("exposure"), confidence=p.opt_decimal("confidence"))
        return _Outcome(
            {"risk": risk},
            warnings=("Historical VaR/ES describe the supplied sample only; they are not forecasts of future losses.",),
        )

    def _val_drawdown(self, p: _Params) -> Any:
        return {"drawdown": self.valuation.maximum_drawdown(p.decimals("returns"))}

    def _val_stress(self, p: _Params) -> Any:
        return {"stressed_value": self.valuation.stress_value(p.amount("value"), fractional_price_shock=p.decimal("fractional_price_shock"))}

    # ------------------------------------------------------------------------------------------------------------
    # Handlers: macro-economic and accounting analysis
    # ------------------------------------------------------------------------------------------------------------

    def _eco_inflation_rate(self, p: _Params) -> Any:
        return {"inflation_rate": self.economics.inflation_rate(p.decimal("earlier_index"), p.decimal("later_index"))}

    def _eco_annualized_inflation(self, p: _Params) -> Any:
        day_count = p.opt_text("day_count")
        if day_count is None:
            day_count = DayCountConvention.ACT_365F.value
        rate = self.economics.annualized_inflation_rate(
            p.decimal("earlier_index"),
            p.decimal("later_index"),
            start=p.date("start"),
            end=p.date("end"),
            day_count=day_count,
        )
        return {"annualized_inflation_rate": rate}

    def _eco_adjust(self, p: _Params) -> Any:
        adjusted = self.economics.adjust_purchasing_power(p.amount("amount"), from_index=p.decimal("from_index"), to_index=p.decimal("to_index"))
        return {"adjusted_amount": adjusted}

    def _eco_real_rate(self, p: _Params) -> Any:
        return {"real_rate": self.economics.real_effective_rate(p.rate("nominal"), p.decimal("inflation"))}

    def _eco_price_return(self, p: _Params) -> Any:
        return {"price_return": self.economics.price_return(p.build("earlier", _observation), p.build("later", _observation))}

    def _accounting_snapshot(self, p: _Params) -> AccountingSnapshot:
        spec = p.mapping("snapshot")
        unknown = sorted(set(spec) - {"as_of", "currency", "items", "period", "basis", "source_reference"})
        if unknown:
            raise FinancialValidationError("unknown accounting snapshot fields", context={"parameter": "snapshot", "unknown": unknown})

        def build(_: Any) -> AccountingSnapshot:
            currency = require_text(spec["currency"], "currency")
            items = _mapping_of(spec["items"])
            period = spec.get("period")
            return AccountingSnapshot(
                as_of=to_date(spec["as_of"], "as_of"),
                currency=currency,
                # A bare number takes the snapshot's explicitly declared currency; a mapping carries its own.
                items={
                    require_text(name, "item"): (_amount(value) if isinstance(value, Mapping) else FinancialAmount(to_decimal(value, name), currency))
                    for name, value in items.items()
                },
                period=None if period is None else _period(period),
                basis=spec.get("basis", "reported"),
                source_reference=spec.get("source_reference"),
            )

        return _decode(build, spec, "snapshot")

    def _eco_ratios(self, p: _Params) -> Any:
        return {"analysis": self.economics.accounting_ratios(self._accounting_snapshot(p))}

    def _eco_reconcile(self, p: _Params) -> Any:
        snapshot = self._accounting_snapshot(p)
        return {"reconciliation": self.economics.reconcile_balance_sheet(snapshot, tolerance=p.opt_decimal("tolerance", Decimal(0)))}

    def _eco_series(self, p: _Params) -> Any:
        series = self.economics.economic_series(
            p.text("indicator_id"), p.text("field_name"),
            from_time=p.moment("from_time"), until_time=p.moment("until_time"), as_of=p.moment("as_of"),
            vintage=p.opt_text("vintage", "latest") or "latest",
        )
        return {"observations": series}

    def _eco_index_level(self, p: _Params) -> Any:
        level = self.economics.index_level(
            p.text("indicator_id"), p.text("field_name"), on=p.moment("on"), as_of=p.moment("as_of"), max_age=_required_age(p)
        )
        return {"observation": level}

    def _eco_cumulative_inflation(self, p: _Params) -> Any:
        value = self.economics.cumulative_inflation(
            p.text("indicator_id"), p.text("field_name"),
            start=p.moment("start"), end=p.moment("end"), as_of=p.moment("as_of"), max_age=_required_age(p),
        )
        return {"cumulative_inflation": value}

    # ------------------------------------------------------------------------------------------------------------
    # Handlers: point-in-time market data
    # ------------------------------------------------------------------------------------------------------------

    def _reject_unused_before_mutation(self, p: _Params, operation: str) -> None:
        leftover = p.unused()
        if leftover:
            raise FinancialValidationError(
                "unknown parameters for this operation; nothing was changed",
                context={"operation": operation, "unknown": leftover},
            )

    def _data_ingest_observations(self, p: _Params) -> Any:
        observations = p.observations()
        self._reject_unused_before_mutation(p, "data.ingest_observations")
        return {"ingested": self.data.ingest_observations(observations), "counts": self.data.counts()}

    def _data_ingest_rates(self, p: _Params) -> Any:
        rates = p.build_many("exchange_rates", _exchange_rate)
        self._reject_unused_before_mutation(p, "data.ingest_exchange_rates")
        return {"ingested": self.data.ingest_exchange_rates(rates), "counts": self.data.counts()}

    def _data_ingest_shared(self, p: _Params) -> Any:
        name = normalize_identifier(p.text("name"), "name")
        self._reject_unused_before_mutation(p, "data.ingest_shared")
        key = f"{self.input_key_prefix}:{name}"
        try:
            payload = self.shared_memory.get(key)
        except Exception as exc:
            raise FinancialDelegationError("SharedMemory input could not be read", context={"key": key}, cause=exc) from exc
        if payload is None:
            raise MissingMarketDataError("no shared-memory input has been published under this name", context={"key": key})
        if not isinstance(payload, Mapping) or not ({"observations", "exchange_rates"} & set(payload)) or set(payload) - {"observations", "exchange_rates"}:
            raise FinancialValidationError(
                "shared input must be a mapping holding observations and/or exchange_rates", context={"key": key}
            )
        inner = _Params(payload, item_limit=self.max_parameter_items)
        observations = inner.build_many("observations", _observation, minimum=0) if payload.get("observations") is not None else ()
        rates = inner.build_many("exchange_rates", _exchange_rate, minimum=0) if payload.get("exchange_rates") is not None else ()
        # Both lists are validated before either is stored, so a bad quote cannot leave a half-ingested batch.
        stored_observations = self.data.ingest_observations(observations) if observations else 0
        stored_rates = self.data.ingest_exchange_rates(rates) if rates else 0
        return _Outcome(
            {"ingested_observations": stored_observations, "ingested_exchange_rates": stored_rates, "counts": self.data.counts()},
            diagnostics={"shared_key": key},
        )

    def _data_latest_observation(self, p: _Params) -> Any:
        found = self.data.latest_observation(
            p.text("instrument_id"), p.text("field_name"),
            as_of=p.moment("as_of"), max_age=p.max_age(), source_id=p.opt_text("source_id"),
        )
        return {"observation": found}

    def _data_latest_rate(self, p: _Params) -> Any:
        found = self.data.latest_exchange_rate(
            p.text("base_currency"), p.text("quote_currency"),
            as_of=p.moment("as_of"), max_age=p.max_age(), source_id=p.opt_text("source_id"),
        )
        return {"exchange_rate": found}

    def _data_history(self, p: _Params) -> Any:
        history = self.data.observation_history(
            p.text("instrument_id"), p.text("field_name"),
            from_time=p.moment("from_time"), until_time=p.moment("until_time"), as_of=p.moment("as_of"),
            source_id=p.opt_text("source_id"), vintage=p.opt_text("vintage") or "all",
        )
        return {"observations": history}

    def _data_convert(self, p: _Params) -> Any:
        converter = self.data.fx_converter(precision=self.calculation.precision)
        conversion = converter.convert(p.amount("amount"), p.text("to_currency"), as_of=p.moment("as_of"), policy=p.fx_policy())
        return {"conversion": conversion}

    def _data_counts(self, p: _Params) -> Any:
        return {"counts": self.data.counts()}

    # ------------------------------------------------------------------------------------------------------------
    # Handlers: subscriptions and recurring revenue
    # ------------------------------------------------------------------------------------------------------------

    def _sub_invoice(self, p: _Params) -> Any:
        return {"invoice": self.subscription.invoice_at(p.build("contract", _contract), p.date("on"))}

    def _sub_projected(self, p: _Params) -> Any:
        charges = self.subscription.projected_charges(
            p.build("contract", _contract), start=p.date("start"), end_exclusive=p.date("end_exclusive")
        )
        return {"charges": charges}

    def _sub_forecast(self, p: _Params) -> Any:
        charges = self.subscription.forecast(
            p.build_many("contracts", _contract), start=p.date("start"), end_exclusive=p.date("end_exclusive")
        )
        return {"charges": charges}

    def _sub_forecast_by_currency(self, p: _Params) -> Any:
        summaries = self.subscription.forecast_by_currency(
            p.build_many("contracts", _contract), start=p.date("start"), end_exclusive=p.date("end_exclusive")
        )
        return {"summaries": summaries}

    def _sub_forecast_in_currency(self, p: _Params) -> Any:
        charges = self.subscription.forecast_in_currency(
            p.build_many("contracts", _contract),
            start=p.date("start"),
            end_exclusive=p.date("end_exclusive"),
            to_currency=p.text("to_currency"),
            as_of=p.moment("as_of"),
            converter=self.data.fx_converter(precision=self.calculation.precision),
            policy=p.fx_policy(),
        )
        return {"charges": charges}

    def _sub_mrr(self, p: _Params) -> Any:
        return {"monthly_recurring_value": self.subscription.monthly_recurring_value(p.build("contract", _contract), as_of=p.date("as_of"))}

    def _sub_arr(self, p: _Params) -> Any:
        return {"annual_recurring_value": self.subscription.annual_recurring_value(p.build("contract", _contract), as_of=p.date("as_of"))}

    def _sub_billing_period(self, p: _Params) -> Any:
        return {"period": self.subscription.billing_period(p.build("contract", _contract), p.date("day"))}

    def _sub_prorated(self, p: _Params) -> Any:
        charge = self.subscription.prorated_charge(
            p.build("contract", _contract), service_start=p.date("service_start"), service_end=p.opt_date("service_end")
        )
        return {"proration": charge}

    def _sub_cancellation(self, p: _Params) -> Any:
        return {"proration": self.subscription.cancellation_credit(p.build("contract", _contract), cancelled_on=p.date("cancelled_on"))}

    def _sub_movement(self, p: _Params) -> Any:
        movement, retention = self.subscription.recurring_value_movement(
            p.build_many("contracts", _contract), opening=p.date("opening"), closing=p.date("closing"), currency=p.text("currency")
        )
        return {"movement": movement, "retention": retention}

    def _sub_clv(self, p: _Params) -> Any:
        value = self.subscription.customer_lifetime_value(
            p.amount("margin_per_period"),
            retention_rate=p.decimal("retention_rate"),
            discount_rate_per_period=p.decimal("discount_rate_per_period"),
            include_current_period=p.boolean("include_current_period", False),
        )
        return _Outcome(
            {"customer_lifetime_value": value},
            assumptions={"horizon": "infinite", "retention": "constant per period"},
            warnings=("CLV assumes a constant retention rate and discount rate over an infinite horizon.",),
        )

    # ------------------------------------------------------------------------------------------------------------
    # Handlers: portfolio valuation and performance
    # ------------------------------------------------------------------------------------------------------------

    def _snapshot(self, p: _Params) -> Any:
        return self.portfolio.value_positions(p.build_many("holdings", _holding), as_of=p.moment("as_of"), max_age=p.max_age())

    def _pf_value_positions(self, p: _Params) -> Any:
        return {"snapshot": self._snapshot(p)}

    def _pf_weights(self, p: _Params) -> Any:
        return {"weights": self.portfolio.allocation_weights(self._snapshot(p))}

    def _targets(self, p: _Params, snapshot: Any) -> Any:
        explicit, equal = p.opt_mapping("target_weights"), p.boolean("equal_weight", False)
        if (explicit is None) == (not equal):
            raise FinancialValidationError("provide exactly one of target_weights or equal_weight=true")
        return self.portfolio.equal_weight_targets(snapshot) if equal else explicit

    def _pf_rebalance(self, p: _Params) -> Any:
        snapshot = self._snapshot(p)
        allocation = self.portfolio.rebalance_amounts(snapshot, self._targets(p, snapshot), rounding=p.rounding())
        return _Outcome({"allocation": allocation}, warnings=("Value differences only: no lot sizes, costs, taxes or settlement are modelled and nothing is traded.",))

    def _pf_rebalance_cost(self, p: _Params) -> Any:
        snapshot = self._snapshot(p)
        rounding = p.rounding()
        allocation = self.portfolio.rebalance_amounts(snapshot, self._targets(p, snapshot), rounding=rounding)
        cost = self.portfolio.rebalance_cost(allocation, p.build_many("fees", _charge), rounding=rounding)
        return {"allocation": allocation, "cost": cost}

    def _pf_value_in_currency(self, p: _Params) -> Any:
        snapshot = self._snapshot(p)
        total, conversions = self.portfolio.convert_snapshot(snapshot, reporting_currency=p.text("reporting_currency"), policy=p.fx_policy())
        return {"total": total, "conversions": conversions}

    def _pf_moments(self, p: _Params) -> Any:
        moments = self.portfolio.return_moments(
            p.strings("instrument_ids"),
            from_time=p.moment("from_time"), until_time=p.moment("until_time"), as_of=p.moment("as_of"),
            periods_per_year=p.integer("periods_per_year", minimum=1),
            field_name=p.opt_text("field_name", "close"), kind=p.opt_text("kind", "simple"), source_id=p.opt_text("source_id"),
        )
        warnings = ["Sample moments are noisy estimates (Michaud 1989)."]
        if not moments.covariance_full_rank_possible:
            warnings.append("Fewer observations than instruments: the sample covariance matrix is singular.")
        return _Outcome({"moments": moments}, warnings=tuple(warnings))

    def _pf_return_statistics(self, p: _Params) -> Any:
        statistics = self.portfolio.return_statistics(
            p.decimals("returns"), periods_per_year=p.integer("periods_per_year", minimum=1),
            risk_free_per_period=p.opt_decimal("risk_free_per_period", Decimal(0)),
        )
        return _Outcome({"statistics": statistics}, warnings=("Sharpe ratios estimated from short samples carry wide sampling error (Lo 2002).",))

    def _pf_relative_statistics(self, p: _Params) -> Any:
        statistics = self.portfolio.relative_statistics(
            p.decimals("returns"), p.decimals("benchmark"), periods_per_year=p.integer("periods_per_year", minimum=1),
            risk_free_per_period=p.opt_decimal("risk_free_per_period", Decimal(0)),
        )
        return {"statistics": statistics}

    def _pf_annualize(self, p: _Params) -> Any:
        value = self.portfolio.annualize_return(
            p.decimal("total_return"), p.decimal("years"), allow_short_period=p.boolean("allow_short_period", False)
        )
        return {"annualized_return": value}

    def _pf_twr(self, p: _Params) -> Any:
        return {"result": self.portfolio.time_weighted_return(p.build_many("valuations", _valuation_point, minimum=2), p.opt_build_many("flows", _external_flow))}

    def _pf_dietz(self, p: _Params) -> Any:
        value = self.portfolio.modified_dietz_return(p.build("begin", _valuation_point), p.build("end", _valuation_point), p.opt_build_many("flows", _external_flow))
        return {"modified_dietz_return": value}

    def _pf_mwr(self, p: _Params) -> Any:
        rate = self.portfolio.money_weighted_return(p.build("begin", _valuation_point), p.build("end", _valuation_point), p.opt_build_many("flows", _external_flow))
        return {"money_weighted_return": rate}

    # -- mean-variance optimization, delegated -----------------------------------------------------------------

    def _pf_optimize(self, p: _Params) -> Any:
        """Mean-variance weights: the subsystem estimates moments and validates; OptimizationAgent solves.

        minimize  lambda * w'Sigma w - mu'w   subject to  sum(w) = 1,  min_weight <= w_i <= max_weight
        The solver works in binary floating point; the returned weights are re-quantized to Decimal, checked against
        the constraints and made to sum to exactly one before any value arithmetic is done with them.
        """

        ids = p.strings("instrument_ids", minimum=2)
        from_time, until_time, as_of = p.moment("from_time"), p.moment("until_time"), p.moment("as_of")
        periods = p.integer("periods_per_year", minimum=1)
        field_name, kind, source_id = p.opt_text("field_name", "close"), p.opt_text("kind", "simple"), p.opt_text("source_id")
        risk_aversion = p.opt_decimal("risk_aversion", _ONE)
        lower, upper = p.opt_decimal("min_weight", Decimal(0)), p.opt_decimal("max_weight", _ONE)
        solver = p.opt_text("solver")
        holdings = p.opt_build_many("holdings", _holding)
        max_age, rounding = p.max_age(), p.rounding()
        n = len(ids)
        if risk_aversion is None or risk_aversion <= 0:
            raise FinancialValidationError("risk_aversion must be positive", context={"parameter": "risk_aversion"})
        if lower is None or upper is None or lower > upper or n * upper < _ONE or n * lower > _ONE:
            raise FinancialValidationError(
                "min_weight/max_weight admit no fully invested portfolio",
                context={"min_weight": str(lower), "max_weight": str(upper), "instruments": n},
            )
        if holdings and {h.instrument.instrument_id for h in holdings} != set(ids):
            raise FinancialValidationError("holdings must cover exactly the instruments being optimized")

        moments = self.portfolio.return_moments(
            ids, from_time=from_time, until_time=until_time, as_of=as_of, periods_per_year=periods,
            field_name=field_name, kind=kind, source_id=source_id,
        )
        weights, solution = self._solve_mean_variance(moments, risk_aversion, lower, upper, solver, p.request_id)

        value: dict[str, Any] = {
            "weights": weights,
            "risk_aversion": risk_aversion,
            "observations": moments.observations,
            "solver": {key: solution.get(key) for key in ("solver", "solver_status", "termination_reason", "feasibility", "optimality")},
        }
        value.update(self._portfolio_statistics(moments, weights, risk_aversion))
        with localcontext() as ctx:
            ctx.prec = self.calculation.precision
            equal = {asset: _ONE / n for asset in moments.asset_ids}
        value["equal_weight_benchmark"] = self._portfolio_statistics(moments, equal, risk_aversion)
        if holdings:
            snapshot = self.portfolio.value_positions(holdings, as_of=as_of, max_age=max_age)
            value["rebalance"] = self.portfolio.rebalance_amounts(snapshot, weights, rounding=rounding)
        warnings = [
            "Sample mean-variance weights are sensitive to estimation error (Michaud 1989); "
            "the equal-weight benchmark is reported because it is often hard to beat out of sample (DeMiguel et al. 2009).",
            "Weights are advisory analysis outputs: no order is created or executed.",
        ]
        if not moments.covariance_full_rank_possible:
            warnings.append("Fewer observations than instruments: the sample covariance matrix is singular.")
        warnings.extend(str(item) for item in (solution.get("warnings") or ()))
        return _Outcome(
            value,
            assumptions={
                "objective": "minimize risk_aversion * w'Sigma w - mu'w",
                "constraints": {"fully_invested": True, "min_weight": str(lower), "max_weight": str(upper)},
                "estimation": "historical sample mean and covariance, annualized",
                "numerics": "solver in binary floating point; weights re-quantized to Decimal",
            },
            warnings=tuple(warnings),
            methodology="Mean-variance optimization (Markowitz 1952), solved by OptimizationAgent",
        )

    def _portfolio_statistics(self, moments: Any, weights: Mapping[str, Decimal], risk_aversion: Decimal) -> dict[str, Any]:
        with localcontext() as ctx:
            ctx.prec = self.calculation.precision
            w = [weights[asset] for asset in moments.asset_ids]
            expected = sum((w[i] * moments.mean[i] for i in range(len(w))), Decimal(0))
            variance = sum((w[i] * w[j] * moments.covariance[i][j] for i in range(len(w)) for j in range(len(w))), Decimal(0))
            return {
                "expected_return": expected,
                "variance": variance,
                "volatility": variance.sqrt() if variance > 0 else Decimal(0),
                "utility": expected - risk_aversion * variance,
            }

    def _solve_mean_variance(self, moments: Any, risk_aversion: Decimal, lower: Decimal, upper: Decimal,
                             solver: str | None, request_id: str) -> tuple[dict[str, Decimal], Mapping[str, Any]]:
        optimizer = self._peer("optimization")
        ids = tuple(moments.asset_ids)
        names = [f"w{index}" for index in range(len(ids))]  # solver-safe variable names; ids may hold any characters
        lam = float(risk_aversion)
        problem = {
            "name": "mean_variance_allocation",
            "variables": {name: {"kind": "real", "lower": float(lower), "upper": float(upper)} for name in names},
            "objectives": {
                "utility": {
                    "sense": "minimize",
                    "linear": {names[i]: -float(moments.mean[i]) for i in range(len(names))},
                    "quadratic": {f"{names[i]}*{names[j]}": lam * float(moments.covariance[i][j]) for i in range(len(names)) for j in range(len(names))},
                }
            },
            "constraints": [{"name": "fully_invested", "sense": "==", "rhs": 1.0, "linear": {name: 1.0 for name in names}}],
        }
        request: dict[str, Any] = {"request_id": f"{request_id}-mean-variance"[:128], "problem": problem, "publish": False}
        if solver is not None:
            request["solver"] = solver
        try:
            response = optimizer.optimize(request)
        except FinancialError:
            raise
        except Exception as exc:
            raise FinancialDelegationError(
                "OptimizationAgent could not solve the mean-variance problem",
                context={"error_type": type(exc).__name__}, cause=exc,
            ) from exc
        solution = response.get("solution") if isinstance(response, Mapping) else None
        if not isinstance(solution, Mapping) or response.get("success") is not True or not isinstance(solution.get("decision_variables"), Mapping):
            raise FinancialDelegationError(
                "OptimizationAgent did not return an optimal feasible solution",
                context={"status": str(response.get("status")) if isinstance(response, Mapping) else None},
            )
        return self._decimal_weights(ids, names, solution["decision_variables"], lower, upper), solution

    def _decimal_weights(self, ids: Sequence[str], names: Sequence[str], raw: Mapping[str, Any], lower: Decimal, upper: Decimal) -> dict[str, Decimal]:
        quantum, tolerance = Decimal("1e-12"), Decimal("1e-7")
        with localcontext() as ctx:
            ctx.prec = self.calculation.precision
            values: list[Decimal] = []
            for name in names:
                number = raw.get(name)
                if isinstance(number, bool) or not isinstance(number, (int, float)) or number != number or abs(number) == float("inf"):
                    raise FinancialDelegationError("OptimizationAgent returned a non-finite weight", context={"variable": name})
                weight = Decimal(repr(float(number))).quantize(quantum)
                if weight < lower - tolerance or weight > upper + tolerance:
                    raise FinancialDelegationError("OptimizationAgent returned a weight outside the requested bounds", context={"variable": name})
                values.append(min(max(weight, lower), upper))
            residual = _ONE - sum(values, Decimal(0))
            if abs(residual) > Decimal("1e-6"):
                raise FinancialDelegationError("OptimizationAgent weights do not sum to one", context={"residual": str(residual)})
            if residual:
                # Rounding dust goes to the weight with the most room in the needed direction; bounds stay respected.
                room = [(upper - v) if residual > 0 else (v - lower) for v in values]
                index = max(range(len(values)), key=room.__getitem__)
                adjusted = values[index] + residual
                if adjusted < lower or adjusted > upper:
                    raise FinancialDelegationError("weights cannot be made to sum to one within the bounds")
                values[index] = adjusted
            if sum(values, Decimal(0)) != _ONE:
                raise FinancialDelegationError("weights could not be normalized to exactly one")
            return dict(zip(ids, values))

    # ------------------------------------------------------------------------------------------------------------
    # Handlers: forecasting and research
    # ------------------------------------------------------------------------------------------------------------

    def _forecast_inputs(self, p: _Params) -> tuple[tuple[MarketObservation, ...], datetime]:
        """Inline observations, or a point-in-time query of the local store (latest vintage by default)."""

        as_of = p.moment("as_of")
        if p.has("observations"):
            return p.observations(), as_of
        history = self.data.observation_history(
            p.text("instrument_id"), p.text("field_name"),
            from_time=p.moment("from_time"), until_time=p.moment("until_time"), as_of=as_of,
            source_id=p.opt_text("source_id"), vintage=p.opt_text("vintage", "latest"),
        )
        return history, as_of

    def _intel_forecast(self, p: _Params) -> Any:
        history, as_of = self._forecast_inputs(p)
        forecast = self.intelligence.forecast(
            history, as_of=as_of, horizon_steps=p.integer("horizon_steps", minimum=1),
            method=p.opt_text("method", "naive"), seasonal_period=p.opt_integer("seasonal_period", minimum=1),
            model_name=p.opt_text("model_name"),
        )
        return _Outcome(
            {"forecast": forecast},
            assumptions={"method": str(getattr(forecast.method, "value", forecast.method)), "interpretation": forecast.interpretation},
            warnings=("Forecasts are model predictions, not facts; no confidence interval is implied.",),
            model_version=forecast.model_version,
        )

    def _intel_backtest(self, p: _Params) -> Any:
        history, as_of = self._forecast_inputs(p)
        backtest = self.intelligence.walk_forward_backtest(
            history, as_of=as_of, method=p.opt_text("method", "naive"), horizon_steps=p.opt_integer("horizon_steps", 1, minimum=1),
            min_training_points=p.opt_integer("min_training_points", minimum=1), seasonal_period=p.opt_integer("seasonal_period", minimum=1),
        )
        return _Outcome({"backtest": backtest}, warnings=("A backtest evaluates the past sample; it does not guarantee future accuracy.",))

    def _intel_assess(self, p: _Params) -> Any:
        signal = self.intelligence.assess_valuation(
            p.build("price", _observation), fair_value=p.amount("fair_value"), as_of=p.moment("as_of"),
            margin=p.opt_decimal("margin"), max_age=p.max_age(),
        )
        return _Outcome({"signal": signal}, warnings=("A research classification is not investment advice or an order.",))

    def _intel_models(self, p: _Params) -> Any:
        return {"models": self.intelligence.registered_models()}

    def register_forecast_model(self, manifest: FinancialModelManifest, predictor: FinancialPredictor) -> None:
        """Register a validated forecasting model. Code objects cannot travel in a request, so this is API-only."""

        self.intelligence.register_model(manifest, predictor)

    # ------------------------------------------------------------------------------------------------------------
    # Handlers: reporting and local memory
    # ------------------------------------------------------------------------------------------------------------

    def _report_render(self, p: _Params) -> Any:
        fmt = p.opt_text("format", "markdown").lower()  # type: ignore[union-attr]
        renderers = {
            "json": self.report.to_json, "markdown": self.report.to_markdown, "html": self.report.to_html, "sldc": self.report.to_sldc,
        }
        if fmt not in renderers:
            raise FinancialValidationError("unsupported report format", context={"format": fmt, "supported": sorted(renderers)})
        title = p.opt_text("title")
        if p.has("result") == p.has("request_id"):
            raise FinancialValidationError("provide exactly one of request_id (a remembered result) or result")
        if p.has("request_id"):
            stored = self.memory.get(p.text("request_id"))
            if stored is None:
                raise FinancialValidationError("no remembered result exists for this request_id")
        else:
            stored = p.build("result", FinancialResult.from_dict)
        document = self.report.from_result(stored, title=title)
        return {"format": fmt, "content": renderers[fmt](document)}

    def _mem_recall(self, p: _Params) -> Any:
        stored = self.memory.get(p.text("request_id"))
        return {"found": stored is not None, "result": stored}

    def _mem_recent(self, p: _Params) -> Any:
        return {"results": self.memory.recent(limit=p.opt_integer("limit", minimum=1))}

    # ------------------------------------------------------------------------------------------------------------
    # Delegation to other SLAI agents (optional, lazy, never forced)
    # ------------------------------------------------------------------------------------------------------------

    def _peer(self, name: str) -> Any:
        """The factory's instance of a peer agent, created on first use."""

        with self._peer_lock:
            peer = self._peers.get(name)
            if peer is not None:
                return peer
            create = getattr(self.agent_factory, "create", None)
            if not callable(create):
                raise FinancialDelegationError(f"no agent factory is available to provide the '{name}' agent", context={"peer": name})
            try:
                peer = create(name, self.shared_memory)
            except Exception as exc:
                raise FinancialDelegationError(
                    f"the '{name}' agent is unavailable", context={"peer": name, "error_type": type(exc).__name__}, cause=exc
                ) from exc
            if peer is None:
                raise FinancialDelegationError(f"the '{name}' agent is unavailable", context={"peer": name})
            self._peers[name] = peer
            return peer

    def _authorize(self, spec: FinancialOperation, request: FinancialRequest) -> dict[str, object]:
        """Ask SafetyAgent to approve a state-changing operation; reading and computing need no approval.

        ``if_available``: use SafetyAgent when it can be obtained, otherwise proceed and say so in the result.
        ``required``: refuse when approval cannot be obtained. A reachable SafetyAgent that errors or declines
        always blocks the mutation (fail closed); its verdict is never overridden.
        """

        record: dict[str, object] = {"mode": self.authorization_mode, "required": spec.mutating, "checked": False}
        if not spec.mutating or self.authorization_mode == "disabled":
            return record
        try:
            safety = self._peer("safety")
            validate = getattr(safety, "validate_action", None)
            if not callable(validate):
                raise FinancialDelegationError("the safety agent offers no validate_action", context={"peer": "safety"})
        except FinancialDelegationError as exc:
            if self.authorization_mode == "required":
                raise UnauthorizedFinancialOperationError(
                    "authorization is required but the safety agent is unavailable; nothing was changed",
                    context={"operation": spec.name}, cause=exc,
                ) from exc
            logger.warning("Safety agent unavailable; proceeding without authorization | operation=%s", spec.name)
            record["reason"] = "safety_agent_unavailable"
            return record
        action = {
            "action_name": f"financial.{spec.name}",
            "operation": spec.name,
            "mutates": "financial market-data store",
            "request_id": request.request_id,
            "parameters": self._parameter_preview(request.parameters),
        }
        try:
            verdict = validate(action, {"agent": self.name, "request_id": request.request_id, "domain": "financial"})
        except Exception as exc:
            raise UnauthorizedFinancialOperationError(
                "safety validation failed; nothing was changed", context={"operation": spec.name, "error_type": type(exc).__name__}, cause=exc
            ) from exc
        if not isinstance(verdict, Mapping) or verdict.get("approved") is not True:
            decision = verdict.get("decision") if isinstance(verdict, Mapping) else None
            raise UnauthorizedFinancialOperationError(
                "the safety agent did not approve this operation; nothing was changed",
                context={"operation": spec.name, "decision": str(decision), "risk_level": str(verdict.get("risk_level")) if isinstance(verdict, Mapping) else None},
            )
        record.update({"checked": True, "decision": verdict.get("decision"), "risk_score": verdict.get("risk_score"), "validation_id": verdict.get("validation_id")})
        return record

    # ------------------------------------------------------------------------------------------------------------
    # SharedMemory coordination
    # ------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _key(prefix: str, request_id: str) -> str:
        return f"{prefix}:{request_id}"

    def _shared_set(self, key: str, value: object, *, tags: Sequence[str] = ()) -> None:
        if not self.publish_shared_memory:
            return
        ttl = None if self.shared_memory_ttl_seconds <= 0 else self.shared_memory_ttl_seconds
        try:
            payload = to_json_safe(value, max_depth=14, max_items=_SHARED_STRUCTURE_ITEMS, max_string_length=self.max_shared_string_chars)
            self.shared_memory.set(
                key, payload, ttl=ttl, tags=list(tags),
                metadata={"agent": self.name, "agent_id": self.agent_id, "schema": "financial_agent.v1"},
            )
        except Exception as exc:
            self._handle_shared_memory_error("set", key, exc)

    def _handle_shared_memory_error(self, operation: str, key: str, exc: BaseException) -> None:
        with self._metrics_lock:
            self._shared_memory_errors += 1
        if self.fail_on_shared_memory_error:
            raise BaseStateError(
                "FinancialAgent shared-memory operation failed",
                component=self.name, operation=f"shared_memory.{operation}",
                context={"key": key, "error_type": type(exc).__name__}, cause=exc,
            ) from exc
        logger.warning("FinancialAgent shared-memory %s degraded | key=%s | error=%s", operation, key, type(exc).__name__)

    def _publish_event(self, event_type: str, payload: Mapping[str, object]) -> None:
        if not self.publish_shared_memory:
            return
        publish = getattr(self.shared_memory, "publish", None)
        if not callable(publish):
            return
        try:
            publish(self.event_channel, {"schema": "financial_agent.event.v1", "event_type": event_type, "agent": self.name, **dict(payload)})
        except Exception as exc:
            self._handle_shared_memory_error("publish", self.event_channel, exc)

    def _parameter_preview(self, parameters: Mapping[str, object]) -> dict[str, object]:
        """Scalars verbatim (bounded); containers reduced to their size, so amounts and holdings stay out."""

        preview: dict[str, object] = {}
        for key, value in list(parameters.items())[: self.max_shared_list_items]:
            if value is None or isinstance(value, (bool, int)):
                preview[str(key)] = value
            elif isinstance(value, str):
                preview[str(key)] = value if len(value) <= 64 else f"<{len(value)} chars>"
            elif isinstance(value, Mapping):
                preview[str(key)] = f"<{len(value)} keys>"
            elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
                preview[str(key)] = f"<{len(value)} items>"
            else:
                preview[str(key)] = f"<{type(value).__name__}>"
        return preview

    def _request_record(self, request: FinancialRequest) -> dict[str, object]:
        record: dict[str, object] = {
            "schema": "financial_agent.request.v1",
            "request_id": request.request_id,
            "domain": request.domain,
            "operation": request.operation,
            "parameters": self._parameter_preview(request.parameters),
            "metadata": redact_mapping(dict(request.metadata)),
        }
        record["fingerprint"] = stable_fingerprint({"domain": request.domain, "operation": request.operation, "parameters": to_json_compatible(dict(request.parameters))})
        return record

    def _bounded(self, envelope: Mapping[str, object]) -> dict[str, object]:
        """Shared-memory copy of an envelope: long result lists are cut and the cut is recorded."""

        shared = dict(envelope)
        result = shared.get("result")
        truncated: dict[str, int] = {}
        limit = self.max_shared_list_items
        if isinstance(result, Mapping):
            trimmed: dict[str, object] = {}
            for key, value in result.items():
                if isinstance(value, list) and len(value) > limit:
                    truncated[str(key)] = len(value) - limit
                    value = value[:limit]
                trimmed[str(key)] = value
            shared["result"] = trimmed
        elif isinstance(result, list) and len(result) > limit:
            truncated["result"] = len(result) - limit
            shared["result"] = result[:limit]
        if truncated:
            shared["truncated_items"] = truncated
        return shared

    def _publish_result(self, request: FinancialRequest, result: FinancialResult, envelope: Mapping[str, object], elapsed: float) -> None:
        if not self.publish_shared_memory:
            return
        name = f"{request.domain}.{request.operation}"
        succeeded = result.status is FinancialResultStatus.SUCCESS
        summary: dict[str, object] = {
            "schema": "financial_agent.summary.v1",
            "request_id": request.request_id,
            "operation": name,
            "status": envelope["status"],
            "success": succeeded,
            "warnings": len(result.warnings),
            "sources": len(result.sources),
            "elapsed_seconds": round(elapsed, 6),
            "fingerprint": envelope["fingerprint"],
        }
        tags = ("financial", name)
        self._shared_set(self._key(self.result_key_prefix, request.request_id), self._bounded(envelope), tags=(*tags, "result"))
        self._shared_set(self._key(self.summary_key_prefix, request.request_id), summary, tags=(*tags, "summary"))
        self._shared_set(self.latest_result_key, summary, tags=(*tags, "latest"))
        if not succeeded:
            error = envelope.get("error")
            error_details = {str(key): value for key, value in error.items()} if isinstance(error, Mapping) else {}
            self._shared_set(self._key(self.error_key_prefix, request.request_id), {"schema": "financial_agent.error.v1", **error_details, "request_id": request.request_id, "operation": name}, tags=(*tags, "error"))
        self._publish_event("completed" if succeeded else "failed", {"request_id": request.request_id, "operation": name, "status": envelope["status"]})

    def _publish_defect(self, request: FinancialRequest, error: BaseException) -> None:
        name = f"{request.domain}.{request.operation}"
        self._shared_set(
            self._key(self.error_key_prefix, request.request_id),
            {"schema": "financial_agent.error.v1", "request_id": request.request_id, "operation": name, "error_type": type(error).__name__, "message": str(getattr(error, "message", None) or error)[:1000], "defect": True},
            tags=("financial", "error", "defect"),
        )
        self._publish_event("defect", {"request_id": request.request_id, "operation": name, "error_type": type(error).__name__})

    def get_financial_result(self, request_id: str) -> Mapping[str, object] | None:
        """An envelope this agent published to SharedMemory (bounded copy), if still present."""

        if not self.publish_shared_memory:
            return None
        key = self._key(self.result_key_prefix, require_text(request_id, "request_id"))
        try:
            value = self.shared_memory.get(key)
        except Exception as exc:
            self._handle_shared_memory_error("get", key, exc)
            return None
        return value if isinstance(value, Mapping) else None

    # ------------------------------------------------------------------------------------------------------------
    # Metrics, capabilities, diagnostics
    # ------------------------------------------------------------------------------------------------------------

    def _record_metrics(self, request: FinancialRequest, spec: FinancialOperation | None, result: FinancialResult | None,
                        elapsed: float, *, error: BaseException | None = None) -> None:
        if not self.metrics_enabled:
            return
        with self._metrics_lock:
            self._requests_total += 1
            self._elapsed_total += max(0.0, elapsed)
            self._by_operation[spec.name if spec is not None else "unsupported"] += 1
            if error is not None:
                self._requests_failed += 1
                self._by_status["defect"] += 1
                self._error_types[type(error).__name__] += 1
                return
            assert result is not None
            self._by_status[str(result.status.value)] += 1  # type: ignore[union-attr]
            if result.status is FinancialResultStatus.SUCCESS:
                self._requests_success += 1
            else:
                self._requests_failed += 1
                kind = result.diagnostics.get("error_type")
                if kind:
                    self._error_types[str(kind)] += 1

    def metrics_snapshot(self) -> dict[str, object]:
        with self._metrics_lock:
            total = self._requests_total
            return {
                "requests_total": total,
                "requests_success": self._requests_success,
                "requests_failed": self._requests_failed,
                "success_rate": (self._requests_success / total) if total else None,
                "elapsed_seconds_total": round(self._elapsed_total, 6),
                "elapsed_seconds_mean": round(self._elapsed_total / total, 6) if total else None,
                "by_operation": dict(self._by_operation),
                "by_status": dict(self._by_status),
                "error_types": dict(self._error_types),
                "shared_memory_errors": self._shared_memory_errors,
                "local_memory_entries": len(self.memory.recent()),
                "market_data": self.data.counts(),
            }

    def capabilities(self) -> dict[str, object]:
        return {
            "agent": self.name,
            "version": __version__,
            "domains": tuple(sorted({self._operations[name].domain for name in self.allowed_operations})),
            "operations": tuple(sorted(self.allowed_operations)),
            "executes_payments": False,
            "executes_trades": False,
            "authorization_mode": self.authorization_mode,
            "optimization_delegation": self.optimization_enabled,
            "shared_memory": self.publish_shared_memory,
            "local_memory": True,
            "point_in_time": True,
        }

    def diagnostics(self) -> dict[str, object]:
        return {
            "agent": self.name,
            "agent_id": self.agent_id,
            "version": __version__,
            "enabled": self.enabled,
            "operations": len(self.allowed_operations),
            "peers_resolved": sorted(self._peers),
            "metrics": self.metrics_snapshot(),
            "runtime": self.runtime_status(),
        }

    def extract_performance_metrics(self, result: object) -> dict[str, float]:
        """Operational telemetry only; financial correctness is not scored here."""

        if not isinstance(result, Mapping):
            return {}
        elapsed = result.get("elapsed_seconds")
        warnings, sources = result.get("warnings"), result.get("sources")
        return {
            "financial.success": 1.0 if result.get("success") is True else 0.0,
            "financial.elapsed_seconds": float(elapsed) if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and elapsed == elapsed else 0.0,
            "financial.warnings": float(len(warnings)) if isinstance(warnings, (list, tuple)) else 0.0,
            "financial.sources": float(len(sources)) if isinstance(sources, (list, tuple)) else 0.0,
        }

    # ------------------------------------------------------------------------------------------------------------
    # Checkpointing: financial (local) memory and, optionally, the point-in-time market-data store
    # ------------------------------------------------------------------------------------------------------------

    def _market_state(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "observations": [item.to_dict() for item in self.data.stored_observations()],
            "exchange_rates": [item.to_dict() for item in self.data.stored_exchange_rates()],
        }

    def _export_checkpoint_state(self) -> Mapping[str, Any]:
        return {
            "schema": self.CHECKPOINT_SCHEMA,
            "version": __version__,
            "metrics": self.metrics_snapshot(),
            "local_memory": self.memory.snapshot() if self.checkpoint_local_memory else None,
            "market_data": self._market_state() if self.checkpoint_market_data else None,
        }

    def _import_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise FinancialValidationError("FinancialAgent checkpoint state must be a mapping")
        if state.get("schema") != self.CHECKPOINT_SCHEMA:
            raise FinancialValidationError("incompatible FinancialAgent checkpoint schema", context={"actual": str(state.get("schema")), "expected": self.CHECKPOINT_SCHEMA})
        if str(state.get("version", "")) != __version__:
            raise FinancialValidationError("incompatible FinancialAgent checkpoint version", context={"actual": str(state.get("version")), "expected": __version__})

        memory_state = state.get("local_memory") if self.restore_local_memory else None
        market = state.get("market_data") if self.checkpoint_market_data else None
        # Everything is decoded and validated before anything is replaced.
        observations: tuple[MarketObservation, ...] = ()
        rates: tuple[ExchangeRate, ...] = ()
        if market is not None:
            if not isinstance(market, Mapping):
                raise FinancialValidationError("checkpoint market_data must be a mapping")
            observations = tuple(_decode(_observation, item, "market_data.observations") for item in market.get("observations", ()))
            rates = tuple(_decode(_exchange_rate, item, "market_data.exchange_rates") for item in market.get("exchange_rates", ()))

        previous_memory = self.memory.snapshot()
        previous_market = self._market_state()
        try:
            if memory_state is not None:
                self.memory.restore(memory_state)
            if market is not None:
                self.data.clear()
                if observations:
                    self.data.ingest_observations(observations)
                if rates:
                    self.data.ingest_exchange_rates(rates)
        except Exception as exc:
            self.memory.restore(previous_memory)
            self.data.clear()
            self.data.ingest_observations(tuple(_observation(item) for item in previous_market["observations"]))
            self.data.ingest_exchange_rates(tuple(_exchange_rate(item) for item in previous_market["exchange_rates"]))
            raise FinancialValidationError(
                "FinancialAgent checkpoint could not be applied; the previous state was kept",
                context={"error_type": type(exc).__name__}, cause=exc,
            ) from exc

    # ------------------------------------------------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------------------------------------------------

    def reset_local_state(self, *, memory: bool = True, market_data: bool = True) -> None:
        """Clear the agent's local state. SharedMemory and metrics are left untouched."""

        if memory:
            self.memory.clear()
        if market_data:
            self.data.clear()

    def shutdown(self) -> None:
        """Stop this runtime; local state and SharedMemory are not cleared."""

        lifecycle = self._runtime_status.lifecycle
        if lifecycle in {RuntimeLifecycle.STOPPING, RuntimeLifecycle.STOPPED}:
            return
        self.transition_runtime_lifecycle(RuntimeLifecycle.STOPPING)
        self.operational_state = "stopping"
        self.transition_runtime_lifecycle(RuntimeLifecycle.STOPPED)
        self.operational_state = "stopped"
        logger.info("FinancialAgent stopped | agent_id=%s", self.agent_id)


__all__ = ["FinancialAgent", "FinancialOperation"]


if __name__ == "__main__":
    print("\n=== Running Financial Agent ===\n")
    printer.status("TEST", "Financial Agent initialized", "info")
    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()
    agent = FinancialAgent(shared_memory=shared_memory, agent_factory=agent_factory, config={"publish_shared_memory": False})
    printer.status("START", agent, "info")

    usd = lambda value: {"amount": value, "currency": "USD"}  # noqa: E731
    present = agent.perform_task(
        {
            "request_id": "financial-agent-smoke-pv",
            "operation": "calculation.present_value",
            "parameters": {
                "future": usd("1000"),
                "rate": {"value": "0.05", "convention": "effective"},
                "start": "2024-01-01",
                "end": "2025-01-01",
            },
        }
    )
    printer.status("TASK #1", present["result"]["present_value"]["amount"], "success" if present["success"] else "error") # type: ignore

    rejected = agent.perform_task({"operation": "calculation.present_value", "parameters": {"future": usd(1000.5)}})
    printer.status("TASK #2", rejected["status"], "success" if not rejected["success"] else "error")

    recalled = agent.perform_task({"operation": "memory.recall", "parameters": {"request_id": "financial-agent-smoke-pv"}})
    printer.status("TASK #3", recalled["result"]["found"], "success" if recalled["result"]["found"] else "error") # type: ignore
    printer.status("METRICS", agent.metrics_snapshot()["requests_total"], "info")
    print("\n=== Test ran successfully ===\n")
