"""SLAI v2.4: reproducible forecasting, research signals, model adapters.

Combines forecasting, decision support and trained-model *inference* contracts.
No synthetic prices, untrained models, direct trading, network access, or
second training framework. The simple baselines are real deterministic models,
not a substitute for validated market forecasting. Backtests use chronological
cutoffs and source retrieval timestamps to reject obvious look-ahead.

Academic foundations: Hyndman & Athanasopoulos, Forecasting: Principles and
Practice; Diebold & Mariano (1995); Bailey et al. on backtest overfitting.
"""
from __future__ import annotations

__version__ = "2.4.0"

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
from enum import Enum
from threading import RLock
from typing import Any, Protocol, runtime_checkable
import re

from .data.series import validate_series
from .financial_data import FinancialData
from .financial_types import FinancialAmount, MarketObservation
from .financial_valuation import FinancialValuation
from .utils.config_loader import get_config_section, load_global_config
from .utils.financial_errors import *
from .utils.financial_helpers import *
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Financial Intelligence")
printer = PrettyPrinter()


class ForecastMethod(str, Enum):
    NAIVE = "naive"
    DRIFT = "drift"
    SEASONAL_NAIVE = "seasonal_naive"
    REGISTERED_MODEL = "registered_model"


class ResearchClassification(str, Enum):
    BUY_CANDIDATE = "buy_candidate"
    SELL_CANDIDATE = "sell_candidate"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class FinancialForecast:
    instrument_id: str
    field_name: str
    issued_at: datetime
    last_observation_at: datetime
    method: ForecastMethod
    horizon_steps: int
    values: tuple[Decimal, ...]
    currency: str | None
    unit: str | None
    source_ids: tuple[str, ...]
    model_name: str | None = None
    model_version: str | None = None
    interpretation: str = "Point forecasts per observation step; future dates and confidence intervals are not inferred."
    source_revisions: tuple[str, ...] = ()
    latest_retrieval_at: datetime | None = None
    model_artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_id", normalize_identifier(self.instrument_id, "instrument_id"))
        object.__setattr__(self, "field_name", normalize_identifier(self.field_name, "field_name"))
        issued, last = to_utc_datetime(self.issued_at), to_utc_datetime(self.last_observation_at)
        if last > issued:
            raise FinancialValidationError("Forecast contains observations after issuance")
        object.__setattr__(self, "issued_at", issued)
        object.__setattr__(self, "last_observation_at", last)
        if not isinstance(self.method, ForecastMethod):
            raise FinancialValidationError("Forecast method must be ForecastMethod")
        require_positive_int(self.horizon_steps, "horizon_steps", maximum=4096)
        if not isinstance(self.values, (tuple, list)) or len(self.values) != self.horizon_steps:
            raise FinancialValidationError("Forecast values must match horizon")
        object.__setattr__(self, "values", tuple(to_decimal(v, "forecast value") for v in self.values))
        if self.currency is not None:
            object.__setattr__(self, "currency", normalize_currency_code(self.currency))
        if self.unit is not None:
            object.__setattr__(self, "unit", require_text(self.unit, "unit", max_length=64))
        if self.currency is None and self.unit is None:
            raise FinancialValidationError("Forecast requires a currency or unit")
        if not isinstance(self.source_ids, (tuple, list)) or not self.source_ids:
            raise FinancialValidationError("Forecast requires source_ids")
        object.__setattr__(self, "source_ids", tuple(sorted({normalize_identifier(x, "source_id") for x in self.source_ids})))
        if not isinstance(self.source_revisions, (list, tuple)):
            raise FinancialValidationError("source_revisions must be list/tuple")
        object.__setattr__(self, "source_revisions", tuple(sorted({normalize_identifier(x, "revision") for x in self.source_revisions})))
        if self.latest_retrieval_at is not None:
            retrieved = to_utc_datetime(self.latest_retrieval_at)
            if retrieved > issued:
                raise InvalidMarketDataError("Forecast source retrieved after issuance")
            object.__setattr__(self, "latest_retrieval_at", retrieved)
        if self.method is ForecastMethod.REGISTERED_MODEL:
            if not self.model_name or not self.model_version:
                raise FinancialModelValidationError("Registered-model forecast requires model provenance")
        elif self.model_name is not None or self.model_version is not None or self.model_artifact_sha256 is not None:
            raise FinancialValidationError("Baseline forecast cannot claim trained-model identity")
        if self.model_name is not None:
            object.__setattr__(self, "model_name", normalize_identifier(self.model_name, "model_name"))
        if self.model_version is not None:
            object.__setattr__(self, "model_version", normalize_identifier(self.model_version, "model_version"))
        if self.model_artifact_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", self.model_artifact_sha256):
            raise FinancialModelValidationError("model_artifact_sha256 must be a SHA-256 digest")
        object.__setattr__(self, "interpretation", require_text(self.interpretation, "interpretation", max_length=1024))

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class ForecastBacktest:
    method: ForecastMethod
    evaluation_count: int
    horizon_steps: int
    mean_absolute_error: Decimal
    root_mean_squared_error: Decimal
    mean_signed_error: Decimal
    first_origin_at: datetime
    last_origin_at: datetime
    interpretation: str = "Chronological rolling-origin evaluation with point-in-time source-availability checks."

    def __post_init__(self) -> None:
        if not isinstance(self.method, ForecastMethod) or self.method is ForecastMethod.REGISTERED_MODEL:
            raise FinancialValidationError("Backtest requires supported baseline method")
        require_positive_int(self.evaluation_count, "evaluation_count")
        require_positive_int(self.horizon_steps, "horizon_steps")
        for key in ("mean_absolute_error", "root_mean_squared_error", "mean_signed_error"):
            object.__setattr__(self, key, to_decimal(getattr(self, key), key))
        if self.mean_absolute_error < 0 or self.root_mean_squared_error < 0:
            raise FinancialDomainError("Forecast error magnitudes must be nonnegative")
        first, last = to_utc_datetime(self.first_origin_at), to_utc_datetime(self.last_origin_at)
        if first > last:
            raise FinancialValidationError("Backtest origin order is invalid")
        object.__setattr__(self, "first_origin_at", first)
        object.__setattr__(self, "last_origin_at", last)

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class ValueResearchSignal:
    """Comparative valuation screen only, never a broker order or recommendation."""

    classification: ResearchClassification
    price: FinancialAmount
    fair_value_assumption: FinancialAmount
    implied_upside: Decimal
    threshold: Decimal
    evaluated_at: datetime
    source_id: str
    interpretation: str = "Rule-based valuation screen; depends entirely on supplied fair-value assumptions."

    def __post_init__(self) -> None:
        if not isinstance(self.classification, ResearchClassification):
            raise FinancialValidationError("Research classification must be ResearchClassification")
        if not isinstance(self.price, FinancialAmount) or self.price.amount <= 0:
            raise FinancialDomainError("Research price must be positive FinancialAmount")
        if not isinstance(self.fair_value_assumption, FinancialAmount) or self.fair_value_assumption.amount < 0:
            raise FinancialDomainError("Research fair value must be nonnegative FinancialAmount")
        require_same_currency(self.price.currency, self.fair_value_assumption.currency, operation="research valuation")
        object.__setattr__(self, "implied_upside", to_decimal(self.implied_upside, "implied_upside"))
        threshold = to_decimal(self.threshold, "threshold")
        if not 0 < threshold < 1:
            raise FinancialDomainError("Research threshold must lie between 0 and 1")
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "evaluated_at", to_utc_datetime(self.evaluated_at))
        object.__setattr__(self, "source_id", normalize_identifier(self.source_id, "source_id"))

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@dataclass(frozen=True, slots=True)
class FinancialModelManifest:
    """External model evidence metadata supplied by a trusted model owner.

    Schema validation is NOT proof that the training metrics are truthful;
    SLAI model governance must independently attest training and evaluation.
    """

    name: str
    version: str
    field_name: str
    trained_data_through: datetime
    validated_at: datetime
    validation_method: str
    validation_metric: str
    validation_score: Decimal
    artifact_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", normalize_identifier(self.name, "model name"))
        object.__setattr__(self, "version", normalize_identifier(self.version, "model version"))
        object.__setattr__(self, "field_name", normalize_identifier(self.field_name, "field_name"))
        training = to_utc_datetime(self.trained_data_through, "trained_data_through")
        validated = to_utc_datetime(self.validated_at, "validated_at")
        if validated < training:
            raise FinancialModelValidationError("Model validation cannot predate training data cutoff")
        object.__setattr__(self, "trained_data_through", training)
        object.__setattr__(self, "validated_at", validated)
        if self.validation_method not in ("walk_forward", "temporal_holdout"):
            raise FinancialModelValidationError("Model validation method must be temporal")
        object.__setattr__(self, "validation_metric", normalize_identifier(self.validation_metric, "validation_metric"))
        metric = to_decimal(self.validation_score, "validation_score")
        if metric < 0:
            raise FinancialModelValidationError("Validation score must be nonnegative")
        object.__setattr__(self, "validation_score", metric)
        if not isinstance(self.artifact_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", self.artifact_sha256):
            raise FinancialModelValidationError("artifact_sha256 must be a SHA-256 hex digest")
        object.__setattr__(self, "artifact_sha256", self.artifact_sha256.lower())

    def to_dict(self) -> dict[str, Any]:
        return to_json_compatible(self)


@runtime_checkable
class FinancialPredictor(Protocol):
    """Trusted, preloaded inference adapter; no model deserialization here."""

    def predict(self, history: tuple[MarketObservation, ...], *, horizon_steps: int) -> Sequence[Decimal | str | int]: ...


class FinancialIntelligence:
    """Deterministic baselines, chronological testing, explicit model governance.

    Registered predictors are trusted application objects with validated
    manifests. They may have their own native runtime; this class owns only
    financial data contracts, output integrity and cutoff validation.
    """

    def __init__(self, data: FinancialData | None = None, valuation: FinancialValuation | None = None):
        self.config = load_global_config()
        self.financial_intelligence_config = get_config_section("financial_intelligence", config=self.config)
        cfg = self.financial_intelligence_config
        try:
            self.max_history_points = require_positive_int(cfg["max_history_points"], "max_history_points", maximum=1000000)
            self.max_horizon_steps = require_positive_int(cfg["max_horizon_steps"], "max_horizon_steps", maximum=4096)
            self.min_backtest_points = require_positive_int(cfg["min_backtest_points"], "min_backtest_points", minimum=2, maximum=100000)
            self.max_models = require_positive_int(cfg["max_models"], "max_models", maximum=10000)
            self.default_value_margin = to_decimal(cfg["default_value_margin"], "default_value_margin")
        except KeyError as exc:
            raise FinancialConfigurationError("Missing financial_intelligence configuration key", context={"key": str(exc)}) from exc
        if not 0 < self.default_value_margin < 1 or self.min_backtest_points >= self.max_history_points:
            raise FinancialConfigurationError("Invalid financial_intelligence margin or history/backtest limits")
        if data is not None and not isinstance(data, FinancialData):
            raise FinancialValidationError("data must be FinancialData or None")
        if valuation is not None and not isinstance(valuation, FinancialValuation):
            raise FinancialValidationError("valuation must be FinancialValuation or None")
        self.data = data if data is not None else FinancialData()
        self.valuation = valuation if valuation is not None else FinancialValuation()
        self._models: dict[str, tuple[FinancialModelManifest, FinancialPredictor]] = {}
        self._model_lock = RLock()

    def _validate_history(
        self, observations: Sequence[MarketObservation], *, as_of: datetime | str,
    ) -> tuple[tuple[MarketObservation, ...], datetime]:
        """Series rules (homogeneity, no look-ahead, ordering) are owned by data.series."""
        return validate_series(observations, as_of=as_of, max_points=self.max_history_points)

    def register_model(self, manifest: FinancialModelManifest, predictor: FinancialPredictor) -> None:
        """Register an externally loaded trusted predictor, no loading or training."""
        if not isinstance(manifest, FinancialModelManifest):
            raise FinancialModelValidationError("manifest must be FinancialModelManifest")
        if not callable(getattr(predictor, "predict", None)):
            raise FinancialModelValidationError("predictor must provide callable predict")
        with self._model_lock:
            if manifest.name in self._models:
                raise FinancialModelValidationError("Model name already registered; unregister explicitly before replacement")
            if len(self._models) >= self.max_models:
                raise FinancialModelValidationError("Model registry exceeds configured max_models")
            self._models[manifest.name] = (manifest, predictor)
        logger.info("Financial inference model registered | name=%s | version=%s", manifest.name, manifest.version)

    def unregister_model(self, name: str) -> bool:
        key = normalize_identifier(name, "model name")
        with self._model_lock:
            return self._models.pop(key, None) is not None

    def registered_models(self) -> tuple[FinancialModelManifest, ...]:
        with self._model_lock:
            return tuple(record[0] for _, record in sorted(self._models.items()))

    @staticmethod
    def _baseline(history: tuple[MarketObservation, ...], horizon_steps: int,
                  method: ForecastMethod, seasonal_period: int | None, precision: int) -> tuple[Decimal, ...]:
        values = tuple(x.value for x in history)
        if method is ForecastMethod.NAIVE:
            return (values[-1],) * horizon_steps
        if method is ForecastMethod.DRIFT:
            if len(values) < 2:
                raise FinancialDomainError("Drift forecast requires at least two observations")
            with localcontext() as ctx:
                ctx.prec = precision
                gradient = (values[-1] - values[0]) / Decimal(len(values) - 1)
                return tuple(values[-1] + gradient * k for k in range(1, horizon_steps + 1))
        if method is ForecastMethod.SEASONAL_NAIVE:
            if seasonal_period is None:
                raise FinancialValidationError("seasonal_period is required for seasonal_naive")
            require_positive_int(seasonal_period, "seasonal_period", minimum=2, maximum=len(values))
            pattern = values[-seasonal_period:]
            return tuple(pattern[k % seasonal_period] for k in range(horizon_steps))
        raise FinancialValidationError("Unsupported baseline forecast method")

    def forecast(
        self, observations: Sequence[MarketObservation], *, as_of: datetime | str,
        horizon_steps: int, method: ForecastMethod | str = ForecastMethod.NAIVE,
        seasonal_period: int | None = None, model_name: str | None = None,
    ) -> FinancialForecast:
        history, now = self._validate_history(observations, as_of=as_of)
        require_positive_int(horizon_steps, "horizon_steps", maximum=self.max_horizon_steps)
        try:
            choice = ForecastMethod(method)
        except ValueError as exc:
            raise FinancialValidationError("Unsupported forecast method") from exc
        model_version = None
        used_model = None
        artifact_sha256 = None
        if choice is ForecastMethod.REGISTERED_MODEL:
            if model_name is None:
                raise FinancialModelUnavailableError("registered_model requires model_name")
            key = normalize_identifier(model_name, "model_name")
            with self._model_lock:
                entry = self._models.get(key)
            if entry is None:
                raise FinancialModelUnavailableError("Requested trained model is unavailable")
            manifest, predictor = entry
            if manifest.field_name != history[0].field_name:
                raise FinancialModelValidationError("Model is not validated for the requested field")
            if now < manifest.validated_at or now < manifest.trained_data_through:
                raise FinancialModelValidationError("Model not validated or trained as of requested timestamp")
            try:
                raw = predictor.predict(history, horizon_steps=horizon_steps)
            except Exception as exc:
                raise FinancialModelError("Registered model inference failed", cause=exc) from exc
            if not isinstance(raw, (list, tuple)) or len(raw) != horizon_steps:
                raise FinancialModelValidationError("Model output length is incompatible with requested horizon")
            try:
                values = tuple(to_decimal(x, "model forecast value") for x in raw)
            except FinancialValidationError as exc:
                raise FinancialModelValidationError("Registered model returned invalid numeric values", cause=exc) from exc
            model_version = manifest.version
            used_model = manifest.name
            artifact_sha256 = manifest.artifact_sha256
        else:
            if model_name is not None:
                raise FinancialValidationError("model_name is only valid for registered_model")
            values = self._baseline(history, horizon_steps, choice, seasonal_period,
                                    self.valuation.calculation.precision)
        source_ids = tuple(sorted({x.source.source_id for x in history}))
        currency_code = None if history[0].currency is None else str(history[0].currency)
        forecast = FinancialForecast(history[0].instrument_id, history[0].field_name, now,
                                     history[-1].observed_at, choice, horizon_steps, values,
                                     currency_code, history[0].unit, source_ids, used_model, model_version,
                                     source_revisions=tuple(sorted({x.source.revision for x in history if x.source.revision})),
                                     latest_retrieval_at=max(x.source.retrieved_at for x in history),
                                     model_artifact_sha256=artifact_sha256)
        logger.debug("Financial forecast evaluated | method=%s | horizon=%s", choice.value, horizon_steps)
        return forecast

    def walk_forward_backtest(
        self, observations: Sequence[MarketObservation], *, as_of: datetime | str,
        method: ForecastMethod | str = ForecastMethod.NAIVE,
        horizon_steps: int = 1, min_training_points: int | None = None,
        seasonal_period: int | None = None,
    ) -> ForecastBacktest:
        """Rolling-origin out-of-sample test for deterministic baselines.

        An origin is eligible only if every training point was retrieved by the origin's decision time,
        and the *next* target observation was not yet observed.
        Learned models need versioned historical checkpoints and are intentionally rejected from this baseline-specific routine.
        """
        history, cutoff = self._validate_history(observations, as_of=as_of)
        require_positive_int(horizon_steps, "horizon_steps", maximum=self.max_horizon_steps)
        train_min = self.min_backtest_points if min_training_points is None else require_positive_int(
            min_training_points, "min_training_points", minimum=2, maximum=self.max_history_points)
        if train_min + horizon_steps > len(history):
            raise FinancialDomainError("Insufficient chronological data for backtest")
        try:
            choice = ForecastMethod(method)
        except ValueError as exc:
            raise FinancialValidationError("Unsupported forecast method") from exc
        if choice is ForecastMethod.REGISTERED_MODEL:
            raise FinancialModelValidationError("Model backtests require historical model checkpoints and separate governance")
        errors: list[Decimal] = []
        origin_times: list[datetime] = []
        with localcontext() as ctx:
            ctx.prec = self.valuation.calculation.precision
            for size in range(train_min, len(history) - horizon_steps + 1):
                training = history[:size]
                origin = max(x.source.retrieved_at for x in training)
                # Reject this origin rather than pretending late revisions were known.
                if origin >= history[size].observed_at:
                    continue
                forecast = self._baseline(training, horizon_steps, choice, seasonal_period,
                                          self.valuation.calculation.precision)
                actual = history[size + horizon_steps - 1].value
                errors.append(forecast[-1] - actual)
                origin_times.append(origin)
            if not errors:
                raise FinancialDomainError("No leak-free evaluation origins at the requested horizon")
            n = Decimal(len(errors))
            mae = sum(map(abs, errors), Decimal(0)) / n
            mse = sum((e * e for e in errors), Decimal(0)) / n
            rmse = mse.sqrt()
            bias = sum(errors, Decimal(0)) / n
        return ForecastBacktest(choice, len(errors), horizon_steps, mae, rmse, bias,
                                min(origin_times), max(origin_times))

    def assess_valuation(
        self, price: MarketObservation, *, fair_value: FinancialAmount,
        as_of: datetime | str, margin: Any | None = None,
        max_age: timedelta | None = None,
    ) -> ValueResearchSignal:
        """Label fair-value dislocation, not personal suitability or trade advice.

        BUY_CANDIDATE means fair value ≥ price × (1+threshold).
        SELL_CANDIDATE means fair value ≤ price × (1-threshold).
        """
        if not isinstance(price, MarketObservation) or price.currency is None:
            raise FinancialValidationError("Price must be currency-qualified MarketObservation")
        if not isinstance(fair_value, FinancialAmount):
            raise FinancialValidationError("fair_value must be FinancialAmount")
        require_same_currency(price.currency, fair_value.currency, operation="valuation research")
        now = to_utc_datetime(as_of, "as_of")
        if price.source.retrieved_at > now:
            raise InvalidMarketDataError("Research price was not available at as_of")
        ttl = timedelta(seconds=self.data.max_age_seconds) if max_age is None else max_age
        price.assert_fresh(as_of=now, max_age=ttl)
        if price.value <= 0 or fair_value.amount < 0:
            raise FinancialDomainError("Research requires positive price and nonnegative fair value")
        threshold = self.default_value_margin if margin is None else to_decimal(margin, "margin")
        if not 0 < threshold < 1:
            raise FinancialDomainError("Valuation margin must fall strictly between zero and one")
        with localcontext() as ctx:
            ctx.prec = self.valuation.calculation.precision
            upside = fair_value.amount / price.value - 1
        classification = (ResearchClassification.BUY_CANDIDATE if upside >= threshold else
                          ResearchClassification.SELL_CANDIDATE if upside <= -threshold else
                          ResearchClassification.HOLD)
        return ValueResearchSignal(classification, FinancialAmount(price.value, price.currency),
                                   fair_value, upside, threshold, now, price.source.source_id)


__all__ = ["ForecastMethod", "ResearchClassification", "FinancialForecast", "ForecastBacktest",
           "ValueResearchSignal", "FinancialModelManifest", "FinancialPredictor", "FinancialIntelligence"]

if __name__ == "__main__":
    print("\n=== Running Financial Intelligence ===\n")
    printer.status("TEST", "Financial Intelligence initialized", "info")
    from datetime import timezone
    from .financial_types import FinancialDataSource
    service = FinancialIntelligence()
    history = []
    for i, value in enumerate(("100", "101", "99", "104", "103", "108", "110", "111"), 1):
        stamp = datetime(2024, 1, i, tzinfo=timezone.utc)
        source = FinancialDataSource("example-feed", stamp)
        history.append(MarketObservation("INDEX.A", "close", Decimal(value), stamp, source, "EUR"))
    forecast = service.forecast(history, as_of=history[-1].observed_at, horizon_steps=2, method="naive")
    assert forecast.values == (Decimal("111"), Decimal("111"))
    report = service.walk_forward_backtest(history, as_of=history[-1].observed_at, min_training_points=3)
    assert report.evaluation_count > 0 and report.mean_absolute_error >= 0
    signal = service.assess_valuation(history[-1], fair_value=FinancialAmount.of("130", "EUR"),
                                      as_of=history[-1].observed_at)
    assert signal.classification is ResearchClassification.BUY_CANDIDATE
    print("\n=== Test ran successfully ===\n")
