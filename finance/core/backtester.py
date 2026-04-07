from __future__ import annotations

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence
from prettytable import PrettyTable
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

from .utils.config_loader import load_global_config, get_config_section
from .utils.financial_errors import (classify_external_exception,  ModelTrainingError,
                                                 DataUnavailableError, log_error, ErrorContext,
                                                 FeatureEngineeringError, PersistenceError,
                                                 InvalidConfigurationError, ValidationError)
from .finance_memory import FinanceMemory
from .batch_manager import BatchManager
from src.tuning.tuner import HyperparamTuner
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Backtester")
printer = PrettyPrinter


@dataclass(slots=True)
class Position:
    symbol: str
    direction: str
    entry_price: float
    quantity: float
    entry_value: float
    entry_time: str

    def pnl_ratio(self, price: float) -> float:
        change = (float(price) - self.entry_price) / max(self.entry_price, 1e-12)
        return float(change if self.direction == "long" else -change)


@dataclass(slots=True)
class WindowResult:
    window_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    portfolio_value_start: float
    portfolio_value_end: float
    trade_count: int
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Backtester:
    def __init__(self, tracker: Any) -> None:
        self.config = load_global_config()
        self.bt_config = get_config_section("backtester")
        self.tracker = tracker
        self.finance_memory = FinanceMemory()

        self.initial_capital = float(self.config.get("initial_capital", self.bt_config.get("initial_capital", 100)))
        self.position_size = float(self.config.get("position_size", 0.1))
        self.stop_loss = float(self.config.get("stop_loss", 0.1))
        self.take_profit = float(self.config.get("take_profit", 0.15))
        self.transaction_cost = float(self.bt_config.get("transaction_cost", 0.001))
        self.strategy = str(self.bt_config.get("strategy", "prediction_based")).lower()
        self.train_window = int(self.bt_config.get("train_window", 180))
        self.test_window = int(self.bt_config.get("test_window", 30))
        self.buffer_size = int(self.bt_config.get("buffer_size", 500))

        self.metrics = {
            "sharpe_ratio": [],
            "sortino_ratio": [],
            "calmar_ratio": [],
            "max_drawdown": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "mean_return": [],
            "volatility": [],
            "win_rate": [],
            "profit_factor": [],
            "kelly_criterion": [],
            "mean_absolute_error": [],
        }
        self.portfolio_history: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.window_results: List[WindowResult] = []
        self._last_aggregated_metrics: Dict[str, float] = {}
        self.model: Optional[Any] = None
        self.feature_columns: List[str] = []
        self.feature_processor: Dict[str, Any] = {}

        self.predictions_table = PrettyTable()
        self.signals_table = PrettyTable()
        self.indicators_table = PrettyTable()
        self.table = None
        self.predictions_table.field_names = ["Symbol", "Actual", "1h Pred", "24h Pred", "1w Pred", "Uncertainty", "P-Value", "Status"]
        self.signals_table.field_names = ["Source", "Key Activity", "Impact", "Confidence"]
        self.signals_table.align = "l"
        self.indicators_table.field_names = ["Indicator", "Value", "Status", "Thresholds"]
        self.indicators_table.align = "l"

        self._validate_configuration()
        self._init_memory()

    def _context(self, operation: str, **metadata: Any) -> ErrorContext:
        return ErrorContext(component="backtester", operation=operation, metadata=metadata or {})

    def _validate_configuration(self) -> None:
        if not (0 < self.position_size <= 1):
            raise InvalidConfigurationError("position_size must be in (0, 1].", context=self._context("validate_config", position_size=self.position_size))
        if self.transaction_cost < 0:
            raise InvalidConfigurationError("transaction_cost must be non-negative.", context=self._context("validate_config", transaction_cost=self.transaction_cost))
        if self.train_window <= 10 or self.test_window <= 1:
            raise InvalidConfigurationError("train_window must be > 10 and test_window must be > 1.", context=self._context("validate_config", train_window=self.train_window, test_window=self.test_window))
        if self.strategy not in {"prediction_based", "momentum", "mean_reversion"}:
            raise InvalidConfigurationError("Unsupported strategy configured.", context=self._context("validate_config", strategy=self.strategy))

    def _init_memory(self) -> None:
        self.finance_memory.add_financial_data(data={}, data_type="backtest_metrics", tags=["backtesting"], priority="high")
        self.finance_memory.add_financial_data(data=[], data_type="trade_history", tags=["backtesting"], priority="medium")

    def _safe_memory_add(self, *, data: Mapping[str, Any], data_type: str, tags: Sequence[str], priority: str = "medium", metadata: Optional[Mapping[str, Any]] = None) -> None:
        try:
            self.finance_memory.add_financial_data(data=dict(data), data_type=data_type, tags=list(tags), priority=priority, metadata=dict(metadata or {}))
        except Exception as exc:
            handled = PersistenceError("Failed to persist backtest artifact.", context=self._context("memory_add", data_type=data_type), cause=exc)
            log_error(handled, logger_=logger)

    def store_last_metrics(self, metrics: dict) -> None:
        self._last_aggregated_metrics = dict(metrics or {})
        logger.info("Stored last metrics with keys=%s", list(self._last_aggregated_metrics.keys()))

    def get_last_metrics(self) -> dict:
        if not self._last_aggregated_metrics:
            results = self.finance_memory.query(data_type="backtest_metrics", tags=[f"backtest_{self.strategy}"], limit=1)
            if results:
                self._last_aggregated_metrics = dict(results[0].get("data", {}))
        return dict(self._last_aggregated_metrics)

    def load_batch_data(self, batch_manager: BatchManager) -> pd.DataFrame:
        if batch_manager is None:
            raise ValidationError("batch_manager is required.", context=self._context("load_batch_data"))
        batches = batch_manager.load_all_batches_from_disk()
        if not batches:
            logger.warning("No batch data loaded from disk.")
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        for idx, batch_dict in enumerate(batches):
            batch_timestamp = batch_dict.get("batch_timestamp")
            batch_payload = batch_dict.get("data")
            if batch_timestamp is None or not isinstance(batch_payload, Mapping):
                logger.warning("Skipping malformed batch at index=%s", idx)
                continue
            for symbol, item_data in batch_payload.items():
                if isinstance(item_data, Mapping):
                    price = item_data.get("price")
                    volume = item_data.get("volume", 0.0)
                elif isinstance(item_data, (int, float)):
                    price = item_data
                    volume = 0.0
                else:
                    continue
                try:
                    rows.append({
                        "timestamp": float(batch_timestamp),
                        "datetime": datetime.fromtimestamp(float(batch_timestamp)),
                        "symbol": str(symbol).upper(),
                        "open": float(price),
                        "high": float(price),
                        "low": float(price),
                        "close": float(price),
                        "volume": max(float(volume), 0.0),
                        "source_file_batch_index": idx,
                    })
                except (TypeError, ValueError):
                    logger.warning("Skipping malformed row in batch=%s symbol=%s", idx, symbol)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        return df.dropna(subset=["datetime", "close"]).sort_values(["symbol", "datetime"]).reset_index(drop=True)

    def _prepare_historical_data(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        if historical_data is None or historical_data.empty:
            raise DataUnavailableError("Historical data is empty.", context=self._context("prepare_historical_data"))
        df = historical_data.copy()
        if "symbol" not in df.columns:
            df["symbol"] = "UNKNOWN"
        if "datetime" not in df.columns:
            if "date" in df.columns:
                df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
            elif "timestamp" in df.columns:
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
            else:
                raise ValidationError("Historical data requires datetime/date/timestamp.", context=self._context("prepare_historical_data"))
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        for col in ["close", "volume"]:
            if col not in df.columns:
                df[col] = 0.0 if col == "volume" else np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["open", "high", "low"]:
            if col not in df.columns:
                df[col] = df["close"]
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df["close"])
        df["volume"] = df["volume"].fillna(0.0)
        df = df.dropna(subset=["datetime", "close"]).sort_values(["symbol", "datetime"]).reset_index(drop=True)
        if df.empty:
            raise DataUnavailableError("No usable historical rows remain after cleaning.", context=self._context("prepare_historical_data"))
        return df

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        enriched = df.copy()
        close = pd.to_numeric(enriched["close"], errors="coerce")
        volume = pd.to_numeric(enriched["volume"], errors="coerce").fillna(0.0)
        enriched["return_1"] = close.pct_change()
        for lag in [1, 2, 3, 5, 10]:
            enriched[f"return_lag_{lag}"] = close.pct_change(lag)
        enriched["volatility_5"] = close.pct_change().rolling(5, min_periods=3).std()
        enriched["volatility_20"] = close.pct_change().rolling(20, min_periods=5).std()
        enriched["momentum_5"] = close / close.shift(5) - 1.0
        enriched["momentum_20"] = close / close.shift(20) - 1.0
        enriched["sma_5"] = close.rolling(5, min_periods=3).mean()
        enriched["sma_20"] = close.rolling(20, min_periods=5).mean()
        enriched["ema_12"] = close.ewm(span=12, adjust=False).mean()
        enriched["ema_26"] = close.ewm(span=26, adjust=False).mean()
        enriched["volume_change"] = volume.pct_change()
        enriched["volume_zscore"] = (volume - volume.rolling(20, min_periods=5).mean()) / (volume.rolling(20, min_periods=5).std() + 1e-8)
        enriched["rsi"] = self._calculate_rsi(close, 14)
        enriched = pd.concat([enriched, self._calculate_macd(close)], axis=1)
        enriched["next_close"] = close.shift(-1)
        enriched["next_return"] = close.pct_change().shift(-1)
        return enriched

    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.rolling(period, min_periods=max(2, period // 2)).mean()
        avg_loss = loss.rolling(period, min_periods=max(2, period // 2)).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame({"macd_line": macd_line, "signal_line": signal_line, "histogram": histogram})

    def _build_model(self, model_type: str) -> Any:
        model_type = str(model_type).strip().lower()
        if model_type == "gradientboosting":
            return GradientBoostingRegressor(random_state=42)
        if model_type == "randomforest":
            return RandomForestRegressor(random_state=42, n_estimators=200, min_samples_leaf=2)
        if model_type == "ridge":
            return Ridge(alpha=1.0, random_state=42)
        if model_type == "lasso":
            return Lasso(alpha=0.001, random_state=42, max_iter=10000)
        raise ValidationError(f"Unsupported model type: {model_type}", context=self._context("build_model", model_type=model_type))

    def _default_param_grid(self, model_type: str) -> Dict[str, List[Any]]:
        if model_type == "gradientboosting":
            return {"model__n_estimators": [100, 200], "model__max_depth": [2, 3], "model__learning_rate": [0.03, 0.05, 0.1]}
        if model_type == "randomforest":
            return {"model__n_estimators": [100, 200], "model__max_depth": [4, 6, None], "model__min_samples_leaf": [1, 2, 4]}
        if model_type == "ridge":
            return {"model__alpha": [0.1, 1.0, 5.0, 10.0]}
        if model_type == "lasso":
            return {"model__alpha": [0.0001, 0.001, 0.01, 0.1]}
        return {}

    def _evaluate_candidate(self, model_type: str, X: pd.DataFrame, y: pd.Series, params: Mapping[str, Any]) -> float:
        model = self._build_model(model_type)
        for key, value in params.items():
            if hasattr(model, key):
                setattr(model, key, value)
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
        split_count = min(4, max(2, len(X) // 30))
        cv = TimeSeriesSplit(n_splits=split_count)
        losses: List[float] = []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            losses.append(float(mean_absolute_error(y_test, preds)))
        return float(np.mean(losses)) if losses else float("inf")

    def _train_model(self, train_data: pd.DataFrame) -> None:
        if train_data is None or train_data.empty:
            raise DataUnavailableError("Training data is empty.", context=self._context("train_model"))

        model_config = self.bt_config.get("model", {})
        target_col = model_config.get("target", "next_close")
        feature_mode = str(model_config.get("feature_mode", "auto")).lower()
        model_type = str(model_config.get("type", "GradientBoosting"))

        data = self._add_features(train_data.copy()).dropna(subset=["close", target_col])
        if data.empty:
            raise FeatureEngineeringError("Training data is empty after feature engineering.", context=self._context("train_model", target_col=target_col))

        exclude = {target_col, "symbol", "datetime", "date", "timestamp"}
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        feature_names = [col for col in numeric_cols if col not in exclude]
        if not feature_names:
            raise FeatureEngineeringError("No numeric features available for training.", context=self._context("train_model"))
        X = data[feature_names].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        y = pd.to_numeric(data[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid_mask = y.notna()
        if valid_mask.sum() < 20:
            raise FeatureEngineeringError("Insufficient finite target values for training.", context=self._context("train_model"))
        X = X.loc[valid_mask].copy()
        y = y.loc[valid_mask].copy()

        if feature_mode == "auto" and X.shape[1] > 1:
            selector = SelectKBest(score_func=f_regression, k=min(15, X.shape[1]))
            try:
                X_selected = selector.fit_transform(X, y)
                selected_indices = selector.get_support(indices=True)
                feature_names = [feature_names[i] for i in selected_indices]
                X = pd.DataFrame(X_selected, columns=feature_names, index=X.index)
            except Exception as exc:
                logger.warning("Feature selection failed, using all numeric features: %s", exc)

        scaler_cls = RobustScaler if model_type.lower() in {"gradientboosting", "randomforest"} else StandardScaler
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler_cls()),
            ("model", self._build_model(model_type)),
        ])

        if bool(self.bt_config.get("hyperparameter_tuning", True)):
            tuning_strategy = str(self.bt_config.get("tuning_strategy", "bayesian")).lower()
            try:
                if tuning_strategy == "grid":
                    grid = GridSearchCV(
                        estimator=pipeline,
                        param_grid=self._default_param_grid(model_type.lower()),
                        cv=TimeSeriesSplit(n_splits=min(5, max(2, len(X) // 20))),
                        scoring="neg_mean_absolute_error",
                        n_jobs=1,
                    )
                    grid.fit(X, y)
                    pipeline = grid.best_estimator_
                elif tuning_strategy == "bayesian":
                    tuner = HyperparamTuner(
                        model_type=model_type,
                        evaluation_function=lambda params, *_args: self._evaluate_candidate(model_type, X, y, params),
                    )
                    best_params = tuner.run_tuning_pipeline(X_data=X, y_data=y)
                    if isinstance(best_params, Mapping):
                        pipeline.set_params(**{f"model__{k}": v for k, v in best_params.items()})
            except Exception as exc:
                logger.warning("Hyperparameter tuning failed, using default model: %s", exc)

        try:
            pipeline.fit(X, y)
        except Exception as exc:
            handled = ModelTrainingError("Backtest model training failed.", context=self._context("train_model", target_col=target_col, model_type=model_type), cause=exc)
            log_error(handled, logger_=logger)
            raise handled from exc

        self.model = pipeline
        self.feature_columns = list(feature_names)
        self.feature_processor = {"feature_names": list(feature_names), "target_col": target_col, "model_type": model_type}

    def _prediction_based_signal(self, point: pd.Series, threshold: float) -> int:
        if self.model is None or not self.feature_columns:
            return 0
        values = []
        for feature in self.feature_columns:
            item = point.get(feature, 0.0)
            try:
                numeric = float(0.0 if pd.isna(item) else item)
            except Exception:
                numeric = 0.0
            values.append(numeric)
        X = pd.DataFrame([values], columns=self.feature_columns).reindex(columns=self.feature_columns)
        try:
            prediction = float(self.model.predict(X)[0])
        except Exception as exc:
            try:
                # Fallback path for models that were fitted with ndarray inputs or
                # feature-name metadata that no longer aligns with runtime DataFrames.
                prediction = float(self.model.predict(X.to_numpy())[0])
            except Exception:
                logger.warning("Prediction failed: %s", exc)
                return 0
        current = float(point["close"])
        point["prediction"] = prediction
        change = (prediction - current) / max(current, 1e-12)
        if change > threshold:
            return 1
        if change < -threshold:
            return -1
        return 0

    def _momentum_signal(self, point: pd.Series, test_data: pd.DataFrame, idx: int) -> int:
        lookback = int(self.bt_config.get("momentum_lookback", 14))
        recent_data = test_data.iloc[max(0, idx - lookback):idx]
        if len(recent_data) < 5:
            return 0
        roc = (float(point["close"]) - float(recent_data.iloc[0]["close"])) / max(float(recent_data.iloc[0]["close"]), 1e-12)
        ema_short = float(recent_data["close"].ewm(span=5).mean().iloc[-1])
        ema_long = float(recent_data["close"].ewm(span=20).mean().iloc[-1])
        if roc > 0.05 and ema_short > ema_long:
            return 1
        if roc < -0.05 and ema_short < ema_long:
            return -1
        return 0

    def _mean_reversion_signal(self, point: pd.Series, test_data: pd.DataFrame, idx: int) -> int:
        lookback = int(self.bt_config.get("mean_reversion_lookback", 20))
        z_threshold = float(self.bt_config.get("z_threshold", 1.5))
        recent_data = test_data.iloc[max(0, idx - lookback):idx]
        if len(recent_data) < 10:
            return 0
        mean = float(recent_data["close"].mean())
        std = float(recent_data["close"].std())
        if std < 1e-12:
            return 0
        z_score = (float(point["close"]) - mean) / std
        if z_score < -z_threshold:
            return 1
        if z_score > z_threshold:
            return -1
        return 0

    def _mark_to_market(self, positions: List[Position], current_price: float) -> float:
        total = 0.0
        for pos in positions:
            if pos.direction == "long":
                total += pos.quantity * current_price
            else:
                total += pos.entry_value * (1.0 + pos.pnl_ratio(current_price))
        return float(total)

    def _close_position(self, position: Position, price: float, reason: str, as_of: Any) -> float:
        pnl_ratio = position.pnl_ratio(price)
        gross_value = position.entry_value * (1.0 + pnl_ratio)
        net_value = gross_value * (1.0 - self.transaction_cost)
        self.trade_history.append({
            "symbol": position.symbol,
            "date": as_of,
            "direction": f"close_{position.direction}",
            "price": float(price),
            "entry_price": float(position.entry_price),
            "quantity": float(position.quantity),
            "pl": float(pnl_ratio),
            "gross_value": float(gross_value),
            "net_value": float(net_value),
            "reason": reason,
            "size": float(position.entry_value),
        })
        return float(net_value)

    def _execute_trade(self, point: pd.Series, signal: int, portfolio_value: float, positions: List[Position]) -> float:
        current_price = float(point["close"])
        symbol = str(point.get("symbol", "TEST")).upper()
        as_of = point.get("datetime", point.name)

        market_value = self._mark_to_market(positions, current_price)
        cash = float(portfolio_value - market_value)

        for pos in positions[:]:
            pnl_ratio = pos.pnl_ratio(current_price)
            reverse_signal = (signal == -1 and pos.direction == "long") or (signal == 1 and pos.direction == "short")
            risk_exit = pnl_ratio <= -self.stop_loss or pnl_ratio >= self.take_profit
            if reverse_signal or risk_exit:
                reason = "signal_flip" if reverse_signal else ("stop_loss" if pnl_ratio <= -self.stop_loss else "take_profit")
                cash += self._close_position(pos, current_price, reason, as_of)
                positions.remove(pos)

        trade_size = float(portfolio_value * self.position_size)
        if signal != 0 and cash > trade_size * (1.0 + self.transaction_cost):
            quantity = trade_size / max(current_price, 1e-12)
            direction = "long" if signal == 1 else "short"
            positions.append(Position(symbol=symbol, direction=direction, entry_price=current_price, quantity=quantity, entry_value=trade_size, entry_time=str(as_of)))
            cash -= trade_size * (1.0 + self.transaction_cost)
            self.trade_history.append({"symbol": symbol, "date": as_of, "direction": direction, "price": current_price, "quantity": quantity, "size": trade_size, "reason": "signal_entry"})

        return float(cash + self._mark_to_market(positions, current_price))

    def _calculate_max_drawdown(self, values: np.ndarray) -> float:
        peaks = np.maximum.accumulate(values)
        drawdowns = (peaks - values) / np.maximum(peaks, 1e-12)
        return float(np.max(drawdowns)) if len(drawdowns) else 0.0

    def _calculate_win_rate(self) -> float:
        completed = [trade for trade in self.trade_history if "pl" in trade]
        if not completed:
            return 0.0
        winners = sum(1 for trade in completed if float(trade["pl"]) > 0)
        return float(winners / len(completed))

    def _calculate_profit_factor(self) -> float:
        completed = [trade for trade in self.trade_history if "pl" in trade]
        gross_profit = sum(max(float(trade["pl"]), 0.0) for trade in completed)
        gross_loss = abs(sum(min(float(trade["pl"]), 0.0) for trade in completed))
        if gross_loss <= 1e-12:
            return float(gross_profit if gross_profit > 0 else 0.0)
        return float(gross_profit / gross_loss)

    def _calculate_kelly(self, win_rate: float, profit_factor: float) -> float:
        if win_rate <= 0 or profit_factor <= 0:
            return 0.0
        kelly = win_rate - ((1.0 - win_rate) / max(profit_factor, 1e-12))
        return float(max(0.0, min(kelly, 1.0)))

    def _compute_metrics(self, predictions: List[float], actuals: List[float], signals: List[int], portfolio_values: List[float]) -> Dict[str, float]:
        if len(predictions) < 2 or len(actuals) < 2 or len(portfolio_values) < 2:
            return {}
        pred_changes = np.diff(np.asarray(predictions, dtype=float)) > 0
        actual_changes = np.diff(np.asarray(actuals, dtype=float)) > 0
        accuracy = float(np.mean(pred_changes == actual_changes)) if len(actual_changes) else 0.0

        actionable = np.asarray(signals[:-1], dtype=int)
        positive_truth = actual_changes
        negative_truth = ~actual_changes
        true_positives = int(np.sum((actionable == 1) & positive_truth))
        false_positives = int(np.sum((actionable == 1) & negative_truth))
        false_negatives = int(np.sum((actionable != 1) & positive_truth))
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

        portfolio_arr = np.asarray(portfolio_values, dtype=float)
        returns = np.diff(portfolio_arr) / np.maximum(portfolio_arr[:-1], 1e-12)
        mean_return = float(np.mean(returns)) if len(returns) else 0.0
        volatility = float(np.std(returns)) if len(returns) else 0.0
        downside = returns[returns < 0]
        downside_dev = float(np.std(downside)) if len(downside) else 0.0
        sharpe = mean_return / volatility if volatility > 1e-12 else 0.0
        sortino = mean_return / downside_dev if downside_dev > 1e-12 else 0.0
        max_drawdown = self._calculate_max_drawdown(portfolio_arr)
        calmar = mean_return / max(max_drawdown, 1e-12) if max_drawdown > 0 else 0.0
        win_rate = self._calculate_win_rate()
        profit_factor = self._calculate_profit_factor()
        kelly = self._calculate_kelly(win_rate, profit_factor)
        mae = float(mean_absolute_error(np.asarray(actuals, dtype=float), np.asarray(predictions, dtype=float)))

        result = {
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "max_drawdown": float(max_drawdown),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "mean_return": float(mean_return),
            "volatility": float(volatility),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "kelly_criterion": float(kelly),
            "mean_absolute_error": float(mae),
        }
        for key, value in result.items():
            self.metrics[key].append(float(value))
        return result

    def _aggregate_metrics(self) -> Dict[str, float]:
        aggregated = {key: float(np.mean(values)) if values else 0.0 for key, values in self.metrics.items()}
        if self.portfolio_history:
            aggregated["initial_portfolio_value"] = float(self.portfolio_history[0]["value"])
            aggregated["final_portfolio_value"] = float(self.portfolio_history[-1]["value"])
            aggregated["total_return"] = (aggregated["final_portfolio_value"] - aggregated["initial_portfolio_value"]) / max(aggregated["initial_portfolio_value"], 1e-12)
        else:
            aggregated["initial_portfolio_value"] = 0.0
            aggregated["final_portfolio_value"] = 0.0
            aggregated["total_return"] = 0.0
        aggregated["trade_count"] = float(len(self.trade_history))
        aggregated["strategy"] = self.strategy
        return aggregated

    def _store_backtest_results(self, metrics: Dict[str, float]) -> None:
        self._safe_memory_add(data=metrics, data_type="backtest_metrics", tags=[f"backtest_{self.strategy}", "results", "backtesting"], priority="high", metadata={"strategy": self.strategy})
        self._safe_memory_add(data={"trades": list(self.trade_history), "portfolio": list(self.portfolio_history), "strategy": self.strategy, "timestamp": time.time(), "windows": [window.to_dict() for window in self.window_results]}, data_type="trade_history", tags=[f"backtest_{self.strategy}", "history", "backtesting"], priority="medium", metadata={"strategy": self.strategy})
        self._last_aggregated_metrics = dict(metrics)

    def walk_forward_test(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        data = self._prepare_historical_data(historical_data)
        symbol = str(data["symbol"].mode().iloc[0])
        data = data[data["symbol"].astype(str).str.upper() == symbol.upper()].copy().sort_values("datetime").reset_index(drop=True)
        data = self._add_features(data)
        if len(data) < self.train_window + self.test_window + 5:
            raise DataUnavailableError("Historical data is insufficient for configured windows.", context=self._context("walk_forward_test", symbol=symbol, rows=len(data)))

        self.portfolio_history = [{"step": 0, "value": self.initial_capital, "symbol": symbol}]
        self.trade_history = []
        self.window_results = []
        for values in self.metrics.values():
            values.clear()

        prediction_threshold = float(self.bt_config.get("prediction_threshold", 0.02))
        portfolio_value = float(self.initial_capital)
        step_counter = 0

        for start in range(0, len(data) - self.train_window - self.test_window + 1, self.test_window):
            train_data = data.iloc[start:start + self.train_window].copy()
            test_data = data.iloc[start + self.train_window:start + self.train_window + self.test_window].copy()
            if train_data.empty or test_data.empty:
                continue
            self._train_model(train_data)

            predictions: List[float] = []
            actuals: List[float] = []
            signals: List[int] = []
            positions: List[Position] = []
            test_portfolio_values: List[float] = [portfolio_value]

            for idx, (_, point) in enumerate(test_data.iterrows()):
                point = point.copy()
                if self.strategy == "prediction_based":
                    signal = self._prediction_based_signal(point, prediction_threshold)
                elif self.strategy == "momentum":
                    signal = self._momentum_signal(point, test_data, idx)
                else:
                    signal = self._mean_reversion_signal(point, test_data, idx)
                signals.append(int(signal))
                portfolio_value = self._execute_trade(point, int(signal), portfolio_value, positions)
                predictions.append(float(point.get("prediction", point["close"])))
                actuals.append(float(point["close"]))
                step_counter += 1
                self.portfolio_history.append({"step": step_counter, "value": float(portfolio_value), "symbol": symbol, "datetime": point.get("datetime")})
                test_portfolio_values.append(float(portfolio_value))

            if positions:
                final_price = float(test_data.iloc[-1]["close"])
                final_dt = test_data.iloc[-1].get("datetime")
                cash = portfolio_value - self._mark_to_market(positions, final_price)
                for pos in positions[:]:
                    cash += self._close_position(pos, final_price, "window_end", final_dt)
                    positions.remove(pos)
                portfolio_value = float(cash)
                self.portfolio_history.append({"step": step_counter + 1, "value": portfolio_value, "symbol": symbol, "datetime": final_dt})
                test_portfolio_values.append(portfolio_value)

            window_metrics = self._compute_metrics(predictions, actuals, signals, test_portfolio_values)
            self.window_results.append(WindowResult(
                window_index=len(self.window_results) + 1,
                train_start=str(train_data["datetime"].iloc[0]),
                train_end=str(train_data["datetime"].iloc[-1]),
                test_start=str(test_data["datetime"].iloc[0]),
                test_end=str(test_data["datetime"].iloc[-1]),
                portfolio_value_start=float(test_portfolio_values[0]),
                portfolio_value_end=float(test_portfolio_values[-1]),
                trade_count=len(self.trade_history),
                metrics=window_metrics,
            ))
            logger.info("Completed window %s | symbol=%s | portfolio=%.4f", len(self.window_results), symbol, portfolio_value)

        metrics = self._aggregate_metrics()
        if metrics:
            self._store_backtest_results(metrics)
        return metrics

    def run_environment_backtest(self, env: Any, max_steps: Optional[int] = None) -> Dict[str, float]:
        if env is None:
            raise ValidationError("env is required.", context=self._context("run_environment_backtest"))
        observation = env.reset()
        done = False
        steps = 0
        max_steps = int(max_steps or getattr(env, "max_steps", 1000))
        while not done and steps < max_steps:
            try:
                action = self._policy_action_from_observation(observation)
                observation, reward, done, info = env.step(action)
            except Exception as exc:
                handled = classify_external_exception(exc, context=self._context("run_environment_backtest", step=steps), message="Trading environment step failed during backtest.")
                log_error(handled, logger_=logger)
                raise handled from exc
            steps += 1
        metrics = env.get_performance_metrics()
        if isinstance(metrics, Mapping):
            self.store_last_metrics(dict(metrics))
            self._safe_memory_add(data=dict(metrics), data_type="backtest_metrics", tags=[f"backtest_env_{getattr(env, 'symbol', 'unknown')}", "environment"], priority="high", metadata={"symbol": getattr(env, "symbol", None)})
        return dict(metrics)

    def _policy_action_from_observation(self, observation: np.ndarray) -> int:
        if observation is None or len(observation) == 0:
            return 0
        momentum = float(np.mean(observation[:min(5, len(observation))]))
        if momentum > 0.5:
            return 2
        if momentum > 0.1:
            return 1
        if momentum < -0.5:
            return 4
        if momentum < -0.1:
            return 3
        return 0

    def plot_equity_curve(self, *, show: bool = False, save_path: Optional[str] = None) -> Optional[str]:
        if not self.portfolio_history:
            logger.warning("No portfolio history available to plot.")
            return None
        df = pd.DataFrame(self.portfolio_history)
        plt.figure(figsize=(10, 5))
        plt.plot(df["value"].values)
        plt.title("Backtest Equity Curve")
        plt.xlabel("Step")
        plt.ylabel("Portfolio Value")
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            return save_path
        if show:
            plt.show()
        else:
            plt.close()
        return None


if __name__ == "__main__":  # pragma: no cover
    from .investor_tracker import InvestorTracker

    rate = 0.02
    symbol = "SPY"

    bt = Backtester(tracker=InvestorTracker(risk_free_rate=rate, benchmark_symbol=symbol))
    print("Backtester initialized:", bt.strategy)
