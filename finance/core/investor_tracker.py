from __future__ import annotations

import math
import os
import statistics
import time
import numpy as np

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from finance.core.utils.financial_errors import (log_error, DataUnavailableError, ErrorContext,
                                                 FinancialAgentError, ValidationError,
                                                 classify_external_exception)
from finance.core.utils.public_sentiment_scraper import PublicSentimentScraper
from finance.core.finance_memory import FinanceMemory
from finance.core.market_data_handler import MarketDataHandler
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Investor Tracker")
printer = PrettyPrinter

DEFAULT_LOOKBACK_DAYS = 252
DEFAULT_BENCHMARK = "SPY"
DEFAULT_TRACKED_SYMBOLS = ["SPY", "QQQ", "TLT", "GLD"]
REGIME_VOL_THRESHOLD = 0.28
BEAR_RETURN_THRESHOLD = -0.05
MAX_RANDOM_PORTFOLIOS = 2500


@dataclass(frozen=True)
class PortfolioOptimizationReport:
    investor_id: str
    regime: str
    method_used: str
    weights: Dict[str, float]
    expected_annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    cvar_95: float
    generated_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EfficientCVaR:
    """
    Lightweight CVaR optimizer using deterministic random search.

    This avoids heavy solver dependencies while still producing stable,
    production-usable allocations for long-only portfolios.
    """

    def __init__(self, returns: np.ndarray, beta: float = 0.95, random_seed: int = 42) -> None:
        returns = np.asarray(returns, dtype=float)
        if returns.ndim != 2 or returns.shape[0] < 20 or returns.shape[1] < 1:
            raise ValidationError("EfficientCVaR requires a 2D returns matrix with at least 20 rows.")
        if not (0.50 < beta < 0.999):
            raise ValidationError("beta must be between 0.50 and 0.999.")
        self.returns = returns
        self.beta = float(beta)
        self.rng = np.random.default_rng(random_seed)

    def _portfolio_cvar(self, weights: np.ndarray) -> float:
        portfolio_returns = self.returns @ weights
        cutoff = np.quantile(portfolio_returns, 1.0 - self.beta)
        tail = portfolio_returns[portfolio_returns <= cutoff]
        if tail.size == 0:
            return 0.0
        return float(-tail.mean())

    def min_cvar(self, labels: Optional[Sequence[str]] = None, n_portfolios: int = MAX_RANDOM_PORTFOLIOS) -> Dict[str, float]:
        n_assets = self.returns.shape[1]
        labels = list(labels or [f"asset_{idx}" for idx in range(n_assets)])
        best_weights = np.ones(n_assets) / n_assets
        best_score = self._portfolio_cvar(best_weights)

        for _ in range(max(250, int(n_portfolios))):
            candidate = self.rng.dirichlet(np.ones(n_assets))
            score = self._portfolio_cvar(candidate)
            if score < best_score:
                best_weights = candidate
                best_score = score

        return {label: float(weight) for label, weight in zip(labels, best_weights)}


class InvestorTracker:
    """Production-ready portfolio tracker and allocator.

    Responsibilities:
    - maintain investor holdings and derived metrics
    - fetch and align historical market data through MarketDataHandler
    - detect broad market regime deterministically
    - optimize allocations using consistent portfolio methods
    - calculate technical indicators, VaR, and portfolio health metrics
    - cache intermediate results into FinanceMemory
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        market_data_handler: Optional[MarketDataHandler] = None,
        memory: Optional[FinanceMemory] = None,
        sentiment_scraper: Optional[Any] = None,
        risk_free_rate: float = 0.02,
        benchmark_symbol: str = DEFAULT_BENCHMARK,
    ) -> None:
        self.api_key = api_key
        self.risk_free_rate = float(risk_free_rate)
        self.benchmark_symbol = benchmark_symbol.upper()
        self.memory = memory if memory is not None else self._safe_build_memory()
        self.market_data_handler = market_data_handler or MarketDataHandler(memory=self.memory)
        self.sentiment_scraper = sentiment_scraper
        if self.sentiment_scraper is None and PublicSentimentScraper is not None:
            try:
                self.sentiment_scraper = PublicSentimentScraper()
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.warning("Sentiment scraper unavailable for InvestorTracker: %s", exc)

        self.investor_db: Dict[str, Dict[str, Any]] = {}
        self.portfolio_metrics: Dict[str, Dict[str, Any]] = {}
        self.tracked_symbols: List[str] = []
        self.last_symbol_update = 0.0
        self._refresh_symbols()

        if printer is not None:  # pragma: no cover - presentation only
            printer.status("INIT", "InvestorTracker initialized", "success")

    def _safe_build_memory(self) -> Optional[FinanceMemory]:
        try:
            return FinanceMemory()
        except Exception as exc:  # pragma: no cover - depends on runtime wiring
            logger.warning("FinanceMemory unavailable for InvestorTracker: %s", exc)
            return None

    def _context(self, operation: str, *, investor_id: Optional[str] = None, symbol: Optional[str] = None, extra: Optional[Mapping[str, Any]] = None) -> ErrorContext:
        return ErrorContext(
            component="investor_tracker",
            operation=operation,
            symbol=symbol,
            metadata={"investor_id": investor_id, **dict(extra or {})},
        )

    def _cache_get(self, namespace: str, key: str) -> Any:
        if self.memory is None:
            return None
        try:
            return self.memory.get_cache(key, namespace=namespace)
        except Exception:
            return None

    def _cache_set(self, namespace: str, key: str, value: Any, *, tags: Optional[List[str]] = None, priority: str = "medium") -> None:
        if self.memory is None:
            return
        try:
            self.memory.set_cache(
                key,
                value,
                namespace=namespace,
                ttl_seconds=300,
                tags=tags or [],
                priority=priority,
            )
        except Exception as exc:
            logger.debug("Failed to cache InvestorTracker value: %s", exc)

    def _refresh_symbols(self) -> None:
        if time.time() - self.last_symbol_update < 86400 and self.tracked_symbols:
            return
        try:
            symbols = self.market_data_handler.get_most_active_symbols(count=250)
            normalized = [str(symbol).upper() for symbol in symbols if isinstance(symbol, str)]
            normalized = [symbol for symbol in normalized if symbol]
            self.tracked_symbols = sorted(set(normalized or DEFAULT_TRACKED_SYMBOLS))
            self.last_symbol_update = time.time()
            logger.info("Tracked symbol universe updated: %s symbols", len(self.tracked_symbols))
        except Exception as exc:
            handled = exc if isinstance(exc, FinancialAgentError) else classify_external_exception(
                exc,
                context=self._context("refresh_symbols"),
                message="Failed to refresh tracked symbols.",
            )
            log_error(handled, logger_=logger, include_traceback=False)
            if not self.tracked_symbols:
                self.tracked_symbols = list(DEFAULT_TRACKED_SYMBOLS)

    def _validate_investor_id(self, investor_id: str) -> str:
        cleaned = (investor_id or "").strip()
        if not cleaned:
            raise ValidationError(
                "investor_id must be a non-empty string.",
                context=self._context("validate_investor_id"),
            )
        return cleaned

    def _normalize_assets(self, assets: Mapping[str, Any]) -> Dict[str, float]:
        normalized: Dict[str, float] = {}
        for symbol, quantity in assets.items():
            clean_symbol = str(symbol).strip().upper()
            qty = float(quantity)
            if abs(qty) < 1e-12:
                continue
            normalized[clean_symbol] = qty
        return normalized

    def get_portfolio(self, investor_id: str) -> Dict[str, Any]:
        investor_id = self._validate_investor_id(investor_id)
        portfolio = self.investor_db.get(investor_id)
        if portfolio is None:
            raise ValidationError(
                f"No portfolio found for investor {investor_id}.",
                context=self._context("get_portfolio", investor_id=investor_id),
            )
        return portfolio

    def update_portfolio(self, investor_id: str, transactions: Mapping[str, Any], *, auto_optimize: bool = True) -> Dict[str, Any]:
        investor_id = self._validate_investor_id(investor_id)
        if not isinstance(transactions, Mapping) or not transactions:
            raise ValidationError(
                "transactions must be a non-empty mapping of symbol -> quantity delta.",
                context=self._context("update_portfolio", investor_id=investor_id),
            )

        portfolio = self.investor_db.get(investor_id, {"assets": {}, "updated_at": None})
        assets = self._normalize_assets(portfolio.get("assets", {}))

        for symbol, delta in transactions.items():
            clean_symbol = str(symbol).strip().upper()
            new_quantity = float(assets.get(clean_symbol, 0.0)) + float(delta)
            if abs(new_quantity) < 1e-12:
                assets.pop(clean_symbol, None)
            else:
                assets[clean_symbol] = new_quantity

        portfolio["assets"] = assets
        portfolio["updated_at"] = time.time()
        self.investor_db[investor_id] = portfolio

        if self.memory is not None:
            try:
                self.memory.add_financial_data(
                    data={"investor_id": investor_id, "assets": assets, "transactions": dict(transactions)},
                    data_type="portfolio",
                    tags=["portfolio", f"investor_{investor_id}"],
                    priority="high",
                    metadata={"investor_id": investor_id},
                )
            except Exception as exc:
                logger.debug("Failed to persist portfolio update: %s", exc)

        if auto_optimize and assets:
            try:
                self.optimize_portfolio(investor_id)
            except Exception as exc:
                logger.warning("Auto-optimization failed for investor %s: %s", investor_id, exc)

        return portfolio

    def _expected_return(self, returns: np.ndarray) -> np.ndarray:
        return np.mean(returns, axis=0) * 252.0

    def _covariance_matrix(self, returns: np.ndarray) -> np.ndarray:
        return np.cov(returns, rowvar=False) * 252.0

    def _portfolio_volatility(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        return float(np.sqrt(max(weights.T @ cov_matrix @ weights, 0.0)))

    def _portfolio_sharpe(self, weights: np.ndarray, returns: np.ndarray, cov_matrix: np.ndarray) -> float:
        expected = float(np.dot(self._expected_return(returns), weights))
        volatility = self._portfolio_volatility(weights, cov_matrix)
        return float((expected - self.risk_free_rate) / (volatility + 1e-12))

    def _run_random_portfolios(self, returns: np.ndarray, n_portfolios: int = MAX_RANDOM_PORTFOLIOS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_assets = returns.shape[1]
        rng = np.random.default_rng(42)
        weights = rng.dirichlet(np.ones(n_assets), size=max(250, int(n_portfolios)))
        cov = self._covariance_matrix(returns)
        expected = self._expected_return(returns)
        portfolio_returns = weights @ expected
        portfolio_vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
        sharpes = (portfolio_returns - self.risk_free_rate) / (portfolio_vols + 1e-12)
        return weights, portfolio_returns, sharpes

    def _mean_variance_weights(self, returns: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
        weights, _, sharpes = self._run_random_portfolios(returns)
        best = weights[int(np.nanargmax(sharpes))]
        return {label: float(weight) for label, weight in zip(labels, best)}

    def _inverse_volatility_weights(self, returns: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
        vols = np.std(returns, axis=0, ddof=1)
        inv = 1.0 / np.clip(vols, 1e-8, None)
        inv = inv / inv.sum()
        return {label: float(weight) for label, weight in zip(labels, inv)}

    def _calculate_historical_returns(self, assets: Iterable[str], lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> Tuple[List[str], Optional[np.ndarray]]:
        labels = [str(asset).upper() for asset in assets if str(asset).strip()]
        if not labels:
            return [], None
        cache_key = f"returns:{','.join(sorted(labels))}:{lookback_days}"
        cached = self._cache_get("portfolio_returns", cache_key)
        if cached:
            return cached["labels"], np.asarray(cached["matrix"], dtype=float)

        symbols, matrix = self.market_data_handler.get_returns_matrix(labels, lookback=lookback_days)
        if not symbols or not matrix:
            return [], None
        returns = np.asarray(matrix, dtype=float)
        if returns.ndim != 2 or returns.shape[0] < 20:
            return [], None
        self._cache_set(
            "portfolio_returns",
            cache_key,
            {"labels": symbols, "matrix": returns.tolist()},
            tags=["returns", *[label.lower() for label in symbols]],
            priority="high",
        )
        return symbols, returns

    def _market_volatility(self, lookback_days: int = 63) -> float:
        _, returns = self._calculate_historical_returns([self.benchmark_symbol], lookback_days=lookback_days)
        if returns is None or returns.size == 0:
            return 0.0
        return float(np.std(returns[:, 0], ddof=1) * np.sqrt(252.0))

    def _detect_market_regime(self, current_regime: str = "unknown", n_states: int = 3, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> str:  # noqa: ARG002
        _ = current_regime
        symbols, returns = self._calculate_historical_returns([self.benchmark_symbol], lookback_days=lookback_days)
        if returns is None or not symbols:
            return "neutral"
        series = returns[:, 0]
        annual_vol = float(np.std(series, ddof=1) * np.sqrt(252.0))
        trailing_20 = float(np.prod(1.0 + series[-20:]) - 1.0) if series.size >= 20 else 0.0
        trailing_60 = float(np.prod(1.0 + series[-60:]) - 1.0) if series.size >= 60 else 0.0
        if annual_vol >= REGIME_VOL_THRESHOLD:
            return "high_volatility"
        if trailing_60 <= BEAR_RETURN_THRESHOLD or (trailing_20 < 0 and trailing_60 < 0):
            return "bear"
        if trailing_20 > 0 and trailing_60 > 0:
            return "bull"
        return "neutral"

    def _score_assets(self, labels: Sequence[str], returns: np.ndarray) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for idx, label in enumerate(labels):
            series = returns[:, idx]
            momentum_20 = float(np.prod(1.0 + series[-20:]) - 1.0) if len(series) >= 20 else 0.0
            volatility = float(np.std(series[-63:], ddof=1) * np.sqrt(252.0)) if len(series) >= 30 else 0.2
            sentiment_adjustment = 0.0
            if self.sentiment_scraper is not None:
                try:
                    sentiment_adjustment = float(self.sentiment_scraper.compute_average_sentiment(label, max_results=6)) * 0.05
                except Exception:
                    sentiment_adjustment = 0.0
            scores[label] = momentum_20 - volatility * 0.25 + sentiment_adjustment
        return scores

    def _adjust_for_regime(self, weights: Dict[str, float], regime: str, returns: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
        if not weights:
            return weights
        adjusted = dict(weights)
        scores = self._score_assets(labels, returns)
        if regime == "bull":
            for label in adjusted:
                adjusted[label] *= max(0.8, 1.0 + scores.get(label, 0.0))
        elif regime == "bear":
            for label in adjusted:
                adjusted[label] *= max(0.5, 1.0 - max(scores.get(label, 0.0), 0.0))
        elif regime == "high_volatility":
            vol_weights = self._inverse_volatility_weights(returns, labels)
            for label in adjusted:
                adjusted[label] = 0.5 * adjusted[label] + 0.5 * vol_weights.get(label, 0.0)
        total = sum(max(weight, 0.0) for weight in adjusted.values())
        if total <= 0:
            equal = 1.0 / len(adjusted)
            return {label: equal for label in adjusted}
        return {label: float(max(weight, 0.0) / total) for label, weight in adjusted.items()}

    def _aggressive_allocation(self, labels: Sequence[str], returns: np.ndarray, alpha: float) -> Dict[str, float]:
        base = self._mean_variance_weights(returns, labels)
        return self._adjust_for_regime(base, "bull", returns, labels)

    def _defensive_allocation(self, labels: Sequence[str], returns: np.ndarray, alpha: float) -> Dict[str, float]:  # noqa: ARG002
        base = EfficientCVaR(returns=returns, beta=min(max(alpha, 0.80), 0.99)).min_cvar(labels=labels)
        return self._adjust_for_regime(base, "bear", returns, labels)

    def _risk_parity_allocation(self, labels: Sequence[str], returns: np.ndarray, alpha: float) -> Dict[str, float]:  # noqa: ARG002
        base = self._inverse_volatility_weights(returns, labels)
        return self._adjust_for_regime(base, "high_volatility", returns, labels)

    def _balanced_allocation(self, labels: Sequence[str], returns: np.ndarray, alpha: float) -> Dict[str, float]:
        cvar_weights = EfficientCVaR(returns=returns, beta=min(max(alpha, 0.80), 0.99)).min_cvar(labels=labels)
        mean_var = self._mean_variance_weights(returns, labels)
        blended = {label: 0.5 * cvar_weights.get(label, 0.0) + 0.5 * mean_var.get(label, 0.0) for label in labels}
        return self._adjust_for_regime(blended, "neutral", returns, labels)

    def optimize_portfolio(self, investor_id: str, method: str = "auto", alpha: float = 0.95) -> Dict[str, float]:
        investor_id = self._validate_investor_id(investor_id)
        portfolio = self.get_portfolio(investor_id)
        assets = portfolio.get("assets", {})
        if not assets:
            raise ValidationError(
                f"No portfolio found for investor {investor_id}.",
                context=self._context("optimize_portfolio", investor_id=investor_id),
            )

        labels, returns = self._calculate_historical_returns(assets.keys())
        if returns is None or not labels:
            raise DataUnavailableError(
                f"Unable to compute historical returns for investor {investor_id}.",
                context=self._context("optimize_portfolio", investor_id=investor_id),
                details={"assets": list(assets)},
            )

        regime = self._detect_market_regime(lookback_days=max(DEFAULT_LOOKBACK_DAYS, returns.shape[0]))
        normalized_method = (method or "auto").strip().lower()
        if normalized_method == "auto":
            if regime == "bull":
                weights = self._aggressive_allocation(labels, returns, alpha)
                method_used = "aggressive"
            elif regime == "bear":
                weights = self._defensive_allocation(labels, returns, alpha)
                method_used = "defensive"
            elif regime == "high_volatility":
                weights = self._risk_parity_allocation(labels, returns, alpha)
                method_used = "risk_parity"
            else:
                weights = self._balanced_allocation(labels, returns, alpha)
                method_used = "balanced"
        elif normalized_method in {"cvar", "defensive"}:
            weights = self._defensive_allocation(labels, returns, alpha)
            method_used = "cvar"
        elif normalized_method in {"mean_variance", "mpt", "aggressive"}:
            weights = self._aggressive_allocation(labels, returns, alpha)
            method_used = "mean_variance"
        elif normalized_method in {"risk_parity", "inverse_volatility"}:
            weights = self._risk_parity_allocation(labels, returns, alpha)
            method_used = "risk_parity"
        elif normalized_method in {"balanced", "equal_weight"}:
            weights = self._balanced_allocation(labels, returns, alpha)
            method_used = "balanced"
        else:
            raise ValidationError(
                f"Unsupported optimization method '{method}'.",
                context=self._context("optimize_portfolio", investor_id=investor_id),
                details={"method": method},
            )

        ordered_weights = np.asarray([weights[label] for label in labels], dtype=float)
        cov = self._covariance_matrix(returns)
        expected_return = float(np.dot(self._expected_return(returns), ordered_weights))
        annual_volatility = self._portfolio_volatility(ordered_weights, cov)
        sharpe_ratio = self._portfolio_sharpe(ordered_weights, returns, cov)
        cvar_95 = float(EfficientCVaR(returns=returns, beta=0.95)._portfolio_cvar(ordered_weights))

        report = PortfolioOptimizationReport(
            investor_id=investor_id,
            regime=regime,
            method_used=method_used,
            weights=dict(weights),
            expected_annual_return=expected_return,
            annual_volatility=annual_volatility,
            sharpe_ratio=sharpe_ratio,
            cvar_95=cvar_95,
            generated_at=time.time(),
            metadata={"assets": labels},
        )
        self.portfolio_metrics[investor_id] = report.to_dict()

        if self.memory is not None:
            try:
                self.memory.add_financial_data(
                    data=report.to_dict(),
                    data_type="metric",
                    tags=["portfolio_optimization", f"investor_{investor_id}", f"regime_{regime}"],
                    priority="high",
                    metadata={"investor_id": investor_id},
                )
            except Exception as exc:
                logger.debug("Failed to persist optimization report: %s", exc)

        return report.weights

    def calculate_technical_indicators(self, symbol: str, window: int = 14) -> Optional[Dict[str, Any]]:
        symbol = str(symbol or "").strip().upper()
        window = max(5, int(window))
        if not symbol:
            raise ValidationError("symbol is required for technical indicator calculation.")

        series = self.market_data_handler.fetch_data(symbol, lookback=max(window * 4, 80), mode="fuse")
        if len(series) < window + 5:
            return None

        closes = np.asarray([float(row["close"]) for row in series], dtype=float)
        highs = np.asarray([float(row["high"]) for row in series], dtype=float)
        lows = np.asarray([float(row["low"]) for row in series], dtype=float)
        volumes = np.asarray([float(row["volume"]) for row in series], dtype=float)

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[:window]) if gains.size >= window else 0.0
        avg_loss = np.mean(losses[:window]) if losses.size >= window else 0.0
        rsi_values: List[float] = []
        for idx in range(window, len(gains)):
            avg_gain = ((avg_gain * (window - 1)) + gains[idx]) / window
            avg_loss = ((avg_loss * (window - 1)) + losses[idx]) / window
            rs = avg_gain / (avg_loss + 1e-12)
            rsi_values.append(float(100.0 - (100.0 / (1.0 + rs))))
        current_rsi = rsi_values[-1] if rsi_values else 50.0

        def ema(values: np.ndarray, span: int) -> np.ndarray:
            if values.size == 0:
                return np.asarray([], dtype=float)
            alpha = 2.0 / (span + 1.0)
            output = np.zeros_like(values, dtype=float)
            output[0] = values[0]
            for idx in range(1, len(values)):
                output[idx] = alpha * values[idx] + (1.0 - alpha) * output[idx - 1]
            return output

        ema12 = ema(closes, 12)
        ema26 = ema(closes, 26)
        macd_line = ema12 - ema26
        signal_line = ema(macd_line, 9)
        macd_histogram = macd_line - signal_line

        middle = np.convolve(closes, np.ones(window) / window, mode="valid")
        rolling_std = np.asarray([np.std(closes[idx - window:idx], ddof=0) for idx in range(window, len(closes) + 1)], dtype=float)
        upper = middle + (2.0 * rolling_std)
        lower = middle - (2.0 * rolling_std)
        typical_price = (highs + lows + closes) / 3.0
        cumulative_volume = np.cumsum(volumes)
        vwap = np.cumsum(typical_price * volumes) / np.clip(cumulative_volume, 1e-12, None)

        return {
            "symbol": symbol,
            "rsi": {
                "value": float(current_rsi),
                "series": rsi_values,
                "overbought": 70.0,
                "oversold": 30.0,
            },
            "macd": {
                "line": float(macd_line[-1]),
                "signal": float(signal_line[-1]),
                "histogram": float(macd_histogram[-1]),
                "trend": "bullish" if macd_histogram[-1] >= 0 else "bearish",
            },
            "bollinger": {
                "upper": float(upper[-1]),
                "middle": float(middle[-1]),
                "lower": float(lower[-1]),
                "bandwidth": float((upper[-1] - lower[-1]) / max(middle[-1], 1e-12)),
            },
            "vwap": float(vwap[-1]),
            "timestamp": time.time(),
        }

    def calculate_portfolio_var(self, investor_id: str, confidence_level: float = 0.99) -> float:
        investor_id = self._validate_investor_id(investor_id)
        if not (0.0 < confidence_level < 1.0):
            raise ValidationError(
                "confidence_level must be between 0 and 1.",
                context=self._context("calculate_portfolio_var", investor_id=investor_id),
            )
        portfolio = self.get_portfolio(investor_id)
        assets = portfolio.get("assets", {})
        if not assets:
            return 0.0

        labels, returns = self._calculate_historical_returns(assets.keys())
        if returns is None or not labels:
            raise DataUnavailableError(
                f"Unable to compute historical returns for investor {investor_id}.",
                context=self._context("calculate_portfolio_var", investor_id=investor_id),
            )

        quantities = np.asarray([float(assets[label]) for label in labels], dtype=float)
        magnitudes = np.abs(quantities)
        weights = magnitudes / np.clip(magnitudes.sum(), 1e-12, None)
        portfolio_returns = returns @ weights
        var = float(-np.quantile(portfolio_returns, 1.0 - confidence_level))
        return max(var, 0.0)

    def calculate_portfolio_metrics(self, investor_id: str) -> Dict[str, Any]:
        investor_id = self._validate_investor_id(investor_id)
        portfolio = self.get_portfolio(investor_id)
        assets = portfolio.get("assets", {})
        if not assets:
            return {"investor_id": investor_id, "assets": {}, "empty": True}
        labels, returns = self._calculate_historical_returns(assets.keys())
        if returns is None or not labels:
            raise DataUnavailableError(
                f"Unable to compute portfolio metrics for investor {investor_id}.",
                context=self._context("calculate_portfolio_metrics", investor_id=investor_id),
            )
        quantities = np.asarray([float(assets[label]) for label in labels], dtype=float)
        magnitudes = np.abs(quantities)
        weights = magnitudes / np.clip(magnitudes.sum(), 1e-12, None)
        cov = self._covariance_matrix(returns)
        portfolio_returns = returns @ weights
        metrics = {
            "investor_id": investor_id,
            "expected_annual_return": float(np.dot(self._expected_return(returns), weights)),
            "annual_volatility": self._portfolio_volatility(weights, cov),
            "sharpe_ratio": self._portfolio_sharpe(weights, returns, cov),
            "var_95": float(-np.quantile(portfolio_returns, 0.05)),
            "var_99": float(-np.quantile(portfolio_returns, 0.01)),
            "cvar_95": float(-portfolio_returns[portfolio_returns <= np.quantile(portfolio_returns, 0.05)].mean()),
            "positions": dict(assets),
            "weights": {label: float(weight) for label, weight in zip(labels, weights)},
            "regime": self._detect_market_regime(),
        }
        self.portfolio_metrics[investor_id] = metrics
        return metrics

    def get_insider_signals(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return grounded public signals when available.

        This method intentionally avoids fabricating insider data. When a
        sentiment scraper is available it returns recent headline-derived
        signals; otherwise it returns an empty list.
        """
        if symbol and self.sentiment_scraper is not None:
            try:
                snippets = self.sentiment_scraper.get_sentiment_snippets(symbol, max_results=6)
                signals: List[Dict[str, Any]] = []
                for source, items in snippets.items():
                    for item in items[:2]:
                        headline = item.get("title") or item.get("text") or item.get("snippet")
                        if headline:
                            signals.append(
                                {
                                    "symbol": symbol.upper(),
                                    "source": source,
                                    "headline": headline,
                                    "timestamp": item.get("timestamp") or item.get("published_at"),
                                }
                            )
                return signals
            except Exception as exc:
                logger.warning("Failed to build public signals for %s: %s", symbol, exc)
        return []


__all__ = [
    "EfficientCVaR",
    "InvestorTracker",
    "PortfolioOptimizationReport",
]


if __name__ == "__main__":
    print("--- Starting InvestorTracker Example Usage ---")

    # --- Setup ---
    # You might need to set dummy environment variables for API keys if your
    # actual data fetching methods require them, although the yfinance
    # placeholder typically doesn't need them configured this way.
    # os.environ["ALPHA_VANTAGE_KEY"] = "YOUR_ALPHA_VANTAGE_KEY"
    # os.environ["FINNHUB_KEY"] = "YOUR_FINNHUB_KEY"

    # Create an instance of InvestorTracker
    # If your __init__ requires shared_memory or get_shared_slailm,
    # you'll need to provide dummy objects for testing purposes.
    # Based on the latest code provided, it doesn't require them anymore.
    try:
        tracker = InvestorTracker()
        print("InvestorTracker instance created.")
    except Exception as e:
        print(f"ERROR: Failed to create InvestorTracker instance: {e}")
        print("Please ensure all dependencies are installed and any required environment variables are set.")
        exit()

    # Define a sample investor and their initial transactions
    investor_id = "investor_001"
    initial_transactions = {
        "AAPL": 10,  # Buy 10 shares of Apple
        "MSFT": 5,   # Buy 5 shares of Microsoft
        "GOOGL": 3,  # Buy 3 shares of Alphabet
        "AMZN": 2    # Buy 2 shares of Amazon
    }
    print(f"\nSimulating initial transactions for investor: {investor_id}")
    print(initial_transactions)

    # --- Test Portfolio Update ---
    try:
        tracker.update_portfolio(investor_id, initial_transactions)
        print(f"\nPortfolio updated for investor: {investor_id}")
        print("Current portfolio:", tracker.investor_db.get(investor_id))
    except Exception as e:
        print(f"ERROR: Failed to update portfolio: {e}")


    # --- Test Portfolio Optimization ---
    print(f"\nAttempting to optimize portfolio for investor: {investor_id}")
    try:
        # This will trigger data fetching, regime detection, and allocation strategy
        optimal_weights = tracker.optimize_portfolio(investor_id, method='cvar', alpha=0.95)
        print("\nPortfolio Optimization Results (Optimal Weights):")
        # Print weights formatted
        for asset, weight in optimal_weights.items():
            print(f"  {asset}: {weight:.4f}")

    except ValueError as ve:
        print(f"ERROR: Could not optimize portfolio - {ve}")
        print("This likely means no portfolio or asset data was available.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during optimization: {e}")


    # --- Test VaR Calculation ---
    print(f"\nCalculating Portfolio VaR for investor: {investor_id}")
    try:
        var_99 = tracker.calculate_portfolio_var(investor_id, confidence_level=0.99)
        print(f"Calculated 99% VaR: {var_99:.4f}")
    except ValueError as ve:
         print(f"ERROR: Could not calculate VaR - {ve}")
         print("This likely means no portfolio or asset data was available.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during VaR calculation: {e}")


    print("\n--- InvestorTracker Example Usage Finished ---")