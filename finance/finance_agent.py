from __future__ import annotations

import calendar
import os
import threading
import time
import numpy as np
import pandas as pd
import pytz
import torch
import torch.nn as nn

from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from typing import Any, Dict, List, Mapping, Optional

from finance.core.adaptive_learning import AdaptiveLearningSystem
from finance.core.backtester import Backtester
from finance.core.batch_manager import BatchManager
from finance.core.cultural_trend_analyzer import CulturalTrendAnalyzer
from finance.core.finance_memory import FinanceMemory
from finance.core.investor_tracker import InvestorTracker
from finance.core.market_data_handler import MarketDataHandler
from finance.core.stock_trading_env import StockTradingEnv
from finance.core.utils.config_loader import get_config_section, load_global_config
from finance.core.utils.data_quality_monitor import DataQualityMonitor
from src.agents.agent_factory import AgentFactory
from src.agents.collaborative.shared_memory import SharedMemory
from src.agents.knowledge.knowledge_cache import KnowledgeCache
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Finance Agent")
printer = PrettyPrinter


@dataclass
class CircuitBreaker:
    threshold: int = 3
    reset_timeout_seconds: int = 300
    failures: int = 0
    last_failure_ts: float = 0.0

    def trip(self) -> None:
        self.failures += 1
        self.last_failure_ts = time.time()

    def reset(self) -> None:
        self.failures = 0
        self.last_failure_ts = 0.0

    def is_open(self) -> bool:
        if self.failures < self.threshold:
            return False
        if (time.time() - self.last_failure_ts) > self.reset_timeout_seconds:
            self.reset()
            return False
        return True


class FinanceAgent(nn.Module):
    """
    Financial orchestration agent for execution + learning + coordination.

    SharedMemory
        - Inter-agent coordination & real-time state sharing
        - Portfolio state, trade signals, market regime
        - High-frequency, low-latency
    FinanceMemory
        - Agent-specific financial data storage
        - Batch data, backtest results, model states
        - Structured query access
    """

    def __init__(self, symbols: Optional[List[Mapping[str, str]]] = None) -> None:
        super().__init__()
        self.config = load_global_config() or {}
        self.agent_config = get_config_section("finance_agent") or {}
        self.bt_config = get_config_section("backtester") or {}
        cb_cfg = get_config_section("circuit_breaker") or {}

        self.position_size = float(self.config.get("position_size", 0.1))
        self.stop_loss = float(self.config.get("stop_loss", 0.1))
        self.take_profit = float(self.config.get("take_profit", 0.15))
        self.learning_cfg = get_config_section("learning_agent") or {}

        self.circuit_breaker = CircuitBreaker(
            threshold=int(cb_cfg.get("threshold", 3)),
            reset_timeout_seconds=int(cb_cfg.get("reset_timeout", 300)),
        )

        # Core subsystem modules (finance/core)
        self.finance_memory = FinanceMemory()
        self.shared_memory = SharedMemory()
        self.batch_manager = BatchManager()
        self.tracker = InvestorTracker()
        self.backtester = Backtester(tracker=self.tracker)
        self.backtester.finance_memory = self.finance_memory

        self.market_handler = MarketDataHandler(
            {
                "alpha_vantage": os.getenv("ALPHA_VANTAGE_KEY"),
                "finnhub": os.getenv("FINNHUB_KEY"),
                "polygon": os.getenv("POLYGON_API_KEY"),
                "enable_yahoo": True,
            }
        )
        self.adaptive_learner = AdaptiveLearningSystem()
        self.trend_analyzer = CulturalTrendAnalyzer(finance_memory=self.finance_memory)
        self.data_quality = DataQualityMonitor()

        self.knowledge_cache = KnowledgeCache()
        self.agent_factory = AgentFactory(config={"knowledge_agent": {"cache": self.knowledge_cache}})

        # Integrates: evaluation, planning, execution, knowledge, learning, reasoning, handler
        self.evaluation_agent = self.agent_factory.create("evaluation", self.shared_memory)
        self.planning_agent = self.agent_factory.create("planning", self.shared_memory)
        self.execution_agent = self.agent_factory.create("execution", self.shared_memory)
        self.knowledge_agent = self.agent_factory.create("knowledge", self.shared_memory)
        self.reasoning_agent = self.agent_factory.create("reasoning", self.shared_memory)
        self.handler_agent = self.agent_factory.create("handler", self.shared_memory)

        self.default_symbol = self.agent_config.get("default_symbol", "AAPL")
        self.symbols = symbols or [{"symbol": self.default_symbol, "name": "Primary"}]
        self.symbol = self.symbols[0]["symbol"].upper()

        self.portfolio_value = float(self.config.get("initial_capital", 10000.0))
        self.cash = self.portfolio_value
        self.open_positions: Dict[str, Dict[str, float]] = {}
        self.monthly_pnl = 0.0
        self.latest_insights: Dict[str, Any] = {}
        self.market_regime = "neutral"

        self.current_month = datetime.now().month
        self.monthly_peak_portfolio_value = self.portfolio_value
        self.closed_trades_this_month: List[Dict[str, Any]] = []

        # --- Financial Goals & Tracking ---
        self.financial_goals = {
            "monthly_profit": {
                "target": 750.0,
                "status": "in_progress",
            },
            "max_drawdown": {
                "limit": 0.15,
                "status": "ok",
            },
            "sentiment_risk": {
                "threshold": 0.5,
                "risk_factor": 1.5,
            },
        }

        # Environment for LearningAgent: sets expected state/action space.
        self.slai_trading_env = StockTradingEnv(
            symbol=self.symbol,
            initial_balance=self.portfolio_value,
            finance_agent=self,
        )
        self.learning_agent = self.agent_factory.create(
            "learning",
            self.shared_memory,
            env=self.slai_trading_env,
        )
        self.learning_agent.training_mode = False
        self._feature_history: deque[torch.Tensor] = deque(maxlen=5)
        self._last_learning_state: Optional[torch.Tensor] = None

        self.knowledge_agent.cache = self.knowledge_cache

        self._knowledge_thread = threading.Thread(target=self._update_knowledge_periodically, daemon=True)
        self._knowledge_thread.start()

        self._run_initial_backtest()
        self._sync_portfolio_state()

    # -------------------------
    # Cross-agent integration
    # -------------------------
    def _build_cycle_context(self, symbol: str) -> Dict[str, Any]:
        """Builds a normalized context payload shared with planning/reasoning/handler."""
        context = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "market_regime": self.market_regime,
            "goals": self.financial_goals,
            "portfolio": {
                "cash": self.cash,
                "positions": self.open_positions,
                "portfolio_value": self._get_total_portfolio_value(),
                "monthly_pnl": self.monthly_pnl,
            },
            "latest_insights": self.latest_insights,
            "risk_factor": self._determine_risk_factor(),
        }
        self.shared_memory.put("finance_context", context, ttl=180)
        return context

    def _knowledge_enrich_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enriches context with grounded knowledge snippets and cached references."""
        symbol = context["symbol"]
        concepts = [
            "market_regime",
            "position_sizing_rules",
            "compliance_guidelines",
            f"{symbol}_risk_factors",
        ]
        knowledge_pack: Dict[str, Any] = {}
        for concept in concepts:
            try:
                results = self.knowledge_agent.contextual_search(concept)
                knowledge_pack[concept] = [doc.get("text", "") for _, doc in results[:5]]
            except Exception as exc:
                logger.debug("Knowledge fetch failed for %s: %s", concept, exc)
                knowledge_pack[concept] = []

        context["knowledge"] = knowledge_pack
        self.shared_memory.put("knowledge_context", {"symbol": symbol, "knowledge": knowledge_pack}, ttl=300)
        self.finance_memory.add_financial_data(
            data={"symbol": symbol, "knowledge": knowledge_pack, "ts": context["timestamp"]},
            data_type="knowledge_snapshot",
            tags=["knowledge", f"symbol_{symbol}"],
            priority="medium",
        )
        return context

    def _reason_market_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Uses reasoning agent to produce a market thesis and confidence score."""
        reasoning_output: Dict[str, Any] = {
            "thesis": "neutral stance",
            "confidence": 0.5,
            "constraints": [],
        }
        try:
            if hasattr(self.reasoning_agent, "perform_task"):
                result = self.reasoning_agent.perform_task({"type": "market_reasoning", "context": context})
            elif hasattr(self.reasoning_agent, "execute"):
                result = self.reasoning_agent.execute({"type": "market_reasoning", "context": context})
            else:
                result = {}
            if isinstance(result, dict):
                reasoning_output.update(
                    {
                        "thesis": result.get("thesis", reasoning_output["thesis"]),
                        "confidence": float(result.get("confidence", reasoning_output["confidence"])),
                        "constraints": result.get("constraints", reasoning_output["constraints"]),
                    }
                )
        except Exception as exc:
            logger.warning("Reasoning agent failed; using fallback thesis: %s", exc)

        context["reasoning"] = reasoning_output
        self.shared_memory.put("reasoning_output", {"symbol": context["symbol"], **reasoning_output}, ttl=180)
        return context

    def _build_trade_plan(self, context: Dict[str, Any], signal: str) -> Dict[str, Any]:
        """Build and validate a trade plan via planning + handler agents."""
        symbol = context["symbol"]
        plan_request = {
            "type": "trade_plan",
            "symbol": symbol,
            "signal": signal,
            "risk_factor": context.get("risk_factor", 1.0),
            "portfolio": context.get("portfolio", {}),
            "reasoning": context.get("reasoning", {}),
            "knowledge": context.get("knowledge", {}),
        }
        generated_plan: Dict[str, Any] = {"steps": [], "approved": signal != "hold", "signal": signal}
        try:
            if hasattr(self.planning_agent, "perform_task"):
                plan_result = self.planning_agent.perform_task(plan_request)
            elif hasattr(self.planning_agent, "generate_plan"):
                plan_result = self.planning_agent.generate_plan(plan_request)
            else:
                plan_result = {}

            if isinstance(plan_result, dict):
                generated_plan["steps"] = plan_result.get("steps", plan_result.get("plan", []))
                generated_plan["approved"] = bool(plan_result.get("approved", generated_plan["approved"]))
        except Exception as exc:
            logger.warning("Planning agent failed; fallback plan in use: %s", exc)

        try:
            handler_payload = {"type": "trade_guardrail_check", "plan": generated_plan, "context": context}
            if hasattr(self.handler_agent, "perform_task"):
                handler_result = self.handler_agent.perform_task(handler_payload)
            elif hasattr(self.handler_agent, "execute"):
                handler_result = self.handler_agent.execute(handler_payload)
            else:
                handler_result = {}
            if isinstance(handler_result, dict):
                if handler_result.get("blocked") is True:
                    generated_plan["approved"] = False
                    generated_plan["block_reason"] = handler_result.get("reason", "handler_block")
        except Exception as exc:
            logger.warning("Handler guardrail check failed: %s", exc)

        self.shared_memory.put("latest_trade_plan", {"symbol": symbol, "plan": generated_plan}, ttl=120)
        self.finance_memory.add_financial_data(
            data={"symbol": symbol, "plan": generated_plan, "ts": context["timestamp"]},
            data_type="trade_plan_snapshot",
            tags=["planning", f"symbol_{symbol}"],
            priority="medium",
        )
        return generated_plan

    def _execute_plan(self, context: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches plan to execution agent and records execution traces."""
        execution_result: Dict[str, Any] = {"status": "noop", "steps_executed": 0}
        try:
            payload = {"type": "execute_trade_plan", "plan": plan, "context": context}
            if hasattr(self.execution_agent, "perform_task"):
                result = self.execution_agent.perform_task(payload)
            elif hasattr(self.execution_agent, "execute"):
                result = self.execution_agent.execute(payload)
            else:
                result = {}
            if isinstance(result, dict):
                execution_result.update(result)
                execution_result.setdefault("status", "success")
        except Exception as exc:
            execution_result = {"status": "failed", "reason": str(exc)}
            logger.warning("Execution agent failed for plan dispatch: %s", exc)

        self.shared_memory.put("execution_result", {"symbol": context["symbol"], **execution_result}, ttl=120)
        return execution_result

    # -------------------------
    # Market calendar utilities
    # -------------------------
    def _is_weekday(self, dt_obj: datetime) -> bool:
        return dt_obj.weekday() < 5

    def _observed_fixed_holiday(self, year: int, month: int, day: int) -> date:
        d = date(year, month, day)
        if d.weekday() == 5:
            return d - timedelta(days=1)
        if d.weekday() == 6:
            return d + timedelta(days=1)
        return d

    def _nth_weekday(self, year: int, month: int, weekday: int, n: int) -> date:
        d = date(year, month, 1)
        while d.weekday() != weekday:
            d += timedelta(days=1)
        return d + timedelta(days=7 * (n - 1))

    def _last_weekday(self, year: int, month: int, weekday: int) -> date:
        last_day = calendar.monthrange(year, month)[1]
        d = date(year, month, last_day)
        while d.weekday() != weekday:
            d -= timedelta(days=1)
        return d

    def _easter_sunday(self, year: int) -> date:
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return date(year, month, day)

    def _us_market_holidays(self, year: int) -> set[date]:
        new_years = self._observed_fixed_holiday(year, 1, 1)
        mlk = self._nth_weekday(year, 1, calendar.MONDAY, 3)
        presidents = self._nth_weekday(year, 2, calendar.MONDAY, 3)
        good_friday = self._easter_sunday(year) - timedelta(days=2)
        memorial = self._last_weekday(year, 5, calendar.MONDAY)
        juneteenth = self._observed_fixed_holiday(year, 6, 19)
        independence = self._observed_fixed_holiday(year, 7, 4)
        labor = self._nth_weekday(year, 9, calendar.MONDAY, 1)
        thanksgiving = self._nth_weekday(year, 11, calendar.THURSDAY, 4)
        christmas = self._observed_fixed_holiday(year, 12, 25)
        return {
            new_years,
            mlk,
            presidents,
            good_friday,
            memorial,
            juneteenth,
            independence,
            labor,
            thanksgiving,
            christmas,
        }

    def _is_trading_holiday(self, dt_obj: datetime) -> bool:
        ny_tz = pytz.timezone("America/New_York")
        ny_date = dt_obj.astimezone(ny_tz).date()
        return ny_date in self._us_market_holidays(ny_date.year)

    def _is_within_trading_hours(self, dt_obj: datetime) -> bool:
        ny_tz = pytz.timezone("America/New_York")
        now_ny = dt_obj.astimezone(ny_tz).time()
        return dt_time(9, 30) <= now_ny <= dt_time(16, 0)

    def _is_market_open(self) -> bool:
        now_utc = datetime.now(pytz.utc)
        return self._is_weekday(now_utc) and not self._is_trading_holiday(now_utc) and self._is_within_trading_hours(now_utc)

    # -------------------------
    # Startup and periodic jobs
    # -------------------------
    def _run_initial_backtest(self) -> None:
        """Runs an initial backtest on startup to ensure metrics are available."""
        try:
            historical_data = self.backtester.load_batch_data(self.batch_manager)
            historical_data = self._prepare_backtest_dataset(historical_data)
            if historical_data.empty:
                days = 365
                idx = pd.date_range(end=pd.Timestamp.now(), periods=days, freq="B")
                prices = np.cumprod(1 + np.random.normal(0.0004, 0.015, days)) * 100
                historical_data = pd.DataFrame(
                    {
                        "datetime": idx,
                        "symbol": self.symbol,
                        "open": prices,
                        "high": prices * 1.01,
                        "low": prices * 0.99,
                        "close": prices,
                        "volume": np.random.randint(150_000, 2_000_000, size=days),
                    }
                )
            self.run_backtest(historical_data)
        except Exception as exc:
            logger.error("Initial backtest failed: %s", exc, exc_info=True)
            self.circuit_breaker.trip()

    def _prepare_backtest_dataset(self, data: Optional[pd.DataFrame]) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame()
        prepared = data.copy()
        if "symbol" not in prepared.columns:
            prepared["symbol"] = self.symbol
        prepared["symbol"] = prepared["symbol"].astype(str).str.upper()
        prepared = prepared[prepared["symbol"] == self.symbol].copy()
        if prepared.empty:
            prepared = data.copy()
            prepared["symbol"] = self.symbol
        if "datetime" in prepared.columns:
            prepared["datetime"] = pd.to_datetime(prepared["datetime"], errors="coerce")
        elif "date" in prepared.columns:
            prepared["datetime"] = pd.to_datetime(prepared["date"], errors="coerce")
        elif "timestamp" in prepared.columns:
            prepared["datetime"] = pd.to_datetime(prepared["timestamp"], unit="s", errors="coerce")
        prepared = prepared.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        if prepared.empty:
            return prepared

        required_rows = max(20, int(getattr(self.backtester, "train_window", 180) + getattr(self.backtester, "test_window", 30) + 5))
        if len(prepared) >= required_rows:
            return prepared

        rows_to_add = required_rows - len(prepared)
        last_close = float(pd.to_numeric(prepared.get("close"), errors="coerce").dropna().iloc[-1]) if "close" in prepared.columns and not prepared["close"].isna().all() else 100.0
        new_dates = pd.bdate_range(start=prepared["datetime"].iloc[-1] + pd.Timedelta(days=1), periods=rows_to_add)
        drift = np.random.normal(0.0003, 0.012, rows_to_add)
        synthetic_close = last_close * np.cumprod(1.0 + drift)
        synthetic = pd.DataFrame(
            {
                "datetime": new_dates,
                "symbol": self.symbol,
                "close": synthetic_close,
                "open": synthetic_close * np.random.uniform(0.995, 1.005, rows_to_add),
                "high": synthetic_close * np.random.uniform(1.0, 1.01, rows_to_add),
                "low": synthetic_close * np.random.uniform(0.99, 1.0, rows_to_add),
                "volume": np.random.randint(100_000, 2_000_000, size=rows_to_add).astype(float),
            }
        )
        logger.info(
            "Backtest dataset too short for configured windows; appended %s synthetic rows (symbol=%s, original_rows=%s, required_rows=%s).",
            rows_to_add,
            self.symbol,
            len(prepared),
            required_rows,
        )
        return pd.concat([prepared, synthetic], ignore_index=True).sort_values("datetime").reset_index(drop=True)

    def _update_knowledge_periodically(self) -> None:
        while True:
            try:
                data = self.backtester.load_batch_data(self.batch_manager)
                if not data.empty:
                    self.optimize_trading_parameters(data)
                self._run_learning_maintenance(data)
                self.latest_insights = self._retrieve_financial_insights()
                self.shared_memory.put("financial_insights", self.latest_insights, ttl=300)
            except Exception as exc:
                logger.warning("Periodic knowledge update failed: %s", exc)
            time.sleep(3600)

    def _run_learning_maintenance(self, historical_data: pd.DataFrame) -> None:
        """Periodic synchronization hook between LearningAgent and AdaptiveLearningSystem."""
        if historical_data is None or historical_data.empty:
            return
        try:
            if hasattr(self.learning_agent, "train_from_embeddings"):
                self.learning_agent.train_from_embeddings()
        except Exception as exc:
            logger.debug("LearningAgent maintenance training skipped: %s", exc)
        try:
            if hasattr(self.adaptive_learner, "get_diagnostics"):
                self.shared_memory.put("learning_maintenance", self.adaptive_learner.get_diagnostics(), ttl=600)
        except Exception:
            pass

    def _retrieve_financial_insights(self) -> Dict[str, Any]:
        concepts = ["market_regime", f"{self.symbol}_risk_factors", "position_sizing_rules", "compliance_guidelines"]
        insights: Dict[str, Any] = {}
        for concept in concepts:
            try:
                results = self.knowledge_agent.contextual_search(concept)
                insights[concept] = [doc.get("text", "") for _, doc in results[:3]]
            except Exception:
                insights[concept] = []
        return insights

    # -------------------------
    # Trading logic
    # -------------------------
    def _build_learning_features(self, symbol: str, lookback: int = 40) -> Optional[torch.Tensor]:
        """Build shared market state features for both adaptive and RL learners."""
        market_data = self.market_handler.fetch_data(symbol, lookback=lookback)
        if not market_data:
            return None
        df = pd.DataFrame(market_data)
        close = pd.to_numeric(df["close"], errors="coerce").ffill()
        if close.empty:
            return None
        current = float(close.iloc[-1])
        momentum_5 = float(close.pct_change(5).iloc[-1]) if len(close) > 6 else 0.0
        momentum_20 = float(close.pct_change(20).iloc[-1]) if len(close) > 21 else momentum_5
        volatility_10 = float(close.pct_change().rolling(10).std().iloc[-1]) if len(close) > 11 else 0.0
        sentiment = float(self.trend_analyzer.get_latest_sentiment(symbol))
        features = torch.tensor(
            [[current, momentum_5, momentum_20, volatility_10, sentiment, self.monthly_pnl]],
            dtype=torch.float32,
        )
        target_dim = int(getattr(self.adaptive_learner, "market_features", features.shape[1]))
        if features.shape[1] < target_dim:
            features = torch.cat([features, torch.zeros((1, target_dim - features.shape[1]), dtype=torch.float32)], dim=1)
        elif features.shape[1] > target_dim:
            features = features[:, :target_dim]
        self._feature_history.append(features)
        return features

    def _rl_action_to_signal(self, action: Any) -> str:
        try:
            idx = int(action)
        except Exception:
            return "hold"
        if idx in (1, 2):
            return "buy"
        if idx in (3, 4):
            return "sell"
        return "hold"

    def _learning_agent_signal(self, features: torch.Tensor) -> str:
        """Derive directional signal from LearningAgent policy output."""
        try:
            obs = features.detach().cpu().numpy().reshape(-1)
            if hasattr(self.learning_agent, "select_agent_strategy"):
                strategy = self.learning_agent.select_agent_strategy(obs)
                strategy_agent = getattr(self.learning_agent, "agents", {}).get(strategy)
                if strategy_agent is not None and hasattr(strategy_agent, "get_action"):
                    return self._rl_action_to_signal(strategy_agent.get_action(obs))
            if hasattr(self.learning_agent, "act"):
                return self._rl_action_to_signal(self.learning_agent.act(obs))
        except Exception as exc:
            logger.debug("Learning agent inference fallback to HOLD: %s", exc)
        return "hold"

    def _generate_hybrid_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Combines AdaptiveLearningSystem price forecast with LearningAgent policy action.
        Consensus boosts confidence; disagreements fall back to adaptive prediction.
        """
        features = self._build_learning_features(symbol, lookback=60)
        if features is None:
            return {"signal": "hold", "source": "no_data", "confidence": 0.0}

        latest = self.market_handler.fetch_data(symbol, lookback=1)
        if not latest:
            return {"signal": "hold", "source": "no_price", "confidence": 0.0}
        current_price = float(latest[0]["close"])

        als_prediction = self.adaptive_learner.robust_predict(features)
        predicted_price = float(als_prediction.get("prediction", current_price))
        threshold = float(self.agent_config.get("signal_threshold", 0.01))
        als_signal = "hold"
        if predicted_price > current_price * (1 + threshold):
            als_signal = "buy"
        elif predicted_price < current_price * (1 - threshold):
            als_signal = "sell"
        rl_signal = self._learning_agent_signal(features)

        confidence = abs(predicted_price - current_price) / max(current_price, 1e-12)
        if als_signal == rl_signal and als_signal != "hold":
            final_signal, source, confidence = als_signal, "als+rl_consensus", min(1.0, 0.5 + confidence)
        elif als_signal != "hold":
            final_signal, source, confidence = als_signal, "als_primary", min(1.0, 0.35 + confidence)
        else:
            final_signal = rl_signal
            source = "rl_primary" if rl_signal != "hold" else "hold"
            confidence = min(1.0, 0.25 + confidence)

        return {
            "signal": final_signal,
            "source": source,
            "confidence": float(confidence),
            "als_signal": als_signal,
            "rl_signal": rl_signal,
            "predicted_price": predicted_price,
            "current_price": current_price,
            "features": features,
        }

    def _generate_signal(self, symbol: str) -> str:
        """Generates a trading signal for a symbol using the ALS."""
        features = torch.randn(1, self.adaptive_learner.market_features)
        prediction = self.adaptive_learner.robust_predict(features)
        current_price = self.market_handler.fetch_data(symbol, lookback=1)[0]['close']
        predicted_price = float(prediction.get('prediction', current_price))

        # Incorporate financial knowledge
        market_insights = self.latest_insights.get("market_regime", [])
        if any("bearish" in insight.lower() for insight in market_insights):
            logger.info("Knowledge: Bearish market regime detected")
            predicted_price *= 0.95  # Adjust prediction downward

        if predicted_price > current_price * (1 + self.agent_config.get('signal_threshold')):
            return 'buy'
        elif predicted_price < current_price * (1 - self.agent_config.get('signal_threshold')):
            return 'sell'
        return 'hold'

    def _calculate_buying_power(self) -> float:
        return max(0.0, float(self.cash))

    def _calculate_pnl(self, position: Dict[str, float], current_price: float) -> float:
        if position.get("direction") != "long":
            return 0.0
        return (current_price - float(position["entry_price"])) * float(position["shares"])

    def _close_position(self, symbol: str, current_price: float) -> None:
        if symbol not in self.open_positions:
            return
        pos = self.open_positions.pop(symbol)
        proceeds = float(pos["shares"]) * current_price * (1 - self.backtester.transaction_cost)
        pnl = self._calculate_pnl(pos, current_price)
        self.cash += proceeds
        self.monthly_pnl += pnl
        self.closed_trades_this_month.append({"symbol": symbol, "pnl": pnl, "timestamp": datetime.utcnow().isoformat()})

    def _update_adaptive_model(self, symbol: str) -> None:
        data = self.market_handler.fetch_data(symbol, lookback=2)
        if len(data) < 2:
            return
        prev_close = float(data[-2]["close"])
        cur_close = float(data[-1]["close"])
        ret = (cur_close / max(prev_close, 1e-12)) - 1.0
        features = self._build_learning_features(symbol, lookback=30)
        if features is None:
            features = torch.tensor([[cur_close, ret, self.monthly_pnl]], dtype=torch.float32)
            if features.shape[1] < getattr(self.adaptive_learner, "market_features", 3):
                pad = getattr(self.adaptive_learner, "market_features", 3) - features.shape[1]
                features = torch.cat([features, torch.zeros((1, pad), dtype=torch.float32)], dim=1)
        self.adaptive_learner.update(features, cur_close)
        self._last_learning_state = features

    def _update_learning_agent(self, reward: float, next_state: Optional[torch.Tensor], action_signal: str) -> None:
        """Feed transition reward to LearningAgent after adaptive update."""
        if self._last_learning_state is None or next_state is None:
            return
        action_id = {"hold": 0, "buy": 1, "sell": 3}.get(action_signal, 0)
        try:
            if hasattr(self.learning_agent, "observe"):
                self.learning_agent.observe(
                    self._last_learning_state.detach().cpu().numpy().reshape(-1),
                    action_id,
                    float(reward),
                    next_state.detach().cpu().numpy().reshape(-1),
                )
            if hasattr(self.learning_agent, "train_from_embeddings"):
                self.learning_agent.train_from_embeddings()
            self.shared_memory.put(
                "learning_feedback",
                {"reward": float(reward), "action": action_signal, "ts": datetime.utcnow().isoformat()},
                ttl=180,
            )
        except Exception as exc:
            logger.debug("Learning agent update skipped: %s", exc)

    def _determine_risk_factor(self) -> float:
        risk_factor = 1.0
        if self.financial_goals["max_drawdown"]["status"] == "breached":
            risk_factor *= 0.5

        target = float(self.financial_goals["monthly_profit"]["target"])
        progress = self.monthly_pnl / target if target > 0 else 1.0
        if progress < 0.25:
            risk_factor *= 1.2
        elif progress > 0.75:
            risk_factor *= 0.8

        sentiment = float(self.trend_analyzer.get_latest_sentiment(self.symbol))
        if sentiment > float(self.financial_goals["sentiment_risk"]["threshold"]):
            risk_factor *= float(self.financial_goals["sentiment_risk"]["risk_factor"])

        return max(0.1, min(2.0, risk_factor))

    def _execute_trading_logic(self, symbol: str, signal_override: Optional[str] = None):
        """
        Core logic for making a trading decision in a single cycle.
        """
        # 1. Check Circuit Breaker
        if self.circuit_breaker.is_open():
            logger.warning(f"Circuit breaker is OPEN for {symbol}. Skipping trade.")
            return

        # 2. Generate Trading Signal using Adaptive Learning System
        hybrid = self._generate_hybrid_signal(symbol)
        signal = signal_override or hybrid.get("signal", "hold")

        # 3. Fetch current market price
        try:
            market_data = self.market_handler.fetch_data(symbol, lookback=1)
            if not market_data:
                logger.warning(f"Could not fetch current price for {symbol}. Skipping trading logic.")
                return
            current_price = market_data[0]["close"]
        except Exception as e:
            logger.error(f"Failed to fetch current price for {symbol}: {e}")
            return

        # 4. Get Risk Factor based on financial goals
        risk_factor = self._determine_risk_factor()

        # 5. Execute Trade based on signal
        if signal == "buy" and self._calculate_buying_power() > 0:
            position_value = self.portfolio_value * self.position_size * risk_factor
            quantity_to_buy = int(position_value / current_price)

            if quantity_to_buy > 0:
                self.open_positions[symbol] = {
                    "direction": "long",
                    "shares": quantity_to_buy,
                    "entry_price": current_price,
                }
                self.cash -= quantity_to_buy * current_price * (1 + self.backtester.transaction_cost)
                logger.info(f"Executed BUY for {quantity_to_buy} shares of {symbol} at ${current_price:.2f}")

        elif signal == "sell":
            if symbol in self.open_positions and self.open_positions[symbol]["direction"] == "long":
                self._close_position(symbol, current_price)
                logger.info(f"Executed SELL to close position for {symbol} at ${current_price:.2f}")

        # 6. Update the adaptive learning model with new data
        pre_value = self._get_total_portfolio_value()
        self._update_adaptive_model(symbol)
        next_state = self._build_learning_features(symbol, lookback=30)
        reward = self._get_total_portfolio_value() - pre_value
        self._update_learning_agent(reward=reward, next_state=next_state, action_signal=signal)

        self.shared_memory.put(
            f"trade_signal:{symbol}",
            {
                "signal": signal,
                "price": current_price,
                "source": hybrid.get("source", "unknown"),
                "confidence": hybrid.get("confidence", 0.0),
                "als_signal": hybrid.get("als_signal", "hold"),
                "rl_signal": hybrid.get("rl_signal", "hold"),
            },
            ttl=120,
        )

    # -------------------------
    # Agent orchestration cycle
    # -------------------------
    def collect_and_process_data(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        try:
            data = self.market_handler.fetch_data(symbol, lookback=30)
            if not data:
                return None
            latest = data[-1]
            self.batch_manager.add_data_point(symbol, latest["close"], latest.get("volume", 0.0))
            quality_ok = self.data_quality.validate_market_data(data) if hasattr(self.data_quality, "validate_market_data") else True
            self.finance_memory.add_financial_data(
                data={"symbol": symbol, "quality_ok": bool(quality_ok), "rows": len(data), "latest": latest},
                data_type="market_cycle_data",
                tags=[f"symbol_{symbol}", "ingestion"],
                priority="medium",
            )
            return data
        except Exception as exc:
            logger.error("Data collection failed for %s: %s", symbol, exc, exc_info=True)
            self.circuit_breaker.trip()
            return None

    def _check_and_update_financial_goals(self) -> None:
        if datetime.now().month != self.current_month:
            self.current_month = datetime.now().month
            self.monthly_pnl = 0.0
            self.monthly_peak_portfolio_value = self._get_total_portfolio_value()
            self.closed_trades_this_month.clear()

        if self.monthly_pnl >= self.financial_goals["monthly_profit"]["target"]:
            self.financial_goals["monthly_profit"]["status"] = "met"

        current_value = self._get_total_portfolio_value()
        peak = max(self.monthly_peak_portfolio_value, 1e-12)
        drawdown = (peak - current_value) / peak
        if drawdown > self.financial_goals["max_drawdown"]["limit"]:
            self.financial_goals["max_drawdown"]["status"] = "breached"
        else:
            self.financial_goals["max_drawdown"]["status"] = "ok"
            self.monthly_peak_portfolio_value = max(self.monthly_peak_portfolio_value, current_value)

    def _detect_market_regime(self) -> str:
        sentiment = float(self.trend_analyzer.get_latest_sentiment(self.symbol))
        if sentiment > 0.3:
            return "bullish"
        if sentiment < -0.3:
            return "bearish"
        return "neutral"

    def _sync_portfolio_state(self) -> None:
        payload = {
            "portfolio_value": self._get_total_portfolio_value(),
            "cash": self.cash,
            "positions": self.open_positions,
            "monthly_pnl": self.monthly_pnl,
            "market_regime": self.market_regime,
        }
        self.shared_memory.put("portfolio_state", payload, ttl=60)
        self.finance_memory.add_financial_data(
            data=payload,
            data_type="portfolio_snapshot",
            tags=["portfolio", f"symbol_{self.symbol}"],
            priority="high",
        )

    def _get_total_portfolio_value(self) -> float:
        holdings_value = 0.0
        for sym, pos in self.open_positions.items():
            try:
                px = float(self.market_handler.fetch_data(sym, lookback=1)[0]["close"])
            except Exception:
                px = float(pos["entry_price"])
            holdings_value += px * float(pos["shares"])
        return float(self.cash + holdings_value)

    def run_backtest(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        self.backtester.store_last_metrics({})
        metrics = self.backtester.walk_forward_test(historical_data)
        self.backtester.store_last_metrics(metrics)
        self.finance_memory.add_financial_data(
            data=metrics,
            data_type="backtest_metrics",
            tags=["backtesting", f"symbol_{self.symbol}"],
            priority="high",
        )
        return metrics

    def optimize_trading_parameters(self, data: pd.DataFrame) -> None:
        param_grid = {
            "position_size": [0.05, 0.1, 0.15],
            "stop_loss": [0.05, 0.1, 0.15],
            "take_profit": [0.1, 0.15, 0.2],
        }
        best = self.backtester.optimize_parameters(data, param_grid)
        self.position_size = float(best.get("position_size", self.position_size))
        self.stop_loss = float(best.get("stop_loss", self.stop_loss))
        self.take_profit = float(best.get("take_profit", self.take_profit))

    def run_cycle(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        symbol = (symbol or self.symbol).upper()
        cycle_ts = datetime.utcnow().isoformat()

        if not self._is_market_open() and not bool(self.agent_config.get("allow_after_hours", False)):
            result = {"status": "skipped", "reason": "market_closed", "symbol": symbol, "time": cycle_ts}
            self.shared_memory.put("last_cycle", result, ttl=120)
            return result

        self.collect_and_process_data(symbol)

        try:
            snippets = self.trend_analyzer.sentiment_scraper.get_sentiment_snippets(symbol)
            text = self.trend_analyzer._aggregate_text(snippets)
            if text and text != "No content found":
                self.trend_analyzer.update_trends(text, symbol)
        except Exception as exc:
            logger.debug("Trend update skipped: %s", exc)

        self.market_regime = self._detect_market_regime()
        self.shared_memory.put("market_regime", {"symbol": symbol, "regime": self.market_regime}, ttl=300)

        context = self._build_cycle_context(symbol)
        context = self._knowledge_enrich_context(context)
        context = self._reason_market_context(context)

        hybrid_signal = self._generate_hybrid_signal(symbol)
        signal = hybrid_signal.get("signal", "hold")
        context["learning"] = {
            "hybrid_signal": signal,
            "signal_source": hybrid_signal.get("source", "unknown"),
            "confidence": hybrid_signal.get("confidence", 0.0),
            "als_signal": hybrid_signal.get("als_signal", "hold"),
            "rl_signal": hybrid_signal.get("rl_signal", "hold"),
        }
        self.shared_memory.put("learning_signal_context", {"symbol": symbol, **context["learning"]}, ttl=180)
        plan = self._build_trade_plan(context, signal)
        execution_result = self._execute_plan(context, plan)
        self.shared_memory.put(
            "agent_status:planning_pipeline",
            {
                "ok": bool(plan.get("approved", False)),
                "signal": signal,
                "execution_status": execution_result.get("status", "unknown"),
                "ts": cycle_ts,
            },
            ttl=120,
        )

        if plan.get("approved", False):
            self._execute_trading_logic(symbol, signal_override=signal)
        else:
            self.shared_memory.put(
                "trade_signal_blocked",
                {
                    "symbol": symbol,
                    "signal": signal,
                    "reason": plan.get("block_reason", "not_approved"),
                    "ts": cycle_ts,
                },
                ttl=180,
            )

        self._check_and_update_financial_goals()
        self._sync_portfolio_state()

        try:
            eval_payload = {
                "type": "cycle_evaluation",
                "symbol": symbol,
                "context": context,
                "plan": plan,
                "execution_result": execution_result,
                "portfolio_state": {
                    "value": self._get_total_portfolio_value(),
                    "cash": self.cash,
                    "monthly_pnl": self.monthly_pnl,
                },
            }
            if hasattr(self.evaluation_agent, "perform_task"):
                self.evaluation_agent.perform_task(eval_payload)
            elif hasattr(self.evaluation_agent, "execute_validation_cycle"):
                self.evaluation_agent.execute_validation_cycle(eval_payload)
            self.shared_memory.put("agent_status:evaluation", {"ok": True, "ts": cycle_ts}, ttl=120)
        except Exception as exc:
            self.shared_memory.put("agent_status:evaluation", {"ok": False, "err": str(exc), "ts": cycle_ts}, ttl=120)

        result = {
            "status": "ok",
            "symbol": symbol,
            "time": cycle_ts,
            "portfolio_value": self._get_total_portfolio_value(),
            "monthly_pnl": self.monthly_pnl,
            "market_regime": self.market_regime,
        }
        self.finance_memory.add_financial_data(
            data=result,
            data_type="cycle_result",
            tags=["cycle", f"symbol_{symbol}"],
            priority="medium",
        )
        self.shared_memory.put("last_cycle", result, ttl=300)
        return result

    def run_execution_loop(self) -> None:
        interval = int(self.agent_config.get("execution_interval_seconds", 600))
        while True:
            try:
                self.run_cycle(self.symbol)
            except Exception as exc:
                logger.error("Execution loop error: %s", exc, exc_info=True)
                self.circuit_breaker.trip()
            time.sleep(interval)


if __name__ == "__main__":
    agent = FinanceAgent()
    logger.info("Finance Agent started for %s", agent.symbol)
    agent.run_execution_loop()
