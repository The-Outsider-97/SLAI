from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from flask import Flask, jsonify, request, send_from_directory

from .finance_agent import FinanceAgent
from logs.logger import get_logger

logger = get_logger("Finance WebApp")

ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"
DATA_DIR = ROOT_DIR / "data"


@dataclass
class AgentRuntimeState:
    initialized: bool = False
    last_error: Optional[str] = None
    last_cycle_at: Optional[str] = None
    total_cycles: int = 0


class FinanceAgentService:
    """Thread-safe runtime wrapper around an eagerly-initialized FinanceAgent."""

    def __init__(self, agent: Optional[FinanceAgent] = None, init_error: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self._agent = agent
        self.runtime = AgentRuntimeState(initialized=agent is not None, last_error=init_error)

    def get_agent(self) -> FinanceAgent:
        if self._agent is None:
            raise RuntimeError(self.runtime.last_error or "FinanceAgent is not initialized.")
        return self._agent

    def run_cycle(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            agent = self.get_agent()
            result = agent.run_cycle(symbol)
            self.runtime.last_cycle_at = datetime.now(timezone.utc).isoformat()
            self.runtime.total_cycles += 1
            return result


def _load_boot_symbols() -> List[Dict[str, str]]:
    stocks_file = DATA_DIR / "north_american_stocks.json"
    if stocks_file.exists():
        try:
            payload = json.loads(stocks_file.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                result = []
                for item in payload:
                    if isinstance(item, dict) and item.get("symbol"):
                        result.append(
                            {
                                "symbol": str(item.get("symbol", "")).upper(),
                                "name": str(item.get("name", "Unknown")),
                            }
                        )
                if result:
                    return result
        except Exception:
            logger.warning("Boot symbol load failed, using fallback list", exc_info=True)
    return [{"symbol": "AAPL", "name": "Apple Inc."}]


_BOOT_AGENT: Optional[FinanceAgent] = None
_BOOT_ERROR: Optional[str] = None
try:
    _BOOT_AGENT = FinanceAgent(symbols=_load_boot_symbols())
    logger.info("FinanceAgent eagerly initialized for Flask app")
except Exception as exc:
    _BOOT_ERROR = f"{type(exc).__name__}: {exc}"
    logger.error("FinanceAgent eager initialization failed: %s", _BOOT_ERROR, exc_info=True)

agent_service = FinanceAgentService(agent=_BOOT_AGENT, init_error=_BOOT_ERROR)


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")

    # -------------------------
    # Frontend routes
    # -------------------------
    @app.route("/")
    @app.route("/index")
    def index_page():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.route("/<path:page>")
    def serve_page(page: str):
        # Support direct html pages and static assets in /static/frontend.
        allowed_pages = {
            "predictions": "predictions.html",
            "signals": "signals.html",
            "indicators": "indicators.html",
            "diagnostics": "diagnostics.html",
            "learning": "learning.html",
            "portfolio": "portfolio.html",
            "trends": "trends.html",
            "dataQuality": "dataQuality.html",
        }
        if page in allowed_pages:
            return send_from_directory(STATIC_DIR, allowed_pages[page])
        target = STATIC_DIR / page
        if target.exists() and target.is_file():
            return send_from_directory(STATIC_DIR, page)
        if target.with_suffix(".html").exists():
            return send_from_directory(STATIC_DIR, f"{page}.html")
        return jsonify({"error": "not_found", "details": f"Unknown page: {page}"}), 404

    # -------------------------
    # Utilities
    # -------------------------
    def _safe_agent():
        try:
            return agent_service.get_agent(), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def _stocks_catalog() -> List[Dict[str, str]]:
        stocks_file = DATA_DIR / "north_american_stocks.json"
        if not stocks_file.exists():
            return [{"symbol": "AAPL", "name": "Apple Inc."}]
        try:
            payload = json.loads(stocks_file.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [
                    {"symbol": str(item.get("symbol", "")).upper(), "name": str(item.get("name", "Unknown"))}
                    for item in payload
                    if isinstance(item, dict) and item.get("symbol")
                ]
        except Exception:
            logger.warning("Failed to parse stock catalog", exc_info=True)
        return [{"symbol": "AAPL", "name": "Apple Inc."}]

    def _latest_price(agent, symbol: str) -> float:
        try:
            bars = agent.market_handler.fetch_data(symbol, lookback=1)
            if bars:
                return float(bars[0].get("close", 0.0))
        except Exception:
            pass
        return 0.0

    # -------------------------
    # API endpoints
    # -------------------------
    @app.route('/api/market/status', methods=['GET'])
    @app.route('/api/market_status', methods=['GET'])
    @app.route('/api/market/status/', methods=['GET'])
    @app.route('/api/market_status/', methods=['GET'])
    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": asdict(agent_service.runtime),
            }
        )

    @app.get("/api/agent/status")
    def agent_status():
        return jsonify(asdict(agent_service.runtime))

    @app.post("/api/agent/cycle")
    def trigger_cycle():
        symbol = request.args.get("symbol")
        try:
            result = agent_service.run_cycle(symbol=symbol)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": "cycle_failed", "details": f"{type(exc).__name__}: {exc}"}), 500

    @app.get("/api/stocks/list")
    def stocks_list():
        return jsonify(_stocks_catalog())

    @app.get("/api/batch_data/<symbol>")
    def batch_data(symbol: str):
        agent, err = _safe_agent()
        if err:
            return jsonify({"error": "agent_unavailable", "details": err}), 503
        symbol = symbol.upper()
        rows: List[Dict[str, Any]] = []
        try:
            batches = agent.batch_manager.load_all_batches_from_disk()
            for batch in batches:
                timestamp = float(batch.get("batch_timestamp", 0))
                item = (batch.get("data") or {}).get(symbol)
                if isinstance(item, dict):
                    rows.append({"timestamp": timestamp, "price": float(item.get("price", 0.0)), "volume": float(item.get("volume", 0.0))})
            rows.sort(key=lambda x: x["timestamp"])
        except Exception as exc:
            return jsonify({"error": "batch_data_failed", "details": str(exc)}), 500
        return jsonify(rows[-500:])

    @app.get("/api/predictions")
    def predictions():
        agent, err = _safe_agent()
        if err:
            return jsonify({"error": "agent_unavailable", "details": err}), 503

        sort_by = request.args.get("sort", "confidence")
        rows: List[Dict[str, Any]] = []
        for stock in _stocks_catalog()[:50]:
            symbol = stock["symbol"]
            current_price = _latest_price(agent, symbol)
            hybrid = agent._generate_hybrid_signal(symbol)
            pred_price = float(hybrid.get("predicted_price", current_price or 0.0))
            confidence = float(hybrid.get("confidence", 0.0))
            delta = pred_price - current_price
            rows.append(
                {
                    "Stocks": stock["name"],
                    "Ticker Symbol": symbol,
                    "Price ($)": round(current_price, 4),
                    "MarketCap": "N/A",
                    "24h Volume": "N/A",
                    "1h Prediction": round(pred_price, 4),
                    "24h Prediction": round(pred_price * 1.002, 4),
                    "1w Prediction": round(pred_price * 1.01, 4),
                    "ELR Component": hybrid.get("source", "unknown"),
                    "Sentiment Impact": round(float(agent.trend_analyzer.get_latest_sentiment(symbol)), 4),
                    "Uncertainty": round(max(0.0, 1.0 - confidence), 4),
                    "p_value": 0.05,
                    "Critical Value": 1.96,
                    "Confidence": round(confidence, 4),
                    "Lower Bound": round(pred_price * (1 - (1 - confidence) * 0.02), 4),
                    "Upper Bound": round(pred_price * (1 + (1 - confidence) * 0.02), 4),
                    "_sort_confidence": confidence,
                    "_sort_delta": delta,
                }
            )

        key_map = {
            "confidence": "_sort_confidence",
            "expected_move": "_sort_delta",
            "ticker": "Ticker Symbol",
            "price": "Price ($)",
        }
        sort_key = key_map.get(sort_by, "_sort_confidence")
        rows.sort(key=lambda x: x.get(sort_key, 0), reverse=True)
        for item in rows:
            item.pop("_sort_confidence", None)
            item.pop("_sort_delta", None)
        return jsonify(rows)

    @app.get("/api/explain/prediction/<symbol>")
    def explain_prediction(symbol: str):
        agent, err = _safe_agent()
        if err:
            return jsonify({"error": "agent_unavailable", "details": err}), 503
        symbol = symbol.upper()
        hybrid = agent._generate_hybrid_signal(symbol)
        explanation = {
            "symbol": symbol,
            "signal": hybrid.get("signal", "hold"),
            "source": hybrid.get("source", "unknown"),
            "confidence": hybrid.get("confidence", 0.0),
            "adaptive_prediction": hybrid.get("predicted_price"),
            "current_price": hybrid.get("current_price"),
            "als_signal": hybrid.get("als_signal", "hold"),
            "rl_signal": hybrid.get("rl_signal", "hold"),
            "market_regime": agent.market_regime,
            "risk_factor": agent._determine_risk_factor(),
        }
        return jsonify(explanation)

    @app.get("/api/indicators")
    def indicators():
        agent, err = _safe_agent()
        if err:
            return jsonify({"error": "agent_unavailable", "details": err}), 503
        tech = agent.backtester._calculate_macd if hasattr(agent.backtester, "_calculate_macd") else None
        signal = agent._generate_hybrid_signal(agent.symbol)
        payload = [
            {"Indicator": "Market Regime", "Value": agent.market_regime, "Status": "info", "Thresholds": "N/A"},
            {"Indicator": "Risk Factor", "Value": round(agent._determine_risk_factor(), 4), "Status": "active", "Thresholds": "0.1 - 2.0"},
            {"Indicator": "Hybrid Confidence", "Value": round(float(signal.get("confidence", 0.0)), 4), "Status": signal.get("source", "unknown"), "Thresholds": "0.0 - 1.0"},
            {"Indicator": "ALS Signal", "Value": signal.get("als_signal", "hold"), "Status": "model", "Thresholds": "buy/sell/hold"},
            {"Indicator": "RL Signal", "Value": signal.get("rl_signal", "hold"), "Status": "model", "Thresholds": "buy/sell/hold"},
        ]
        if tech:
            payload.append({"Indicator": "MACD Engine", "Value": "enabled", "Status": "ok", "Thresholds": "N/A"})
        return jsonify(payload)

    @app.get("/api/diagnostics")
    def diagnostics():
        agent, err = _safe_agent()
        if err:
            return jsonify({"error": "agent_unavailable", "details": err}), 503
        diagnostics = {
            "market_regime": agent.market_regime,
            "circuit_breaker_open": agent.circuit_breaker.is_open(),
            "portfolio_value": round(agent._get_total_portfolio_value(), 4),
            "cash": round(agent.cash, 4),
            "open_positions": len(agent.open_positions),
            "monthly_pnl": round(agent.monthly_pnl, 4),
            "last_cycle_at": agent_service.runtime.last_cycle_at,
            "total_cycles": agent_service.runtime.total_cycles,
        }
        return jsonify(diagnostics)

    @app.get("/api/learning")
    def learning_insights():
        agent, err = _safe_agent()
        if err:
            return jsonify({"error": "agent_unavailable", "details": err}), 503
        signal = agent._generate_hybrid_signal(agent.symbol)
        payload = {
            "training_mode": bool(getattr(agent.learning_agent, "training_mode", False)),
            "feature_history_length": len(agent._feature_history),
            "hybrid_signal": signal.get("signal", "hold"),
            "source": signal.get("source", "unknown"),
            "confidence": signal.get("confidence", 0.0),
            "als_signal": signal.get("als_signal", "hold"),
            "rl_signal": signal.get("rl_signal", "hold"),
            "adaptive_diagnostics": agent.adaptive_learner.get_diagnostics() if hasattr(agent.adaptive_learner, "get_diagnostics") else {},
        }
        return jsonify(payload)

    @app.get("/api/portfolio")
    def portfolio():
        agent, err = _safe_agent()
        if err:
            return jsonify({"error": "agent_unavailable", "details": err}), 503

        open_positions = []
        for symbol, pos in agent.open_positions.items():
            current = _latest_price(agent, symbol)
            shares = float(pos.get("shares", 0.0))
            entry = float(pos.get("entry_price", 0.0))
            unrealized = (current - entry) * shares
            open_positions.append(
                {
                    "symbol": symbol,
                    "direction": pos.get("direction", "long"),
                    "entry_price": entry,
                    "current_price": current,
                    "shares": shares,
                    "initial_size": entry * shares,
                    "unrealized_pnl": unrealized,
                    "timestamp": pos.get("timestamp"),
                }
            )

        peak = max(agent.monthly_peak_portfolio_value, 1e-12)
        drawdown = (peak - agent._get_total_portfolio_value()) / peak

        payload = {
            "portfolio_value": agent._get_total_portfolio_value(),
            "cash": agent.cash,
            "monthly_pnl": agent.monthly_pnl,
            "monthly_peak_value": agent.monthly_peak_portfolio_value,
            "drawdown": drawdown,
            "open_positions": open_positions,
            "financial_goals": agent.financial_goals,
            "market_regime": agent.market_regime,
            "risk_factor": agent._determine_risk_factor(),
        }
        return jsonify(payload)

    @app.get("/api/signals")
    def signals():
        agent, err = _safe_agent()
        if err:
            return jsonify({"error": "agent_unavailable", "details": err}), 503
        symbol = request.args.get("symbol", agent.symbol).upper()
        hybrid = agent._generate_hybrid_signal(symbol)
        return jsonify(
            [
                {"Source": "Adaptive Learning", "Key Activity": f"Pred={hybrid.get('predicted_price', 0):.4f}", "Impact": hybrid.get("als_signal", "hold"), "Confidence": hybrid.get("confidence", 0.0)},
                {"Source": "Learning Agent", "Key Activity": "Policy action projection", "Impact": hybrid.get("rl_signal", "hold"), "Confidence": hybrid.get("confidence", 0.0)},
                {"Source": "Fusion Engine", "Key Activity": hybrid.get("source", "unknown"), "Impact": hybrid.get("signal", "hold"), "Confidence": hybrid.get("confidence", 0.0)},
            ]
        )

    @app.get("/api/trends")
    def trends():
        agent, err = _safe_agent()
        if err:
            return jsonify({"error": "agent_unavailable", "details": err}), 503
        symbol = request.args.get("symbol", agent.symbol).upper()
        sentiment = float(agent.trend_analyzer.get_latest_sentiment(symbol))
        report = agent.trend_analyzer.generate_trend_report(symbol) if hasattr(agent.trend_analyzer, "generate_trend_report") else {}
        return jsonify({"symbol": symbol, "symbol_sentiment": {symbol: sentiment}, "report": report})

    @app.get("/api/data_quality")
    @app.get("/api/dataQuality")
    def data_quality():
        agent, err = _safe_agent()
        if err:
            return jsonify({"error": "agent_unavailable", "details": err}), 503
        symbol = request.args.get("symbol", agent.symbol).upper()
        data = agent.market_handler.fetch_data(symbol, lookback=60)
        quality_ok = bool(agent.data_quality.validate_market_data(data)) if hasattr(agent.data_quality, "validate_market_data") else True
        return jsonify({"symbol": symbol, "records": len(data), "quality_ok": quality_ok})

    @app.get("/api/optimization")
    def optimization():
        agent, err = _safe_agent()
        if err:
            return jsonify({"error": "agent_unavailable", "details": err}), 503
        result = agent.finance_memory.query(data_type="parameter_optimization", tags=["backtesting", "optimization"], limit=1)
        return jsonify(result[0] if result else {})

    @app.get("/api/backtest/metrics")
    def backtest_metrics():
        agent, err = _safe_agent()
        if err:
            return jsonify({"error": "agent_unavailable", "details": err}), 503
        return jsonify(agent.backtester.get_last_metrics())

    @app.get("/api/compliance/disclosures")
    def compliance_disclosures():
        return jsonify(
            {
                "risk_statement": "This dashboard is for analytics/education; not investment advice.",
                "model_registry": "SLAI-FIN-AGENT-v2",
                "data_sources": ["Yahoo", "Polygon", "Finnhub", "AlphaVantage", "Public Sentiment"],
                "last_compliance_check": datetime.now(timezone.utc).isoformat(),
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    # Use threaded mode so UI API polling and agent cycle endpoints can run concurrently.
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
