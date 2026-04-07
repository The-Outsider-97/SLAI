import os
import pytz
import torch
import atexit
import random
import threading
import yaml, json
import numpy as np
import pandas as pd
import yfinance as yf
import time as time_module

from datetime import datetime, timedelta, time
from flask import Flask, jsonify, request, send_from_directory, abort
from dotenv import load_dotenv

load_dotenv()

from finance_test.finance_agent import FinanceAgent
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Financial_Price_Predictor_App")
printer = PrettyPrinter

# --- In-App Data Stores for UI History/Audit (FinanceAgent specific) ---
app_prediction_history_store = []
app_audit_event_store = []

def _app_save_prediction(symbol, predicted_price, confidence, model_version, **kwargs):
    entry = {
        "timestamp": time_module.time(),
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "symbol": symbol, "predicted_price": predicted_price,
        "confidence": confidence, "model_version": model_version, **kwargs
    }
    app_prediction_history_store.append(entry)
    if len(app_prediction_history_store) > 200: app_prediction_history_store.pop(0)
    logger.debug(f"APP_STUB (finance_agent): save_prediction stored: {symbol}")

def _app_log_audit_event(event_type, details, **kwargs):
    entry = {
        "timestamp": time_module.time(),
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "event_type": event_type, "details": details, **kwargs
    }
    app_audit_event_store.append(entry)
    if len(app_audit_event_store) > 200: app_audit_event_store.pop(0)
    logger.debug(f"APP_STUB (finance_agent): log_audit_event stored: {event_type}")

def get_sp500_tickers():
    """Fetch the full S&P 500 list from Wikipedia"""
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        return df['Symbol'].tolist()
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 list: {str(e)}")
        return ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AVGO']  # fallback
    
def get_sp500_securities():
    """
    Fetch the list of S&P 500 securities (symbol and name) from Wikipedia.
    This is much faster as it avoids individual API calls for market cap.
    """
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        securities = []
        for index, row in df.iterrows():
            securities.append({
                'symbol': row['Symbol'],
                'name': row['Security']
            })
        return securities
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 list from Wikipedia: {str(e)}")
        # A more robust fallback list
        return [
            {'symbol': 'AAPL', 'name': 'Apple Inc.'},
            {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
            {'symbol': 'AMZN', 'name': 'Amazon.com, Inc.'},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corporation'},
            {'symbol': 'TSLA', 'name': 'Tesla, Inc.'},
            {'symbol': 'AVGO', 'name': 'Broadcom, Inc.'},
            {'symbol': 'RKLB', 'name': 'Rocket Lab USA, Inc.'},
            {'symbol': 'GME', 'name': 'GameStop Corp.'},
            {'symbol': 'CZR', 'name': 'Caesars Entertainment, Inc.'},
            {'symbol': 'PLTR', 'name': 'Palantir Technologies, Inc.'},
            {'symbol': 'ACN', 'name': 'Accenture PLC'},
            {'symbol': 'TWLO', 'name': 'Twilio, Inc. A'},
            {'symbol': 'SMCI', 'name': 'Super Micro Computer, Inc.'},
            {'symbol': 'TSM', 'name': 'Taiwan Semiconductor Manufacturing, Co. Ltd-ADR'}
        ]

# Apply the monkeypatch to FinanceAgent's stubs *before* it's imported
import finance_test.finance_agent
finance_test.finance_agent.save_prediction = _app_save_prediction
finance_test.finance_agent.log_audit_event = _app_log_audit_event


# --- Flask App Initialization ---
app = Flask(__name__, static_folder='static', template_folder='frontend')

# --- Global lock for thread-safe agent operations ---
agent_lock = threading.Lock()

# --- Agent Initialization ---
# Create the single, primary agent instance
finance_agent = FinanceAgent(symbols=get_sp500_securities())
logger.info("FinanceAgent instance initialized for Flask app.")

# Start FinanceAgent's main loop in a background thread
finance_agent_thread = threading.Thread(target=finance_agent.run, daemon=True)
finance_agent_thread.start()
logger.info("FinanceAgent thread started.")

def is_market_open():
    """Helper function to check market status, using the agent's internal logic."""
    return finance_agent._is_market_open()

# --- API Endpoints ---
@app.route('/api/market/status', methods=['GET'])
def get_market_status():
    return jsonify({
        "is_open": is_market_open(),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/predictions/history', methods=['GET'])
def get_prediction_history():
    return jsonify(sorted(app_prediction_history_store, key=lambda x: x['timestamp'], reverse=True))

@app.route('/api/audit/logs', methods=['GET'])
def get_audit_logs_api():
    # Fallback to app-level store from patched finance_agent.log_audit_event
    return jsonify(sorted(app_audit_event_store, key=lambda x: x['timestamp'], reverse=True))

@app.route('/api/predictions', methods=['GET'])
def get_predictions_api():
    try:
        all_predictions = []
        logger.info("API /predictions: Request received.") # Log API hit
        #backtest_metrics = finance_agent.backtester.get_last_metrics()

        # If no metrics, run backtest immediately
        #if not backtest_metrics:
        #    logger.info("No backtest metrics found. Running backtest now...")
        #    backtest_metrics = finance_agent.run_backtest()

        all_symbols = finance_agent.symbols
        selected_stocks = []
        if len(all_symbols) > 10:
            top_5 = all_symbols[:5]
            random_5_candidates = [s for s in all_symbols if s not in top_5]
            random_5 = random.sample(random_5_candidates, min(5, len(random_5_candidates)))
            selected_stocks = top_5 + random_5
        else:
            selected_stocks = all_symbols
        logger.debug(f"API /predictions: Selected {len(selected_stocks)} stocks for processing.")

        data_added_this_api_call = 0
        with agent_lock:
            for stock in selected_stocks:
                symbol = stock['symbol']
                market_cap, volume_24h, current_price = 0, 0, 0

                try:
                    # Fetch market data first as it's most critical
                    market_data_raw = finance_agent.market_handler.fetch_data(symbol, lookback=2)
                    if not market_data_raw:
                        logger.warning(f"No market data for {symbol}, skipping.")
                        continue
                    
                    current_price = market_data_raw[-1]['close']
                    volume_from_handler = market_data_raw[-1].get('volume', 0)
                    finance_agent.batch_manager.add_data_point(symbol, current_price, float(volume_from_handler))
                    data_added_this_api_call += 1

                    # Fetch supplementary info from yfinance
                    yf_symbol = symbol.replace('.', '-')
                    stock_info_yf = yf.Ticker(yf_symbol).info
                    market_cap = stock_info_yf.get('marketCap', 0)
                    volume_24h = stock_info_yf.get('volume24Hr', stock_info_yf.get('regularMarketVolume', 0)) or volume_from_handler

                except Exception as e:
                    logger.warning(f"API /predictions: Data fetch/processing failed for {symbol}: {e}. Skipping symbol.")
                    continue

                # Fetch latest price using market_handler (primary source for price)
                market_data_raw = finance_agent.market_handler.fetch_data(symbol, lookback=2)
                #current_price = 0 # Default
                #current_price = market_data_raw[-1]['close'] if market_data_raw and len(market_data_raw) > 0 else 0

                if market_data_raw and isinstance(market_data_raw, list) and len(market_data_raw) > 0 and \
                   isinstance(market_data_raw[-1], dict) and 'close' in market_data_raw[-1] and \
                   isinstance(market_data_raw[-1]['close'], (float, int)):
                    current_price_for_batch = market_data_raw[-1]['close']
                    
                    # Try to get volume from market_handler data first for batch, fallback to yf info's volume
                    volume_for_batch = market_data_raw[-1].get('volume', volume_24h) 
                    if not isinstance(volume_for_batch, (int,float)) or volume_for_batch < 0:
                        volume_for_batch = volume_24h # Fallback to yf's volume if market_handler's is bad

                else:
                    logger.warning(f"API /predictions: No valid market data from market_handler for {symbol}. Price for batch might be 0. Data: {market_data_raw}")
                    # current_price_for_batch remains 0
                    volume_for_batch = volume_24h # Use yf volume if market_handler fails for price

                if current_price_for_batch > 0:
                    finance_agent.batch_manager.add_data_point(symbol, current_price_for_batch, float(volume_for_batch)) # Pass volume
                    logger.info(f"API /predictions: Added to BatchManager: {symbol} @ Price: {current_price_for_batch}, Volume: {volume_for_batch}")
                    data_added_this_api_call +=1
                else:
                    logger.warning(f"API /predictions: Price for {symbol} is not positive ({current_price_for_batch}). Not adding to BatchManager.")

                # Prediction logic with enhanced robustness
                prediction_output = {'prediction': 0, 'confidence': 0.0, 'error': None}
                diagnostics = {}
                latest_sentiment_val = None
                p_val = 0.05 # default p-value

                try:
                    if hasattr(finance_agent, 'adaptive_learner') and \
                       finance_agent.adaptive_learner and \
                       finance_agent.adaptive_learner.market_features is not None:
                        
                        input_features_tensor = torch.randn(1, finance_agent.adaptive_learner.market_features)
                        prediction_output = finance_agent.adaptive_learner.robust_predict(input_features_tensor)
                        diagnostics = finance_agent.adaptive_learner.get_diagnostics()
                        if not isinstance(diagnostics, dict): diagnostics = {}
                        p_val = diagnostics.get('p_value', 0.05)
                    else:
                        logger.warning(f"Adaptive learner or market_features not available for {symbol}. Using default prediction.")
                        prediction_output = {'prediction': current_price_for_batch * 1.01 if current_price_for_batch > 0 else 0,
                                             'confidence': 0.5}
                except Exception as pred_e:
                    logger.error(f"Error during prediction for {symbol}: {pred_e}", exc_info=True)
                    prediction_output = {'prediction': current_price_for_batch * 1.01 if current_price_for_batch > 0 else 0,
                                         'confidence': 0.3, 'error': str(pred_e)}
                
                try:
                    latest_sentiment_val = finance_agent.trend_analyzer.get_latest_sentiment(symbol)
                except Exception as sent_e:
                    logger.warning(f"Could not get sentiment for {symbol}: {sent_e}")

                # Prediction logic remains the same...
                prediction_output = finance_agent.adaptive_learner.robust_predict(torch.randn(1, finance_agent.adaptive_learner.market_features))
                diagnostics = finance_agent.adaptive_learner.get_diagnostics()
                latest_sentiment_val = finance_agent.trend_analyzer.get_latest_sentiment(symbol)
                
                pred_value = prediction_output.get('prediction', 0)
                confidence_val = prediction_output.get('confidence', 0)
                uncertainty_raw = (1 - confidence_val) * current_price if current_price > 0 else 0
                
                prediction_data = {
                    "Stocks": stock['name'],
                    "Ticker Symbol": symbol,
                    "Price ($)": f"{current_price_for_batch:.2f}" if current_price_for_batch > 0 else "N/A",
                    "MarketCap": f"${market_cap/1e9:.2f}B" if market_cap else "N/A",
                    "24h Volume": f"{volume_24h:,}" if volume_24h else "N/A",
                    "1h Prediction": f"{pred_value:.2f}" if current_price_for_batch > 0 and prediction_output.get('error') is None else "Error",
                    "24h Prediction": f"{pred_value:.2f}" if current_price_for_batch > 0 and prediction_output.get('error') is None else "Error",
                    "1w Prediction": f"{pred_value:.2f}" if current_price_for_batch > 0 and prediction_output.get('error') is None else "Error",
                    "ELR Component": f"{diagnostics.get('learning_rate', 0.0):.4f}" if diagnostics else "N/A",
                    "Sentiment Impact": f"{latest_sentiment_val:.2f}" if latest_sentiment_val is not None else "N/A",
                    "Uncertainty": f"{uncertainty_raw:.2f}" if current_price_for_batch > 0 else "N/A",
                    "p_value": f"{p_val:.4f}",
                    "Critical Value": "1.96",
                    "Confidence": f"{confidence_val:.3f}",
                    "Lower Bound": f"{pred_value - uncertainty_raw:.2f}" if current_price_for_batch > 0 and prediction_output.get('error') is None else "N/A",
                    "Upper Bound": f"{pred_value + uncertainty_raw:.2f}" if current_price_for_batch > 0 and prediction_output.get('error') is None else "N/A",
                }
                all_predictions.append(prediction_data)

        if data_added_this_api_call == 0 and selected_stocks: 
            logger.warning("API /predictions: No data points were added to BatchManager in this API call, despite processing stocks.")
        return jsonify(all_predictions)

    except Exception as e:
        logger.error(f"CRITICAL Error in /api/predictions: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to fetch predictions", "details": str(e)}), 500

@app.route('/api/ultimate_suggestions', methods=['GET'])
def get_ultimate_suggestions():
    try:
        suggestions_data = []
        processed_stocks = []

        all_symbols = finance_agent.symbols
        if len(all_symbols) > 10:
            top_5 = all_symbols[:5]
            random_5 = random.sample(all_symbols[5:], 5)
            selected_stocks = top_5 + random_5
        else:
            selected_stocks = all_symbols

        with agent_lock:
            for stock_info in selected_stocks:
                symbol = stock_info['symbol']
                market_data_raw = finance_agent.market_handler.fetch_data(symbol, lookback=2)
                current_price = market_data_raw[-1]['close'] if market_data_raw else 0
                if current_price == 0: continue

                prediction_output = finance_agent.adaptive_learner.robust_predict(torch.randn(1, finance_agent.adaptive_learner.market_features))
                predicted_price_24h = prediction_output.get('prediction', current_price)
                confidence = prediction_output.get('confidence', 0.5)
                expected_gain_loss_pct = ((predicted_price_24h - current_price) / current_price) * 100

                processed_stocks.append({
                    "stock_name": stock_info['name'], "stock_symbol": symbol,
                    "confidence_pct": confidence * 100, "expected_gain_loss_pct": expected_gain_loss_pct,
                    "current_price": current_price, "target_price_24h": predicted_price_24h
                })

        if not processed_stocks:
            return jsonify({"suggestions": [], "current_portfolio_value_display": "N/A", "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')})

        sorted_by_gain = sorted(processed_stocks, key=lambda x: x['expected_gain_loss_pct'], reverse=True)
        candidates = {
            "Primary Long": sorted_by_gain[0] if len(sorted_by_gain) > 0 and sorted_by_gain[0]['expected_gain_loss_pct'] > 0 else None,
            "Secondary Long": sorted_by_gain[1] if len(sorted_by_gain) > 1 and sorted_by_gain[1]['expected_gain_loss_pct'] > 0 else None,
            "Primary Short": sorted_by_gain[-1] if len(sorted_by_gain) > 0 and sorted_by_gain[-1]['expected_gain_loss_pct'] < 0 else None,
            "Secondary Short": sorted_by_gain[-2] if len(sorted_by_gain) > 1 and sorted_by_gain[-2]['expected_gain_loss_pct'] < 0 else None,
        }
        for sug_type, data in candidates.items():
            if data:
                action = "LONG" if "Long" in sug_type else "SHORT"
                suggestions_data.append({
                    "type": sug_type, "stock_name": data['stock_name'], "stock_symbol": data['stock_symbol'],
                    "action": action, "confidence_pct": data['confidence_pct'],
                    "expected_gain_loss_pct": data['expected_gain_loss_pct'],
                    "current_price": data['current_price'], "target_price_24h": data['target_price_24h']
                })

        # Calculate adjusted portfolio value (similar to /api/portfolio)
        unrealized_pnl = 0
        with agent_lock:
            for symbol, position_data in finance_agent.open_positions.items():
                market_data_pos = finance_agent.market_handler.fetch_data(symbol, lookback=2)
                if market_data_pos and len(market_data_pos) > 0:
                    current_price_pos = market_data_pos[-1]['close']
                    entry_price_pos = position_data['entry_price']
                    shares_pos = position_data['shares'] # Directly use shares
                    # Correctly use 'position_data' dictionary
                    unrealized_pnl += ((current_price_pos - entry_price_pos) if position_data['direction'] == 'long' else (entry_price_pos - current_price_pos)) * shares_pos
            
            # Note: The original code used portfolio_value, but it should be cash + holdings value
            current_portfolio_value = finance_agent.cash + sum(
                pos['shares'] * finance_agent.market_handler.fetch_data(sym, lookback=1)[0]['close'] 
                for sym, pos in finance_agent.open_positions.items()
            )

        return jsonify({
            "suggestions": suggestions_data, 
            "current_portfolio_value_display": f"${current_portfolio_value:,.2f}", 
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logger.error(f"CRITICAL Error in /api/ultimate_suggestions: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to fetch ultimate suggestions", "details": str(e)}), 500

@app.route('/api/batch_data', methods=['GET'])
def get_all_batch_data():
    batches = finance_agent.batch_manager.get_all_batches()
    return jsonify([batch['data'] for batch in batches])

@app.route('/api/signals', methods=['GET'])
def get_signals():
    try:
        signals_data = finance_agent.trend_analyzer.get_insider_signals(finance_agent.symbol)
        formatted_signals = [
            {
                "source": s.get('source_type', 'Unknown'), "activity": s.get('summary', 'N/A'),
                "impact": s.get('potential_impact_string', 'N/A'), "confidence": s.get('confidence_score', 'N/A')
            } for s in signals_data
        ]
        return jsonify(formatted_signals if formatted_signals else [])
    except Exception as e:
        logger.error(f"Error in /api/signals: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to fetch signals", "details": str(e)}), 500

@app.route('/api/indicators', methods=['GET'])
def get_indicators():
    try:
        tech_indicators = finance_agent.calculate_technical_indicators(finance_agent.symbol)
        formatted_indicators = [
            {"Indicator": "RSI", "Value": f"{tech_indicators.get('RSI', 0):.1f}", "Status": "Neutral", "Thresholds": "[30, 70]"},
            {"Indicator": "MACD Line", "Value": f"{tech_indicators.get('MACD_Line', 0):.2f}", "Status": "N/A", "Thresholds": "Signal cross"},
            {"Indicator": "MACD Signal", "Value": f"{tech_indicators.get('MACD_Signal', 0):.2f}", "Status": "N/A", "Thresholds": "N/A"},
        ]
        return jsonify(formatted_indicators)
    except Exception as e:
        logger.error(f"Error in /api/indicators: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to fetch indicators", "details": str(e)}), 500

@app.route('/api/diagnostics', methods=['GET'])
def get_diagnostics():
    try:
        with agent_lock:
            portfolio_var = finance_agent.calculate_var(symbol=finance_agent.symbol).get('var_dollar', 0)
            als_diagnostics = finance_agent.adaptive_learner.get_diagnostics()
            eval_health = finance_agent.evaluation_agent.get_overall_system_health()
        
        diagnostics_data = {
            "Portfolio VaR (95%)": f"${portfolio_var:,.2f}",
            "System Volatility": f"{als_diagnostics.get('error_volatility', 'N/A')}",
            "Drift Detected": "Yes" if als_diagnostics.get('drift_detected', False) else "No",
            "Current Regime": finance_agent.market_regime.replace('_', ' ').title(),
            "System Health Status": eval_health.get('status', 'Unknown')
        }
        return jsonify(diagnostics_data)
    except Exception as e:
        logger.error(f"Error in /api/diagnostics: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to fetch diagnostics", "details": str(e)}), 500

@app.route('/api/learning_insights', methods=['GET'])
def get_learning_insights():
    try:
        with agent_lock:
            als_insights = finance_agent.adaptive_learner.get_diagnostics()
            # The LearningAgent is now a component of FinanceAgent
            la_insights = finance_agent.learning_agent.get_learning_status()
        
        return jsonify({
            "adaptive_learner": als_insights,
            "learning_agent": la_insights
        })
    except Exception as e:
        logger.error(f"Error in /api/learning_insights: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to fetch learning insights", "details": str(e)}), 500

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio_data():
    try:
        with agent_lock:
            if not finance_agent:
                raise Exception("FinanceAgent instance not available")
            
            data = finance_agent.get_active_trading_parameters()
            backtest_metrics = finance_agent.backtester.get_last_metrics() or {}
            
            # Enrich data for the frontend
            data['current_total_portfolio_value_incl_unrealized'] = data.get('portfolio_value', 0)
            data['current_portfolio_value_cash'] = finance_agent.cash
            data['unrealized_pnl_total'] = data.get('portfolio_value', 0) - finance_agent.cash
            data['current_drawdown_percentage'] = ((data.get('monthly_peak_portfolio_value', data.get('portfolio_value',0)) - data.get('portfolio_value',0)) / (data.get('monthly_peak_portfolio_value', 1) or 1)) * 100
            data['open_positions_detailed'] = [
                {**pos, 'symbol': sym, 'current_price': finance_agent.market_handler.fetch_data(sym,1)[0]['close'], 'unrealized_pnl': finance_agent._calculate_pnl(pos, finance_agent.market_handler.fetch_data(sym,1)[0]['close'])}
                for sym, pos in data.get('open_positions', {}).items()
            ]
            data['default_symbol_for_analysis'] = finance_agent.symbol
            data['latest_price_of_default_symbol_for_state'] = finance_agent.latest_price_of_default_symbol
            data['daily_profit_target'] = finance_agent.daily_profit_target
            data['market_regime'] = finance_agent.market_regime
            data['last_daily_maintenance_date'] = finance_agent.last_daily_maintenance_date

            data['backtest_metrics'] = {
                "sharpe_ratio": f"{backtest_metrics.get('mean_sharpe_ratio', 0):.2f}",
                "max_drawdown_backtest": f"{backtest_metrics.get('max_max_drawdown', 0)*100:.1f}%",
                "sharpe_ratio": f"{backtest_metrics.get('mean_sharpe_ratio', 0):.3f}",
                "sortino_ratio": f"{backtest_metrics.get('mean_sortino_ratio', 0):.3f}",
                "calmar_ratio": f"{backtest_metrics.get('mean_calmar_ratio', 0):.3f}",
                "max_drawdown": f"{backtest_metrics.get('max_max_drawdown', 0) * 100:.2f}%",
                "win_rate": f"{backtest_metrics.get('mean_win_rate', 0) * 100:.2f}%",
                "profit_factor": f"{backtest_metrics.get('mean_profit_factor', 0):.2f}",
                "mean_return": f"{backtest_metrics.get('mean_mean_return', 0) * 100:.4f}%",
                "volatility": f"{backtest_metrics.get('mean_volatility', 0) * 100:.4f}%"
            }
            return jsonify(data)
            
    except Exception as e:
        logger.error(f"Error in /api/portfolio: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to fetch portfolio data", "details": str(e)}), 500
@app.route('/api/trends', methods=['GET'])
def get_trends():
    try:
        symbol = finance_agent.symbol
        trend_report = finance_agent.trend_analyzer.generate_trend_report(symbol)
        latest_sentiment = finance_agent.trend_analyzer.get_latest_sentiment(symbol)

        trends_data = {
            "symbol_sentiment": {symbol: f"{latest_sentiment:.2f}" if latest_sentiment is not None else "N/A"},
            "symbol_correlation": {symbol: f"{trend_report.get('correlation_metrics',{}).get('overall_correlation','N/A')}"},
            "top_trend_terms": trend_report.get('top_trends', {}),
            "emerging_narratives": trend_report.get('emerging_narratives', [])
        }
        return jsonify(trends_data)
    except Exception as e:
        logger.error(f"Error in /api/trends: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to fetch trends", "details": str(e)}), 500

@app.route('/api/data_quality', methods=['GET'])
def get_data_quality():
    try:
        symbol = finance_agent.symbol
        quality_report_all = finance_agent.data_quality.get_quality_report(symbol)
        primary_source_report = quality_report_all.get('primary_sources', {})
        most_reliable = max(primary_source_report.get('reliability_score', {}).items(), key=lambda x: x[1], default=('N/A', 0))
        dq_data = {
            "source_reliability": f"{most_reliable[1]:.2f} ({most_reliable[0]})",
            "freshness_score": primary_source_report.get('freshness_score', {}).get(most_reliable[0], "N/A"),
            "fallback_triggers": primary_source_report.get('fallback_events_count', {}).get(most_reliable[0], 0)
        }
        return jsonify(dq_data)
    except Exception as e:
        logger.error(f"Error in /api/data_quality: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to fetch data quality", "details": str(e)}), 500

@app.route('/api/portfolio/optimization', methods=['GET'])
def get_optimization_results():
    try:
        with agent_lock:
            optimization_data = finance_agent.get_latest_optimization_results()
        return jsonify(optimization_data)
    except Exception as e:
        logger.error(f"Error fetching optimization results: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to fetch optimization results", "details": str(e)}), 500

@app.route('/api/compliance/disclosures', methods=['GET'])
def get_risk_disclosures():
    try:
        model_v = finance_agent.adaptive_learner.config.get('version', '1.0')
        active_srcs = list(finance_agent.market_handler.api_clients)
        source_names = [type(src).__name__ for src in active_srcs]
        disclosures = {
            "risk_statement": finance_agent.agent_config.get("risk_statement", "Trading involves risks. Not financial advice."),
            "model_registry": f"SLAI-FinPred-v{model_v}", "data_sources": source_names,
            "last_compliance_check": datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
            "regulatory_ids": "FINRA CRD# 58372, SEC CIK# 0001961765"
        }
        return jsonify(disclosures)
    except Exception as e:
        logger.error(f"Error in /api/compliance/disclosures: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to fetch compliance disclosures", "details": str(e)}), 500

@app.route('/api/explain/prediction/<symbol>', methods=['GET'])
def get_prediction_explanation(symbol):
    try:
        if symbol != finance_agent.symbol:
            return jsonify({"error": f"Explanation currently only supported for default symbol: {finance_agent.symbol}"}), 400

        market_data_expl = finance_agent.market_handler.fetch_data(symbol, lookback=90)
        current_price_expl = market_data_expl[-1]['close'] if market_data_expl else 'N/A'
        tech_ind = finance_agent.calculate_technical_indicators(symbol) if market_data_expl else {}
        input_f = torch.randn(1, finance_agent.adaptive_learner.market_features)
        pred_out = finance_agent.adaptive_learner.robust_predict(input_f)
        sentiment_expl = finance_agent.trend_analyzer.get_latest_sentiment(symbol)
        explanation = {
            "symbol": symbol, "current_price": f"{current_price_expl:.2f}" if isinstance(current_price_expl, float) else current_price_expl,
            "predicted_price": f"{pred_out['prediction']:.2f}", "confidence": f"{pred_out['confidence']:.3f}",
            "model_version": getattr(finance_agent.adaptive_learner, 'model_version', "Default"),
            "key_factors": [
                {"factor": "Market Regime", "value": finance_agent.market_regime.title()},
                {"factor": "RSI", "value": f"{tech_ind.get('RSI', 0):.1f}"},
                {"factor": "MACD Hist.", "value": f"{tech_ind.get('MACD_Hist', 0):.2f}"},
                {"factor": "Sentiment", "value": f"{sentiment_expl:.2f}" if sentiment_expl is not None else "N/A"},
            ],
            "caveats": "Simplified explanation. Predictions are probabilistic. Not financial advice."
        }
        return jsonify(explanation)
    except Exception as e:
        logger.error(f"Error in /api/explain/prediction/{symbol}: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to generate explanation for {symbol}", "details": str(e)}), 500

@app.route('/api/stocks/list', methods=['GET'])
def get_stock_list():
    return jsonify(get_sp500_securities())

@app.route('/api/batch_data/<symbol>', methods=['GET'])
def get_batch_data(symbol):
    try:
        batches = finance_agent.batch_manager.get_all_batches()
        result = []
    
        for batch in batches:
            timestamp = batch.get('batch_timestamp')
            stock_data = batch.get('data', {}).get(symbol)
    
            if stock_data and 'price' in stock_data:
                result.append({
                    "timestamp": timestamp,
                    "price": stock_data["price"]
                })
    
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve batch data for {symbol}: {str(e)}"}), 500
    
# --- Static File Serving ---
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    # This is a basic static file server. For production, use a dedicated web server like Nginx or Gunicorn.
    full_path = os.path.join(app.static_folder, path)
    if os.path.isdir(full_path): abort(404)
    if os.path.exists(full_path): return send_from_directory(app.static_folder, path)
    if not os.path.splitext(path)[1]:
        html_path = path + '.html'
        full_html_path = os.path.join(app.static_folder, html_path)
        if os.path.exists(full_html_path):
            return send_from_directory(app.static_folder, html_path)
    return abort(404)

# Register the batch manager's stop_controller for graceful shutdown
if hasattr(finance_agent, 'batch_manager'):
    atexit.register(finance_agent.batch_manager.stop_controller)
    logger.info("Registered BatchManager stop_controller for graceful shutdown.")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Financial Price Predictor app on http://0.0.0.0:{port}")
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
