

import os
import time
import pytz
import torch
import threading
import numpy as np
import pandas as pd
import torch.nn as nn
import yfinance as yf

from typing import Dict, List, Optional
from scipy.stats import norm
from datetime import datetime, timedelta, time as dt_time
from datetime import datetime

from finance_test.config_loader import load_global_config, get_config_section
from finance_test.database import save_prediction, log_audit_event
from finance_test.env.stock_trading_env import StockTradingEnv
from finance_test.core.market_data_handler import MarketDataHandler
from finance_test.core.data_quality_monitor import DataQualityMonitor
from finance_test.core.adaptive_learning import AdaptiveLearningSystem
from finance_test.core.cultural_trend_analyzer import CulturalTrendAnalyzer
from finance_test.core.investor_tracker import InvestorTracker
from finance_test.core.finance_memory import FinanceMemory 
from finance_test.core.batch_manager import BatchManager
from finance_test.core.backtester import Backtester
from src.agents.planning.planning_types import Task, TaskType, Any
from src.agents.knowledge.knowledge_cache import KnowledgeCache
from src.agents.collaborative.shared_memory import SharedMemory
from src.agents.agent_factory import AgentFactory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Finance Agent")
printer = PrettyPrinter

# Stubbing database functions if not fully set up, as per original code context
save_prediction = lambda *args, **kwargs: logger.info(f"DB save_prediction stubbed. Args: {args}, Kwargs: {kwargs}")
log_audit_event = lambda *args, **kwargs: logger.info(f"DB log_audit_event stubbed. Args: {args}, Kwargs: {kwargs}")


class CircuitBreaker:
    def __init__(self):
        self.config = load_global_config()
        self.cb_config = get_config_section('circuit_breaker')
        self.threshold = self.cb_config.get('threshold')
        self.reset_timeout = self.cb_config.get('reset_timeout')
        self.failure_count = 0
        self.last_failure = 0
        self.state = "CLOSED"

    def trip(self):
        self.failure_count += 1
        self.last_failure = time.time()
        if self.failure_count >= self.threshold:
            self.state = "OPEN"
        
    def is_open(self):
        if time.time() - self.last_failure > self.reset_timeout:
            self.failure_count = 0
            self.state = "CLOSED"
        return self.state == "OPEN"

class FinanceAgent(nn.Module):
    """
    An autonomous financial agent that orchestrates data analysis, trend monitoring,
    adaptive learning, and trade execution to achieve defined financial goals.
    
    SharedMemory
        - Inter-agent coordination & real-time state sharing
        - Portfolio state, trade signals, market regime
        - High-frequency, low-latency
    FinanceMemory
        - Agent-specific financial data storage
        - Batch data, backtest results, model states
        - Structured query access
    """
    def __init__(self, symbols=None):
        super().__init__()
        self.config = load_global_config()
        self.position_size = self.config.get('position_size')  # % of portfolio per trade
        self.stop_loss = self.config.get('stop_loss')  # 5% stop loss
        self.take_profit = self.config.get('take_profit')  # 10% take profit

        self.agent_config = get_config_section('finance_agent')
        self.bt_config = get_config_section('backtester')
        self.adaptive_learner_config = get_config_section('adaptive_learning_system')
        self.learning_agent_config = get_config_section('learning_agent')
        self.tracker = InvestorTracker()

        # Initialize core components
        self.circuit_breaker = CircuitBreaker()
        self.data_quality = DataQualityMonitor()
        self.market_handler = MarketDataHandler({
            "alpha_vantage": os.getenv("ALPHA_VANTAGE_KEY"),
            "finnhub": os.getenv("FINNHUB_KEY"),
            "polygon": os.getenv("POLYGON_API_KEY"),
            "enable_yahoo": True
        })
        self.adaptive_learner = AdaptiveLearningSystem()
        self.trend_analyzer = CulturalTrendAnalyzer()
        self.investor_tracker = self.tracker
        self.finance_memory = FinanceMemory()
        self.local_memory = self.finance_memory
        self.batch_manager = BatchManager()
        self.batch_manager.memory = self.finance_memory
        self.backtester = Backtester(tracker=self.tracker)
        self.backtester.finance_memory = self.finance_memory
        self.backtester.batch_manager = self.batch_manager
        self.knowledge_cache = KnowledgeCache()

        # State management
        self.portfolio_value = float(self.config.get('initial_capital'))
        self.cash = self.portfolio_value  # Initialize cash with the starting capital
        self.holdings_value = 0.0 # Initialize holdings value
        self.shares_held = {} # Use a dict to track shares per symbol
        self.open_positions = {}
        self.default_analysis_symbol = self.agent_config.get('default_symbol', 'CZR')
        self.symbol = self.default_analysis_symbol # FinanceAgent's current primary symbol

        self.latest_price_of_default_symbol = 0.0
        self.market_regime = 'neutral'
        self.indicators_for_default_symbol = {}
        self.last_trade_time = 0
        self.simulation_mode = self.agent_config.get('simulation_mode', False)
        self.training_mode = self.agent_config.get('training_mode', True)

        self.symbols = symbols or [{'symbol': self.default_analysis_symbol, 'name': 'Default Analysis Stock'}]

        self.last_update_time = 300
        self.daily_profit_target = 0.0
        self.last_daily_maintenance_date = None

        self.current_symbol_index = 0 # For rotate_symbol if used

        # --- Financial Goals & Tracking ---
        self.financial_goals = {
            'monthly_profit': {
                'target': 750.0, # target earnings in 30 days
                'status': 'in_progress', # in_progress, met, failed
            },
            'max_drawdown': {
                'limit': 0.15, # 15%
                'status': 'ok', # ok, breached
            },
            'sentiment_risk': {
                'threshold': 0.5,
                'risk_factor': 1.5, # Increase position size by 50%
            }
        }
        self.current_month = datetime.now().month
        self.monthly_pnl = 0.0
        self.monthly_peak_portfolio_value = self.portfolio_value
        self.closed_trades_this_month = []
        
        # --- Environment & Agent Initialization ---
        # Environment for LearningAgent: This sets the expected state/action space for self.learning_agent
        self.slai_trading_env = StockTradingEnv(
            symbol=self.default_analysis_symbol, 
            initial_balance=self.portfolio_value
        )
        logger.info(f"FinanceAgent: Initializing LearningAgent with env for symbol '{self.default_analysis_symbol}', obs_space_shape: {self.slai_trading_env.observation_space.shape[0]}")
        
        # Initialize other agents
        self.shared_memory = SharedMemory()
        self.agent_factory = AgentFactory(config={"knowledge_agent": {"cache": self.knowledge_cache}})
        self.evaluation_agent = self.agent_factory.create("evaluation", self.shared_memory)
        self.planning_agent = self.agent_factory.create("planning", self.shared_memory)
        self.execution_agent = self.agent_factory.create("execution", self.shared_memory)
        self.knowledge_agent = self.agent_factory.create("knowledge", self.shared_memory)
        self.learning_agent = self.agent_factory.create("learning", self.shared_memory, env=self.slai_trading_env)
        self.learning_agent.training_mode = False

        self.knowledge_agent.cache = self.knowledge_cache

        self.env_thread = None
        self.knowledge_update_thread = threading.Thread(target=self._update_knowledge_periodically, daemon=True)
        self.knowledge_update_thread.start()
        self.latest_insights = {}
        self._init_trading_environment()
        self._register_financial_tasks()
        self._run_initial_backtest()

    def _run_initial_backtest(self):
        """Runs an initial backtest on startup to ensure metrics are available."""
        logger.info("Attempting to run initial backtest on startup...")
        try:
            # Load all available historical data from batch files
            historical_data = self.backtester.load_batch_data(self.batch_manager)

            if historical_data.empty or len(historical_data) < self.bt_config.get('train_window', 180):
                 logger.warning(f"Not enough historical data for initial backtest. Found {len(historical_data)} records. Generating mock data to populate UI.")
                 dates = pd.date_range(start='2024-01-01', periods=365)
                 prices = np.cumprod(1 + np.random.normal(0.0005, 0.02, 365)) * 150
                 mock_data = pd.DataFrame({
                     'datetime': dates, 'symbol': 'MOCK',
                     'close': prices, 'volume': np.random.randint(100000, 1000000, 365)
                 })
                 historical_data = mock_data

            logger.info("Running initial walk-forward test...")
            self.run_backtest(historical_data)
            logger.info("Initial backtest completed successfully. Metrics are now available.")
        except Exception as e:
            logger.error(f"Failed to run initial backtest: {e}", exc_info=True)

    def _update_knowledge_periodically(self):
        """Periodically updates financial knowledge insights"""
        logger.info("Starting initial background setup: running backtest and optimization...")
        historical_data = self.backtester.load_batch_data(self.batch_manager)
        if historical_data.empty:
            logger.warning("No historical data found for initial setup. Skipping backtest and optimization.")
            return

        while True:            
            self.optimize_trading_parameters(historical_data)
            logger.info("Initial background setup completed.")

            try:
                self.latest_insights = self._retrieve_financial_insights()
                logger.info(f"Updated financial insights: {len(self.latest_insights)} concepts")
            except Exception as e:
                logger.error(f"Knowledge update failed: {e}")
            time.sleep(3600)  # Update hourly

    def _retrieve_financial_insights(self) -> Dict[str, Any]:
        """Retrieves relevant financial knowledge using KnowledgeAgent"""
        concepts = [
            "market_regime", 
            f"{self.symbol}_risk_factors",
            "position_sizing_rules",
            "compliance_guidelines"
        ]
        
        insights = {}
        for concept in concepts:
            try:
                # Use KnowledgeAgent's contextual search for financial concepts
                results = self.knowledge_agent.contextual_search(concept)
                insights[concept] = [doc['text'] for _, doc in results[:3]]  # Top 3 insights
            except Exception as e:
                logger.warning(f"Failed to retrieve {concept}: {e}")
                
        return insights
    
    def get_latest_optimization_results(self) -> Dict:
        """Retrieves the latest parameter optimization results from memory."""
        logger.info("Querying FinanceMemory for latest parameter optimization results...")
        results = self.finance_memory.query(
            data_type='parameter_optimization',
            tags=['backtesting', 'optimization'],
            limit=1 
        )
        if results:
            logger.info(f"Found {len(results)} optimization result(s). Returning the latest.")
            return results[0].get('data', {})
        return {}

    def _is_market_open(self):
        now_utc = datetime.now(pytz.utc)
        if not self._is_weekday(now_utc):
            self.simulation_mode = False
            self.env_thread.join(timeout=5)
            return False
        logger.info("Stopped RL training mode")
        if self._is_trading_holiday(now_utc):
            return False
        return self._is_within_trading_hours(now_utc)
    
    def _is_weekday(self, dt_obj):
        return dt_obj.weekday() < 5  # Monday-Friday

    def _is_trading_holiday(self, dt_obj):
        # This is a simplified holiday check. For production, use a proper holiday calendar library.
        ny_tz = pytz.timezone('America/New_York')
        ny_date = dt_obj.astimezone(ny_tz).date()
        holidays_month_day = [
            (1, 1),   # New Year's Day
            (7, 4),   # Independence Day
            (12, 25), # Christmas
        ]
        if ny_date.month == 1 and ny_date.weekday() == 0 and 15 <= ny_date.day <= 21:
            return True
        if ny_date.month == 5 and ny_date.weekday() == 0 and ny_date.day >= (31-6):
            import calendar
            last_day_of_month = calendar.monthrange(ny_date.year, ny_date.month)[1]
            if ny_date.day > last_day_of_month - 7:
                return True
        return (ny_date.month, ny_date.day) in holidays_month_day

    def _is_within_trading_hours(self, dt_obj):
        ny_tz = pytz.timezone('America/New_York')
        ny_time = dt_obj.astimezone(ny_tz).time()
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        return market_open <= ny_time <= market_close

    def _get_trading_days_in_month(self, year, month):
        import calendar
        num_days = calendar.monthrange(year, month)[1]
        trading_days = 0
        for day_val in range(1, num_days + 1):
            date_obj = datetime(year, month, day_val).replace(tzinfo=pytz.utc)
            if self._is_weekday(date_obj) and not self._is_trading_holiday(date_obj):
                trading_days += 1
        return trading_days

    def _init_trading_environment(self):
        """Initializes the trading environment for simulation or live trading."""
        self.trading_env = self.slai_trading_env

    def _register_financial_tasks(self):
        """Defines and registers the financial tasks and their decomposition methods with the PlanningAgent."""

        # --- Define Primitive Task Effects (the actual logic) ---
        def effect_collect_data(state, params):
            symbol = params['symbol']
            data = self.collect_and_process_data(symbol)
            state[f'data_collected_{symbol}'] = bool(data)
            logger.info(f"PLAN_EFFECT: Data collected for {symbol}")
            return state

        def effect_generate_signal(state, params):
            symbol = params['symbol']
            signal_info = self._generate_signal(symbol)
            state[f'signal_{symbol}'] = signal_info['signal']
            state[f'confidence_{symbol}'] = signal_info['confidence']
            logger.info(f"PLAN_EFFECT: Signal for {symbol} is {signal_info['signal']}")
            return state

        def effect_determine_risk(state, params):
            risk_factor = self._determine_risk_factor()
            state['risk_factor'] = risk_factor
            logger.info(f"PLAN_EFFECT: Risk factor determined as {risk_factor}")
            return state
            
        def effect_execute_trade(state, params):
            symbol = params['symbol']
            signal = state.get(f'signal_{symbol}', 'hold')
            confidence = state.get(f'confidence_{symbol}', 0.5)
            risk_factor = state.get('risk_factor', 1.0)
            
            # This calls the actual trade execution logic
            self._execute_trade_action(symbol, signal, confidence, risk_factor)
            state[f'trade_executed_{symbol}'] = True
            return state

        # --- Define Primitive Tasks ---
        collect_data_task = Task(
            name="CollectData", task_type=TaskType.PRIMITIVE,
            preconditions=[lambda state, params: params.get('symbol') is not None],
            effects=[effect_collect_data]
        )
        
        generate_signal_task = Task(
            name="GenerateSignal", task_type=TaskType.PRIMITIVE,
            preconditions=[lambda state, params: state.get(f"data_collected_{params.get('symbol')}", False)],
            effects=[effect_generate_signal]
        )

        determine_risk_task = Task(
            name="DetermineRiskFactor", task_type=TaskType.PRIMITIVE,
            preconditions=[lambda state, params: self.financial_goals is not None],
            effects=[effect_determine_risk]
        )
        
        execute_trade_task = Task(
            name="ExecuteTrade", task_type=TaskType.PRIMITIVE,
            preconditions=[lambda state, params: state.get(f"signal_{params.get('symbol')}", 'hold') != 'hold'],
            effects=[effect_execute_trade]
        )
        
        # --- Define Abstract Task ---
        optimize_portfolio_task = Task(
            name="OptimizePortfolio", task_type=TaskType.ABSTRACT,
            methods=[
                [collect_data_task, generate_signal_task, determine_risk_task, execute_trade_task]
            ]
        )

        # --- Register all tasks with the Planning Agent ---
        for task in [collect_data_task, generate_signal_task, determine_risk_task, execute_trade_task, optimize_portfolio_task]:
            self.planning_agent.register_task(task)
        
        logger.info("Registered all financial tasks with the Planning Agent.")

    def _execute_trade_action(self, symbol: str, signal: str, confidence: float, risk_factor: float):
        """The actual low-level action of opening or closing a position."""
        # Compliance verification
        if not self._check_trade_compliance(signal, symbol):
            logger.warning(f"Trade blocked by compliance: {signal} {symbol}")
            return

        if signal == "hold":
            return
            
        current_price = self.market_handler.fetch_data(symbol, lookback=1)[0]['close']

        # Close logic
        if symbol in self.open_positions:
            position = self.open_positions[symbol]
            if (position['direction'] == 'long' and signal == 'sell') or \
               (position['direction'] == 'short' and signal == 'buy'):
                self._close_position(symbol, current_price)
            
        # Open logic
        if signal == "buy" and symbol not in self.open_positions:
            buying_power = self._calculate_buying_power()
            position_value = self.portfolio_value * self.position_size * risk_factor * confidence
            
            if buying_power > position_value > 0:
                shares = position_value / current_price
                self.open_positions[symbol] = {
                    'direction': 'long', 'entry_price': current_price, 'shares': shares, 'timestamp': time.time()
                }
                self.cash -= position_value
                log_audit_event('position_opened', {'symbol': symbol, 'details': self.open_positions[symbol]})
                save_prediction('position_opened', {'symbol': symbol, 'details': self.open_positions[symbol]})
                logger.info(f"ACTION: Opened LONG position for {shares:.2f} shares of {symbol} at ${current_price:.2f}")


    def _check_trade_compliance(self, action: str, symbol: str) -> bool:
        """Checks trade against regulatory guidelines"""
        guidelines = self.latest_insights.get("compliance_guidelines", [])
        for guideline in guidelines:
            if "restricted" in guideline and symbol in guideline:
                logger.warning(f"Compliance restriction for {symbol}: {guideline}")
                return False
                
            if "short_selling" in guideline and action == "sell":
                logger.info(f"Applying short selling restrictions: {guideline}")
                return False
                
        return True


    def run_execution_loop(self):
        """Main operational loop, running at 10-minute intervals during market hours."""
        logger.info("Starting main execution loop.")
        last_evaluation_time = time.time()
        
        while True:
            if self.circuit_breaker.is_open():
                logger.warning("Circuit breaker is OPEN. Skipping trading cycle.")
                time.sleep(60)
                continue
    
            if self._is_market_open():
                logger.info("Market is open. Starting execution cycle.")
                try:
                    self.monitor_trends(self.symbol)
                    self.collect_and_process_data(self.symbol)
                    
                    # Generate signal and execute trade
                    signal = self._generate_signal(self.symbol)
                    if signal != 'hold':
                        risk_factor = self._determine_risk_factor()
                        trade_quantity = int((self._calculate_buying_power() * self.position_size * risk_factor) / self.latest_price_of_default_symbol)
                        if trade_quantity > 0:
                            self._generate_trade(self.symbol, signal, trade_quantity)

                    self._check_and_update_financial_goals()
                    self._sync_portfolio_state()
                except Exception as e:
                    logger.error(f"Error in execution loop: {e}", exc_info=True)
                    self.circuit_breaker.trip()
                current_time = time.time()
                if current_time - last_evaluation_time > 14400:  # 4 hours
                    self._run_periodic_evaluation()
                    last_evaluation_time = current_time
            else:
                logger.info("Market is closed. Standing by.")
            
            time.sleep(900) # Wait for 10 minutes
            
            # Knowledge-based regime detection
            self.market_regime = self._detect_market_regime()

    def monitor_trends(self, symbol: str):
        """Monitors cultural and sentiment trends for a symbol."""
        logger.info(f"Monitoring trends for {symbol}...")
        try:
            sentiment_snippets = self.trend_analyzer.sentiment_scraper.get_sentiment_snippets(symbol)
            aggregated_text = self.trend_analyzer._aggregate_text(sentiment_snippets)
            
            if aggregated_text and aggregated_text != "No content found":
                self.trend_analyzer.update_trends(aggregated_text, symbol)
                report = self.trend_analyzer.generate_trend_report(symbol)
                
                self.finance_memory.add_financial_data(
                    data=report,
                    data_type='trend_report',
                    tags=[f'trend_report_{symbol}'],
                    priority='medium'
                )
                logger.info(f"Trend report for {symbol} generated and stored.")
            else:
                logger.warning(f"No new text content found for {symbol} to update trends.")

        except Exception as e:
            logger.error(f"Error during trend monitoring for {symbol}: {e}", exc_info=True)
            
    def _execute_trading_logic(self, symbol: str):
        """
        Core logic for making a trading decision in a single cycle.
        """
        # 1. Check Circuit Breaker
        if self.circuit_breaker.is_open():
            logger.warning(f"Circuit breaker is OPEN for {symbol}. Skipping trade.")
            return

        # 2. Generate Trading Signal using Adaptive Learning System
        signal = self._generate_signal(symbol)
        
        # 3. Fetch current market price
        try:
            market_data = self.market_handler.fetch_data(symbol, lookback=1)
            if not market_data:
                logger.warning(f"Could not fetch current price for {symbol}. Skipping trading logic.")
                return
            current_price = market_data[0]['close']
        except Exception as e:
            logger.error(f"Failed to fetch current price for {symbol}: {e}")
            return
            
        # 4. Get Risk Factor based on financial goals
        risk_factor = self._determine_risk_factor()

        # 5. Execute Trade based on signal
        if signal == 'buy' and self._calculate_buying_power() > 0:
            position_value = self.portfolio_value * self.position_size * risk_factor
            quantity_to_buy = int(position_value / current_price)
            
            if quantity_to_buy > 0:
                self.open_positions[symbol] = {
                    'direction': 'long', 
                    'shares': quantity_to_buy, 
                    'entry_price': current_price
                }
                self.cash -= quantity_to_buy * current_price * (1 + self.backtester.transaction_cost)
                logger.info(f"Executed BUY for {quantity_to_buy} shares of {symbol} at ${current_price:.2f}")

        elif signal == 'sell':
            if symbol in self.open_positions and self.open_positions[symbol]['direction'] == 'long':
                self._close_position(symbol, current_price)
                logger.info(f"Executed SELL to close position for {symbol} at ${current_price:.2f}")

        # 6. Update the adaptive learning model with new data
        self._update_adaptive_model(symbol)

    def _check_and_update_financial_goals(self):
        """Checks and updates the status of defined financial goals."""
        # Update monthly P&L
        if datetime.now().month != self.current_month:
            self.current_month = datetime.now().month
            self.monthly_pnl = 0.0
            self.monthly_peak_portfolio_value = self.portfolio_value
            self.closed_trades_this_month = []

        # Check profit goal
        if self.monthly_pnl >= self.financial_goals['monthly_profit']['target']:
            self.financial_goals['monthly_profit']['status'] = 'met'
        
        # Check drawdown
        current_portfolio_value = self._get_total_portfolio_value()
        current_drawdown = (self.monthly_peak_portfolio_value - current_portfolio_value) / self.monthly_peak_portfolio_value
        if current_drawdown > self.financial_goals['max_drawdown']['limit']:
            self.financial_goals['max_drawdown']['status'] = 'breached'
        else:
            self.financial_goals['max_drawdown']['status'] = 'ok'
            if current_portfolio_value > self.monthly_peak_portfolio_value:
                self.monthly_peak_portfolio_value = current_portfolio_value

        # Compare against industry benchmarks
        benchmarks = self.knowledge_agent.get_references_for_concepts(
            ["portfolio_benchmarks"], 
            k=3
        )
        if benchmarks:
            avg_benchmark = sum(float(b.split()[0]) for b in benchmarks if b.replace('.','',1).isdigit()) / len(benchmarks)
            if self.monthly_pnl < avg_benchmark * 0.8:
                logger.warning(f"Underperforming benchmarks: {self.monthly_pnl:.2f} vs {avg_benchmark:.2f}")

    def _calculate_portfolio_value(self) -> float:
        """Calculates current total portfolio value (cash + unrealized P&L)."""
        unrealized_pnl = 0.0
        for symbol, pos in self.open_positions.items():
            try:
                # Fetch the latest price to calculate unrealized P&L
                latest_data = self.market_handler.fetch_data(symbol, lookback=1)
                if latest_data:
                    current_price = latest_data[0]['close']
                    unrealized_pnl += self._calculate_pnl(pos, current_price)
            except Exception as e:
                logger.error(f"Could not fetch price for {symbol} to calculate unrealized P&L: {e}")
        return self.cash + sum(pos['entry_price'] * pos['shares'] for pos in self.open_positions.values()) + unrealized_pnl

    def _determine_risk_factor(self) -> float:
        """Determines a risk multiplier based on goal statuses and sentiment."""
        risk_factor = 1.0
        # Adjust based on drawdown
        if self.financial_goals['max_drawdown']['status'] == 'breached':
            logger.warning("Max drawdown breached. Reducing risk factor significantly.")
            risk_factor *= 0.5
        
        # Adjust based on profit goal progress
        progress = self.monthly_pnl / self.financial_goals['monthly_profit']['target'] if self.financial_goals['monthly_profit']['target'] > 0 else 1.0
        if progress < 0.25:
            risk_factor *= 1.2  # Increase risk if far behind
        elif progress > 0.75:
            risk_factor *= 0.8  # Decrease risk when close to target

        # Adjust based on sentiment
        sentiment = self.trend_analyzer.get_latest_sentiment(self.symbol)
        if sentiment > self.financial_goals['sentiment_risk']['threshold']:
            risk_factor *= self.financial_goals['sentiment_risk']['risk_factor']

        # Apply position sizing rules from knowledge
        sizing_rules = self.latest_insights.get("position_sizing_rules", [])
        for rule in sizing_rules:
            if "volatility" in rule and "reduce position" in rule:
                risk_factor *= 0.8
                logger.info("Applying volatility position sizing rule")

        return max(0.1, min(2.0, risk_factor)) # Clamp risk factor

    def collect_and_process_data(self, symbol: str) -> Optional[List[Dict]]:
        """Collects, processes, and batches market data for a symbol."""
        try:
            data = self.market_handler.fetch_data(symbol, lookback=30)
            if data and data[-1]:
                latest_price = data[-1]['close']
                latest_volume = data[-1].get('volume', 0)
                self.batch_manager.add_data_point(symbol, latest_price, latest_volume)
                self.latest_price_of_default_symbol = latest_price
                self.indicators_for_default_symbol = self.calculate_technical_indicators(symbol)
                self._update_adaptive_model(symbol)
                return data
            else:
                logger.warning(f"No data fetched for {symbol}.")
                return None
        except Exception as e:
            logger.error(f"Data collection failed for {symbol}: {e}", exc_info=True)
            return None
    
    def run(self):
        """The main entry point for the agent's execution loop."""
        logger.info(f"FinanceAgent main loop started for symbol: {self.symbol}")
        # In a real scenario, this would be a long-running process
        # For this example, we simulate a few cycles
        for _ in range(3):
            self.run_execution_loop()
            time.sleep(1)

    def calculate_var(self, symbol: str = None, portfolio: Dict[str, float] = None,
        confidence_level: float = 0.95, time_horizon: int = 1, method: str = "historical") -> Dict[str, float]:
        """
        Calculates Value at Risk (VaR) using historical simulation or parametric method.
        
        Args:
            symbol: Single asset symbol (for asset-level VaR)
            portfolio: Dictionary of {symbol: weight} for portfolio VaR
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
            method: "historical" or "parametric"
        
        Returns:
            Dictionary with VaR values and metadata
        """
        symbols = [symbol] if symbol else [self.symbol]
        if portfolio:
            symbols = list(portfolio.keys())
            weights = np.array(list(portfolio.values()))
        else:
            weights = np.ones(len(symbols)) / len(symbols)
    
        returns = self._calculate_historical_returns(symbols, lookback_days=252)
        if returns is None or returns.size == 0:
            return {"error": "No historical returns data available"}
    
        returns = returns[~np.isnan(returns).any(axis=1)]
        
        if method == "historical":
            portfolio_returns = returns @ weights
            sorted_returns = np.sort(portfolio_returns)
            index = int((1 - confidence_level) * len(sorted_returns))
            var = -sorted_returns[index]
        else:
            cov_matrix = np.cov(returns, rowvar=False)
            portfolio_std = np.sqrt(weights.T @ cov_matrix @ weights)
            var = portfolio_std * norm.ppf(confidence_level) * np.sqrt(time_horizon)
    
        return {
            "var_percentage": var * 100,
            "var_dollar": var * self.portfolio_value,
            "confidence_level": confidence_level,
            "time_horizon": time_horizon,
            "method": method, "assets": symbols,
            "calculation_time": datetime.now().isoformat()
        }

    def _calculate_historical_returns(self, symbols: list, lookback_days=252):
        """Fetches historical data and calculates daily returns."""
        all_returns = []
        for symbol in symbols:
            data = self.market_handler.fetch_data(symbol, lookback=lookback_days + 1)
            if data:
                df = pd.DataFrame(data).set_index('date')
                returns = df['close'].pct_change().dropna()
                all_returns.append(returns)
        if not all_returns: return None
        return pd.concat(all_returns, axis=1).dropna().values

    def train_learning_agent(self, historical_data: pd.DataFrame):
        """Trains the LearningAgent using historical market data."""
        if historical_data.empty:
            logger.error("Cannot train LearningAgent: historical_data is empty.")
            return

        logger.info(f"Starting training for LearningAgent on {self.symbol} data.")
        self.learning_agent.training_mode = True
        training_env = StockTradingEnv(self.symbol, data_df=historical_data, initial_balance=self.initial_capital)
        
        num_episodes = self.learning_agent_config.get('training_episodes', 10)
        for episode in range(num_episodes):
            obs, _ = training_env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self._generate_trading_action_from_rl(obs)
                next_obs, reward, done, _, _ = training_env.step(action)
                self.learning_agent.observe(obs, action, reward, next_obs)
                obs = next_obs
                total_reward += reward
            self.learning_agent.train_from_embeddings()
            logger.info(f"Episode {episode+1} finished. Total reward: {total_reward}")
        self.learning_agent.training_mode = False

    def run_simulation(self):
        """Runs the trading simulation in a separate thread for continuous learning/evaluation."""
        if not self.simulation_mode:
            logger.info("Simulation mode is disabled.")
            return

        def simulation_loop():
            obs = self.trading_env.reset()
            while self.simulation_mode:
                action = self._generate_action_for_backtest(obs)
                obs, reward, done, _ = self.trading_env.step(action)
                if self.training_mode:
                    self.learning_agent.observe(obs, action, reward, self.trading_env._get_observation())
                    self.learning_agent.train_from_embeddings()
                if done:
                    obs = self.trading_env.reset()
                time.sleep(1) # Simulate real-time steps

        self.env_thread = threading.Thread(target=simulation_loop, daemon=True)
        self.env_thread.start()
        logger.info("Trading simulation thread started.")

    def _generate_trading_action_from_rl(self, observation: np.ndarray) -> int:
        """
        Generates a trading action by first using the LearningAgent's meta-controller
        to select an RL strategy, and then using that strategy's agent to decide on an action.
        """
        if not hasattr(self, 'learning_agent') or self.learning_agent is None:
            logger.error("LearningAgent is not initialized. Cannot generate RL action.")
            return 0  # Default to 'Hold' action

        try:
            # 1. Use the meta-controller to select the best agent strategy for the current state.
            # The observation serves as the state embedding for the meta-controller.
            selected_strategy_name = self.learning_agent.select_agent_strategy(observation)
            
            # 2. Get the actual agent instance for the selected strategy.
            strategy_agent = self.learning_agent.agents.get(selected_strategy_name)

            if strategy_agent is None:
                logger.warning(f"Strategy '{selected_strategy_name}' selected but no corresponding agent found. Defaulting to 'Hold'.")
                return 0 # Hold action

            # 3. Use the selected agent to determine the specific action (Buy/Sell/Hold).
            if hasattr(strategy_agent, 'get_action') and callable(strategy_agent.get_action):
                action = strategy_agent.get_action(observation)
                logger.debug(f"RL action generated: Strategy='{selected_strategy_name}', Action={action}")
                return action
            else:
                 logger.warning(f"Agent for strategy '{selected_strategy_name}' does not have a 'get_action' method. Defaulting to 'Hold'.")
                 return 0

        except Exception as e:
            logger.error(f"Error generating RL trading action: {e}", exc_info=True)
            return 0 # Default to 'Hold' on any error

    def calculate_technical_indicators(self, symbol: str) -> Dict:
        """Calculates technical indicators from market data."""
        data = self.market_handler.fetch_data(symbol, lookback=60)
        if not data: return {}
        df = pd.DataFrame(data).set_index('date')
        final_indicators = {}
        rsi_series = self.backtester._calculate_rsi(df['close'])
        if not rsi_series.empty and not pd.isna(rsi_series.iloc[-1]):
            final_indicators['RSI'] = rsi_series.iloc[-1]
        macd_df = self.backtester._calculate_macd(df['close'])
        if not macd_df.empty:
            last_macd_row = macd_df.iloc[-1]
            if not pd.isna(last_macd_row['macd_line']):
                final_indicators['MACD_Line'] = last_macd_row['macd_line']
            if not pd.isna(last_macd_row['signal_line']):
                final_indicators['MACD_Signal'] = last_macd_row['signal_line']
            if not pd.isna(last_macd_row['histogram']):
                final_indicators['MACD_Hist'] = last_macd_row['histogram']
        return final_indicators

    def _update_adaptive_model(self, symbol: str):
        """Updates the adaptive learning model with the latest data."""
        data = self.market_handler.fetch_data(symbol, lookback=2)
        if len(data) == 2:
            features = torch.randn(1, self.adaptive_learner.market_features)
            true_price = data[-1]['close']
            self.adaptive_learner.update(features, true_price)
            logger.debug(f"Adaptive learner updated for {symbol}.")

    def _generate_trade(self, symbol: str, action: str, quantity: int):
        """Generates and executes a trade task via the planning and execution agents."""
        logger.info(f"Generating trade: {action.upper()} {quantity} shares of {symbol}")
        # Create a high-level task for the planner
        trade_task = Task(
            name=f"trade_{action}_{symbol}",
            task_type=TaskType.ABSTRACT,
            parameters={'symbol': symbol, 'action': action, 'quantity': quantity},
            goal_state={'position_adjusted': True} # Define a clear goal
        )
        self.planning_agent.register_task(trade_task)

        # Let the planner decompose it into executable steps
        plan = self.planning_agent.generate_plan(trade_task)
        if plan:
            logger.info(f"Plan generated for trade task. Executing...")
            # The execution agent handles the plan. This is a simplified call.
            result = self.execution_agent.perform_task({'plan': plan})
            # In a real system, you'd monitor the result.
            if result.get('status') == 'success':
                # Update internal state based on successful execution
                if action == 'buy':
                    self.open_positions[symbol] = {'direction': 'long', 'shares': quantity, 'entry_price': self.latest_price_of_default_symbol}
                    self.cash -= quantity * self.latest_price_of_default_symbol
                elif action == 'sell':
                    self._close_position(symbol, self.latest_price_of_default_symbol)
            else:
                logger.error(f"Trade execution failed for {symbol}. Reason: {result.get('reason')}")
        else:
            logger.error(f"Could not generate a plan for the trade task: {symbol} {action}")

    def _close_position(self, symbol: str, current_price: float):
        """Closes an existing position and records P&L."""
        if symbol in self.open_positions:
            position = self.open_positions.pop(symbol)
            pnl = self._calculate_pnl(position, current_price)
            self.cash += (position['entry_price'] * position['shares']) + pnl
            self.monthly_pnl += pnl
            self.closed_trades_this_month.append({'symbol': symbol, 'pnl': pnl})
            logger.info(f"Closed position for {symbol}. P&L: ${pnl:.2f}")

    def _calculate_pnl(self, position: Dict, current_price: float) -> float:
        """Calculates realized or unrealized P&L for a specific position."""
        if not all(k in position for k in ['direction', 'entry_price', 'shares']):
            return 0.0
        
        if position['direction'] == 'long':
            return (current_price - position['entry_price']) * position['shares']
        elif position['direction'] == 'short':
            return (position['entry_price'] - current_price) * position['shares']
        return 0.0
    
    def _detect_market_regime(self) -> str:
        """Determines market regime using knowledge insights"""
        regime_keywords = {
            "bullish": ["bull", "growth", "rally"],
            "bearish": ["bear", "decline", "recession"],
            "volatile": ["volatile", "uncertain", "choppy"]
        }
        
        insights = ' '.join(self.latest_insights.get("market_regime", []))
        scores = {regime: sum(insights.count(kw) for kw in kws) 
                 for regime, kws in regime_keywords.items()}
        
        if not any(scores.values()):
            return "neutral"
            
        return max(scores, key=scores.get)

    def _generate_signal(self, symbol: str) -> str:
        """Generates a trading signal for a symbol using the ALS."""
        features = torch.randn(1, self.adaptive_learner.market_features)
        prediction = self.adaptive_learner.robust_predict(features)
        current_price = self.market_handler.fetch_data(symbol, lookback=1)[0]['close']

        # Incorporate financial knowledge
        market_insights = self.latest_insights.get("market_regime", [])
        if any("bearish" in insight.lower() for insight in market_insights):
            logger.info("Knowledge: Bearish market regime detected")
            predicted_price *= 0.95  # Adjust prediction downward

        predicted_price = prediction['prediction']
        if predicted_price > current_price * (1 + self.agent_config.get('signal_threshold')):
            return 'buy'
        elif predicted_price < current_price * (1 - self.agent_config.get('signal_threshold')):
            return 'sell'
        return 'hold'

    def _calculate_buying_power(self) -> float:
        """Calculates the current buying power."""
        return self.cash

    def run_backtest(self, historical_data: pd.DataFrame):
        """Runs a backtest using the trading environment."""
        self.backtester.store_last_metrics({}) # Clear previous metrics
        metrics = self.backtester.walk_forward_test(historical_data)
        self.backtester.store_last_metrics(metrics)
        return metrics

    def _generate_action_for_backtest(self, observation: np.ndarray) -> int:
        """Generates an action for backtesting, can be simplified."""
        # This can use the same logic as live trading or a simpler model
        return self._generate_trading_action_from_rl(observation)

    def optimize_trading_parameters(self, data: pd.DataFrame):
        """Optimizes trading parameters using the backtester."""
        param_grid = {
            'position_size': [0.05, 0.1, 0.15],
            'stop_loss': [0.05, 0.1, 0.15],
            'take_profit': [0.1, 0.15, 0.2]
        }
        best_params = self.backtester.optimize_parameters(data, param_grid)
        logger.info(f"Optimized parameters: {best_params}")
        # Apply best parameters
        self.position_size = best_params.get('position_size', self.position_size)
        self.stop_loss = best_params.get('stop_loss', self.stop_loss)
        self.take_profit = best_params.get('take_profit', self.take_profit)

    def _run_periodic_evaluation(self):
        """Execute comprehensive system evaluation"""
        logger.info("Running periodic system evaluation")
        try:
            # Prepare evaluation parameters
            evaluation_params = {
                'portfolio_state': self.get_active_trading_parameters(),
                'dashboard_data': self.get_dashboard_data(),
                'risk_factors': self._determine_risk_factor()
            }
            
            # Run full validation cycle
            evaluation_report = self.evaluation_agent.execute_validation_cycle(evaluation_params)
            
            # Store and process results
            self.finance_memory.add_financial_data(
                data=evaluation_report,
                data_type='evaluation_report',
                tags=['periodic_evaluation'],
                priority='high'
            )
            
            # Handle critical findings
            if evaluation_report.get('status') == 'critical':
                self._handle_critical_evaluation(evaluation_report)
                
        except Exception as e:
            logger.error(f"Periodic evaluation failed: {e}", exc_info=True)
            self.circuit_breaker.trip()

    def get_active_trading_parameters(self) -> dict:
        """Returns current goals, P&L, risk factor, and open positions for API."""
        return {
            'financial_goals': self.financial_goals,
            'monthly_pnl': self.monthly_pnl,
            'risk_factor': self._determine_risk_factor(),
            'open_positions': self.open_positions,
            'portfolio_value': self._calculate_portfolio_value()
        }

    def get_dashboard_data(self):
        """Prepare data for dashboard display"""
        # Create a dummy tensor for prediction if no real data is available yet
        diagnostics = self.adaptive_learner.get_diagnostics()
        dummy_features = torch.randn(1, self.adaptive_learner.market_features)
        prediction = self.adaptive_learner.robust_predict(dummy_features)
        indicators = self.calculate_technical_indicators(self.symbol)
        env_metrics = self.trading_env.get_performance_metrics() if hasattr(self, 'trading_env') else {}
        return {
            'portfolio_value': self._get_total_portfolio_value(),
            'positions': self.open_positions,
            'prediction': prediction,
            'diagnostics': diagnostics,
            'indicators': indicators,
            'signals': self.trend_analyzer.get_insider_signals(self.symbol),
            'market_regime': self.market_regime,
            'environment_metrics': env_metrics
        }

    def _handle_critical_evaluation(self, report: Dict):
        """Mitigate issues found in evaluation"""
        logger.critical(f"Evaluation detected critical issues: {report.get('critical_issues', [])}")
        
        # Reduce position sizes
        original_size = self.position_size
        self.position_size = max(0.01, self.position_size * 0.5)
        logger.warning(f"Reduced position size from {original_size} to {self.position_size}")
        
        # Log safety incident
        self.evaluation_agent.safety_evaluator.log_incident({
            'type': 'risk_threshold_breach',
            'severity': 0.9,
            'context': report
        })
        
        # Trigger emergency protocols
        self.shared_memory.put("emergency_override", True, ttl=3600)
        
    def _sync_portfolio_state(self):
        logger.debug("Syncing portfolio state to shared memory.")
        portfolio_state = {
            'value': self._get_total_portfolio_value(),
            'positions': self.open_positions,
            'pnl': self.monthly_pnl
        }

        # SharedMemory for real-time access
        self.shared_memory.put(key="portfolio_state", value=portfolio_state, ttl=60)

        # FinanceMemory for historical, durable storage and analysis
        self.finance_memory.add_financial_data(
            data=self.get_active_trading_parameters(),
            data_type='portfolio_snapshot',
            tags=['daily_snapshot', f'symbol_{self.symbol}']
        )

    def _get_total_portfolio_value(self) -> float:
        """Calculates current total portfolio value (cash + unrealized P&L of open positions)."""
        unrealized_pnl = 0
        for symbol, position in self.open_positions.items():
            try:
                current_price = self.market_handler.fetch_data(symbol, lookback=1)[0]['close']
                unrealized_pnl += self._calculate_pnl(position, current_price)
            except (IndexError, TypeError):
                logger.warning(f"Could not fetch price for {symbol} to calculate unrealized P&L.")
        
        holdings_value = sum(pos['shares'] * pos['entry_price'] for pos in self.open_positions.values())
        return self.cash + holdings_value + unrealized_pnl

    def start_training(self):
        """Start RL training mode"""
        self.training_mode = True
        self.simulation_mode = True
        self._init_trading_environment()
        self.run_simulation()
        logger.info("Started RL training mode")

    def stop_training(self):
        """Stop RL training mode"""
        self.training_mode = False
        self.simulation_mode = False
        if self.env_thread and self.env_thread.is_alive():
            self.simulation_mode = False # Signal thread to stop
            self.env_thread.join(timeout=5)
        logger.info("Stopped RL training mode")

        
# Initialize and run the agent
if __name__ == "__main__":
    agent = FinanceAgent()
    agent_thread = threading.Thread(target=agent.run, daemon=True)
    agent_thread.start()
    try:
        logger.info("Finance Agent started. Main thread monitoring.")
        while True:
            time.sleep(60)
            dashboard_data = agent.get_dashboard_data()
            logger.info(f"MONITOR | Portfolio Value: ${dashboard_data.get('portfolio_value', 0):.2f}")
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Stopping Finance Agent.")