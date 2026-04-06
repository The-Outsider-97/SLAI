from __future__ import annotations

import math
import random
import gymnasium as gym
import numpy as np
import pandas as pd

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from gymnasium import spaces

from finance.core.utils.config_loader import load_global_config, get_config_section
from finance.core.utils.financial_errors import (log_error, DataUnavailableError,
                                                 ErrorContext, PersistenceError, ValidationError)
from finance.core.finance_memory import FinanceMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Stock Trading Env")
printer = PrettyPrinter


@dataclass(slots=True)
class TradeRecord:
    step: int
    action: str
    shares: float
    price: float
    gross_value: float
    net_value: float
    fee: float
    symbol: str
    timestamp: str
    realized_pnl: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StockTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, symbol, mode='train', finance_agent=None, data_df: Optional[pd.DataFrame] = None, initial_balance=None):
        super().__init__()
        self.config = load_global_config()
        self.env_config = get_config_section('stock_trading_env')
        self.backtester_config = get_config_section('backtester')
        self.als_config = get_config_section('adaptive_learning_system')

        self.initial_capital = float(initial_balance if initial_balance is not None else self.config.get('initial_capital', 100.0))
        self.position_size_percentage = float(self.config.get('position_size', 0.1))
        self.stop_loss = float(self.config.get('stop_loss', 0.1))
        self.take_profit = float(self.config.get('take_profit', 0.15))
        self.transaction_cost = float(self.backtester_config.get('transaction_cost', 0.001))
        self.max_steps = int(self.env_config.get('max_steps', 1000))
        self.price_window = int(self.env_config.get('price_window', 10))
        self.allow_synthetic_data = bool(self.env_config.get('allow_synthetic_data', mode == 'train'))

        self.finance_memory = FinanceMemory()
        self.finance_agent = finance_agent
        self.mode = str(mode).strip().lower()
        self.symbol = self._normalize_symbol(symbol)
        self.data_df = data_df

        self.data = self._load_historical_data()
        self.indicators = self._compute_technical_indicators()

        self.indicator_columns = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Pct', 'Volume_MA', 'Volume_Pct_Change', 'Momentum_5D', 'Momentum_20D']
        self.portfolio_feature_count = 5
        self.observation_dim = self.price_window + len(self.indicator_columns) + self.portfolio_feature_count

        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(self.observation_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

        self.current_step = 0
        self.position = 0
        self.cash = float(self.initial_capital)
        self.holdings = 0.0
        self.holdings_value = 0.0
        self.shares_held = 0.0
        self.entry_price = 0.0
        self.portfolio_history: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []

        logger.info("StockTradingEnv initialized | symbol=%s mode=%s rows=%s observation_dim=%s initial_capital=%.2f", self.symbol, self.mode, len(self.data), self.observation_dim, self.initial_capital)

    def _context(self, operation: str, **metadata: Any) -> ErrorContext:
        return ErrorContext(component='stock_trading_env', operation=operation, symbol=self.symbol, metadata=metadata or {})

    def _normalize_symbol(self, symbol: Any) -> str:
        if isinstance(symbol, str) and symbol.strip():
            return symbol.strip().upper()
        if isinstance(symbol, Mapping):
            candidate = symbol.get('symbol')
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip().upper()
        if isinstance(symbol, Sequence) and symbol and isinstance(symbol[0], Mapping):
            candidate = symbol[0].get('symbol')
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip().upper()
        raise ValidationError('A valid symbol is required.', context=ErrorContext(component='stock_trading_env', operation='normalize_symbol'))

    def _safe_memory_add(self, *, data: Mapping[str, Any], data_type: str, tags: Sequence[str], priority: str = 'medium', metadata: Optional[Mapping[str, Any]] = None) -> None:
        try:
            self.finance_memory.add_financial_data(data=dict(data), data_type=data_type, tags=list(tags), priority=priority, metadata=dict(metadata or {}))
        except Exception as exc:
            handled = PersistenceError('Failed to persist trading environment artifact.', context=self._context('memory_add', data_type=data_type), cause=exc)
            log_error(handled, logger_=logger)

    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            raise DataUnavailableError('Historical data frame is empty.', context=self._context('validate_dataframe'))
        validated = df.copy()
        if 'symbol' in validated.columns:
            validated = validated[validated['symbol'].astype(str).str.upper() == self.symbol]
        else:
            validated['symbol'] = self.symbol
        if validated.empty:
            raise DataUnavailableError('No rows remain for the requested symbol.', context=self._context('validate_dataframe'))
        if 'date' in validated.columns:
            validated['date'] = pd.to_datetime(validated['date'], errors='coerce')
        elif 'datetime' in validated.columns:
            validated['date'] = pd.to_datetime(validated['datetime'], errors='coerce')
        elif 'timestamp' in validated.columns:
            validated['date'] = pd.to_datetime(validated['timestamp'], unit='s', errors='coerce')
        else:
            raise ValidationError('Historical data requires one of: date, datetime, or timestamp.', context=self._context('validate_dataframe'))
        for column in ['close', 'volume']:
            if column not in validated.columns:
                if column == 'volume':
                    validated[column] = 0.0
                else:
                    raise ValidationError(f'Historical data is missing required column: {column}', context=self._context('validate_dataframe'))
            validated[column] = pd.to_numeric(validated[column], errors='coerce')
        for column in ['open', 'high', 'low']:
            if column not in validated.columns:
                validated[column] = validated['close']
            validated[column] = pd.to_numeric(validated[column], errors='coerce').fillna(validated['close'])
        validated['volume'] = validated['volume'].fillna(0.0)
        validated = validated.dropna(subset=['date', 'close']).sort_values('date').reset_index(drop=True)
        if validated.empty:
            raise DataUnavailableError('Validated market data is empty after cleaning.', context=self._context('validate_dataframe'))
        return validated

    def _load_historical_data(self) -> pd.DataFrame:
        if self.data_df is not None and not self.data_df.empty:
            return self._validate_dataframe(self.data_df)

        market_data_records = self.finance_memory.query(data_type='market_ohlcv_daily', tags=[f'symbol_{self.symbol}', 'historical_market_data'], limit=self.max_steps + 250)
        if market_data_records:
            payloads = [dict(record.get('data', {})) for record in market_data_records if isinstance(record.get('data'), Mapping)]
            if payloads:
                return self._validate_dataframe(pd.DataFrame(payloads))

        if not self.allow_synthetic_data:
            raise DataUnavailableError(f'No historical market data available for {self.symbol}.', context=self._context('load_historical_data'))

        logger.warning("No historical data for '%s'; generating synthetic training data.", self.symbol)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=self.max_steps, freq='B')
        base_path = np.cumprod(1 + np.random.normal(0.0005, 0.02, len(dates))) * random.uniform(50, 250)
        volumes = np.random.randint(100000, 5000000, len(dates)).astype(float)
        sim_df = pd.DataFrame({
            'date': dates,
            'open': base_path * np.random.uniform(0.99, 1.01, len(dates)),
            'high': base_path * np.random.uniform(1.0, 1.03, len(dates)),
            'low': base_path * np.random.uniform(0.97, 1.0, len(dates)),
            'close': base_path,
            'volume': volumes,
            'symbol': self.symbol,
        })
        return self._validate_dataframe(sim_df)

    def _compute_technical_indicators(self) -> pd.DataFrame:
        df = self.data.copy()
        delta = df['close'].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.rolling(14, min_periods=7).mean()
        avg_loss = loss.rolling(14, min_periods=7).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        df['RSI'] = 100 - (100 / (1 + rs))
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        sma20 = df['close'].rolling(20, min_periods=10).mean()
        std20 = df['close'].rolling(20, min_periods=10).std()
        bb_upper = sma20 + (std20 * 2)
        bb_lower = sma20 - (std20 * 2)
        df['BB_Pct'] = (df['close'] - bb_lower) / ((bb_upper - bb_lower) + 1e-8)
        df['Volume_MA'] = df['volume'].rolling(10, min_periods=3).mean()
        df['Volume_Pct_Change'] = df['volume'].pct_change()
        df['Momentum_5D'] = df['close'].pct_change(5)
        df['Momentum_20D'] = df['close'].pct_change(20)
        return df

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, return_info: bool = False):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.cash = float(self.initial_capital)
        self.holdings = 0.0
        self.holdings_value = 0.0
        self.shares_held = 0.0
        self.entry_price = 0.0
        self.portfolio_history = []
        self.trade_history = []
        self._safe_memory_add(data={'event': 'env_reset', 'symbol': self.symbol, 'timestamp': pd.Timestamp.now().isoformat(), 'mode': self.mode}, data_type='system', tags=[f'symbol_{self.symbol}', f'mode_{self.mode}'], priority='medium')
        obs = self._get_observation()
        info = {'symbol': self.symbol, 'mode': self.mode}
        return (obs, info) if return_info else obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if not isinstance(action, (int, np.integer)):
            raise ValidationError('Action must be an integer.', context=self._context('step'))
        if action < 0 or action >= self.action_space.n:
            raise ValidationError('Action is out of range for the action space.', context=self._context('step', action=int(action)))

        price = float(self.data['close'].iloc[self.current_step])
        prev_value = self.cash + (self.holdings * price)
        trade_executed = False
        stop_loss_triggered = False
        take_profit_triggered = False

        if self.position != 0 and self.entry_price > 0:
            current_return = (price / self.entry_price - 1.0) * (1 if self.position > 0 else -1)
            if current_return <= -self.stop_loss:
                stop_loss_triggered = True
                action = 3 if self.position > 0 else 4
            elif current_return >= self.take_profit:
                take_profit_triggered = True
                action = 3 if self.position > 0 else 4

        if action == 0:
            pass
        elif action in (1, 2):
            trade_size = 0.10 if action == 1 else 0.25
            max_value = self.cash * trade_size
            shares_to_buy = math.floor(max_value / max(price * (1.0 + self.transaction_cost), 1e-12))
            if shares_to_buy > 0:
                gross_cost = shares_to_buy * price
                fee = gross_cost * self.transaction_cost
                total_cost = gross_cost + fee
                if self.cash >= total_cost:
                    self.holdings += float(shares_to_buy)
                    self.shares_held = self.holdings
                    self.cash -= total_cost
                    self.position = 1
                    self.entry_price = price if self.entry_price <= 0 else self.entry_price
                    trade_executed = True
                    self.trade_history.append(TradeRecord(step=self.current_step, action='buy', shares=float(shares_to_buy), price=price, gross_value=gross_cost, net_value=total_cost, fee=fee, symbol=self.symbol, timestamp=pd.Timestamp.now().isoformat()).to_dict())
        elif action in (3, 4):
            trade_size = 0.10 if action == 3 else 0.25
            if self.holdings > 0:
                shares_to_sell = min(self.holdings, max(1.0, math.floor(self.holdings * trade_size)))
                gross_proceeds = shares_to_sell * price
                fee = gross_proceeds * self.transaction_cost
                net_proceeds = gross_proceeds - fee
                realized_pnl = (price - self.entry_price) * shares_to_sell - fee if self.entry_price > 0 else 0.0
                self.holdings -= shares_to_sell
                self.shares_held = self.holdings
                self.cash += net_proceeds
                trade_executed = True
                self.trade_history.append(TradeRecord(step=self.current_step, action='sell', shares=float(shares_to_sell), price=price, gross_value=gross_proceeds, net_value=net_proceeds, fee=fee, symbol=self.symbol, timestamp=pd.Timestamp.now().isoformat(), realized_pnl=float(realized_pnl)).to_dict())
                if self.holdings <= 0:
                    self.position = 0
                    self.entry_price = 0.0

        self.current_step += 1
        done = bool(self.current_step >= min(len(self.data) - 1, self.max_steps - 1))
        current_index = min(self.current_step, len(self.data) - 1)
        valuation_price = float(self.data['close'].iloc[current_index])
        new_value = self.cash + (self.holdings * valuation_price)
        step_return = (new_value - prev_value) / max(prev_value, 1e-12)
        turnover_penalty = self.transaction_cost if trade_executed else 0.0
        drawdown_penalty = 0.0
        if self.portfolio_history:
            peak = max(point['value'] for point in self.portfolio_history)
            if peak > 0:
                drawdown_penalty = max(0.0, (peak - new_value) / peak) * 0.01
        reward = float(step_return - turnover_penalty - drawdown_penalty)

        self.portfolio_history.append({'step': self.current_step, 'value': float(new_value), 'cash': float(self.cash), 'holdings': float(self.holdings), 'price': float(valuation_price)})
        self._store_step_data(int(action), reward, trade_executed, stop_loss_triggered, take_profit_triggered)
        next_obs = self._get_observation()
        return next_obs, reward, done, {'new_value': float(new_value), 'trade_executed': trade_executed, 'stop_loss': stop_loss_triggered, 'take_profit': take_profit_triggered, 'position': int(self.position), 'cash': float(self.cash), 'holdings': float(self.holdings)}

    def _get_observation(self) -> np.ndarray:
        idx = min(self.current_step, len(self.data) - 1)
        price_series = self.data['close'].iloc[max(0, idx - self.price_window + 1):idx + 1].to_numpy(dtype=float)
        if len(price_series) < self.price_window:
            price_series = np.pad(price_series, (self.price_window - len(price_series), 0), mode='constant')
        mean_price = float(price_series.mean()) if len(price_series) else 0.0
        std_price = float(price_series.std()) + 1e-8
        normalized_prices = (price_series - mean_price) / std_price if len(price_series) else np.zeros(self.price_window)

        indicator_values: List[float] = []
        for column in self.indicator_columns:
            value = float(self.indicators[column].iloc[idx]) if column in self.indicators.columns else 0.0
            indicator_values.append(0.0 if not np.isfinite(value) else value)
        indicator_arr = np.asarray(indicator_values, dtype=float)
        indicator_mean = float(indicator_arr.mean()) if len(indicator_arr) else 0.0
        indicator_std = float(indicator_arr.std()) + 1e-8
        normalized_indicators = (indicator_arr - indicator_mean) / indicator_std if len(indicator_arr) else np.zeros(0)

        current_price = float(self.data['close'].iloc[idx])
        portfolio_state = np.array([
            float(self.position),
            float(self.cash / max(self.initial_capital, 1e-12)),
            float((self.holdings * current_price) / max(self.initial_capital, 1e-12)),
            float(self.entry_price / max(current_price, 1e-12)) if self.entry_price > 0 else 0.0,
            1.0 if self.position != 0 else 0.0,
        ], dtype=float)

        observation = np.concatenate([normalized_prices, normalized_indicators, portfolio_state]).astype(np.float32)
        if observation.shape[0] != self.observation_dim:
            padded = np.zeros(self.observation_dim, dtype=np.float32)
            padded[:min(self.observation_dim, observation.shape[0])] = observation[:self.observation_dim]
            observation = padded
        return observation

    def _store_step_data(self, action: int, reward: float, trade_executed: bool, stop_loss: bool, take_profit: bool):
        idx = min(self.current_step, len(self.data) - 1)
        step_data = {
            'step': int(self.current_step),
            'action': int(action),
            'reward': float(reward),
            'price': float(self.data['close'].iloc[idx]),
            'cash': float(self.cash),
            'holdings': float(self.holdings),
            'portfolio_value': float(self.cash + self.holdings * self.data['close'].iloc[idx]),
            'trade_executed': bool(trade_executed),
            'stop_loss_triggered': bool(stop_loss),
            'take_profit_triggered': bool(take_profit),
            'symbol': self.symbol,
            'timestamp': pd.Timestamp.now().isoformat(),
        }
        self._safe_memory_add(data=step_data, data_type='env_step', tags=[f'symbol_{self.symbol}', f'mode_{self.mode}', 'env_step'], priority='medium', metadata={'symbol': self.symbol, 'mode': self.mode})
        if trade_executed and self.trade_history:
            trade_data = {**self.trade_history[-1], 'symbol': self.symbol, 'position': self.position, 'portfolio_value': step_data['portfolio_value'], 'timestamp': pd.Timestamp.now().isoformat()}
            self._safe_memory_add(data=trade_data, data_type='trade', tags=[f'symbol_{self.symbol}', f'mode_{self.mode}', 'trade'], priority='high', metadata={'symbol': self.symbol, 'mode': self.mode})

    def _close_position(self, price: float, reason: str = 'manual_close'):
        if self.holdings <= 0:
            self.position = 0
            self.entry_price = 0.0
            return
        gross_proceeds = self.holdings * price
        fee = gross_proceeds * self.transaction_cost
        net_proceeds = gross_proceeds - fee
        realized_pnl = (price - self.entry_price) * self.holdings - fee if self.entry_price > 0 else 0.0
        record = TradeRecord(step=self.current_step, action='close', shares=float(self.holdings), price=float(price), gross_value=float(gross_proceeds), net_value=float(net_proceeds), fee=float(fee), symbol=self.symbol, timestamp=pd.Timestamp.now().isoformat(), realized_pnl=float(realized_pnl)).to_dict()
        record['reason'] = reason
        self.trade_history.append(record)
        self.cash += net_proceeds
        self.holdings = 0.0
        self.shares_held = 0.0
        self.position = 0
        self.entry_price = 0.0

    def render(self, mode='human'):
        idx = min(self.current_step, len(self.data) - 1)
        current_price = float(self.data['close'].iloc[idx])
        portfolio_value = self.cash + self.holdings * current_price
        print(f'\n=== Step {self.current_step} [{self.symbol}] ===')
        print(f'Price: {current_price:.2f} | Position: {self.position}')
        print(f'Cash: ${self.cash:.2f} | Holdings: {self.holdings:.4f}')
        print(f'Portfolio Value: ${portfolio_value:.2f}')
        if self.trade_history and self.trade_history[-1]['step'] == self.current_step:
            last_trade = self.trade_history[-1]
            print(f"Last Trade: {last_trade['action']} {last_trade['shares']:.4f} shares @ {last_trade['price']:.2f}")
        if idx > 20:
            print('\nTechnical Indicators:')
            print(f"RSI: {self.indicators['RSI'].iloc[idx]:.2f}")
            print(f"MACD: {self.indicators['MACD'].iloc[idx]:.4f}")
            print(f"MACD Hist: {self.indicators['MACD_Hist'].iloc[idx]:.4f}")
            print(f"Bollinger %: {self.indicators['BB_Pct'].iloc[idx]:.2f}")
            print(f"5D Momentum: {self.indicators['Momentum_5D'].iloc[idx]:.4f}")
        print('=' * 40)

    def get_performance_metrics(self) -> dict:
        if not self.portfolio_history:
            return {}
        values = [float(point['value']) for point in self.portfolio_history]
        initial_value = values[0]
        final_value = values[-1]
        total_return = (final_value - initial_value) / max(initial_value, 1e-12)
        returns_series = pd.Series(values).pct_change().dropna()
        volatility = float(returns_series.std() * np.sqrt(252)) if not returns_series.empty else 0.0
        peak = values[0]
        max_drawdown = 0.0
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / max(peak, 1e-12)
            max_drawdown = max(max_drawdown, dd)
        completed = [trade for trade in self.trade_history if float(trade.get('realized_pnl', 0.0)) != 0.0]
        winners = sum(1 for trade in completed if float(trade.get('realized_pnl', 0.0)) > 0.0)
        realized_pnl = sum(float(trade.get('realized_pnl', 0.0)) for trade in completed)
        return {
            'initial_value': float(initial_value),
            'final_value': float(final_value),
            'total_return': float(total_return),
            'annualized_volatility': float(volatility),
            'max_drawdown': float(max_drawdown),
            'sharpe_ratio': float(total_return / volatility) if volatility > 1e-12 else 0.0,
            'num_trades': int(len(self.trade_history)),
            'win_rate': float(winners / len(completed)) if completed else 0.0,
            'realized_pnl': float(realized_pnl),
            'symbol': self.symbol,
            'mode': self.mode,
        }


if __name__ == '__main__':  # pragma: no cover
    print("\n===== Testing Stock Trading Environment =====")
    printer.status("INIT", "Testing Stock Trading Environment\n", "info")
    env = StockTradingEnv(symbol='NVDA', mode='train')
    print(f"{env}")

    obs = env.reset()
    print('Observation shape:', obs.shape)
    print("\n===== Succesfully tested Stock Trading Environment =====\n")
