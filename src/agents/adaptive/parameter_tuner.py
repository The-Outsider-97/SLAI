import numpy as np
from typing import Deque
from collections import deque
from logs.logger import get_logger

logger = get_logger(__name__)

class LearningParameterTuner:
    def __init__(self, initial_params: dict):
        self.params = {
            'learning_rate': 0.01,
            'exploration_rate': 0.3,
            'discount_factor': 0.95,
            'temperature': 1.0,
            **initial_params
        }
        self.performance_history: Deque[float] = deque(maxlen=100)
        self._min_learning_rate = 1e-4
        self._max_learning_rate = 0.1
        self._min_exploration = 0.01
        
    def adapt(self, recent_rewards: list) -> None:
        """Core self-tuning algorithm based on:
        - Sutton's Automatic Step Size Adaptation
        - Darken & Moody's Search-then-converge schedule
        """
        if not recent_rewards:
            return
            
        # Calculate performance volatility
        reward_var = np.var(recent_rewards)
        mean_reward = np.mean(recent_rewards)
        
        # Dynamic learning rate adjustment
        if reward_var < 0.1:  # Stable performance
            self.params['learning_rate'] *= 0.995
            logger.debug(f"Reducing learning rate to {self.params['learning_rate']:.4f}")
        elif reward_var > 1.0:  # High volatility
            self.params['learning_rate'] *= 1.01
            logger.debug(f"Increasing learning rate to {self.params['learning_rate']:.4f}")
            
        # Exploration rate adaptation
        if mean_reward < 0:
            self.params['exploration_rate'] = min(
                self.params['exploration_rate'] * 1.1,
                0.5
            )
            logger.warning(f"Negative rewards detected. Boosting exploration to {self.params['exploration_rate']:.2f}")
            
        self._apply_bounds()

    def decay_exploration(self, decay_factor: float = 0.9995) -> None:
        """Exponential decay with performance-based modulation"""
        base_decay = decay_factor
        
        # If recent performance is good, decay faster
        if len(self.performance_history) > 10:
            recent_avg = np.mean(list(self.performance_history)[-10:])
            if recent_avg > 0.7 * np.max(self.performance_history):
                base_decay **= 2  # Accelerate decay
                
        self.params['exploration_rate'] = max(
            self.params['exploration_rate'] * base_decay,
            self._min_exploration
        )
        logger.debug(f"Exploration rate decayed to {self.params['exploration_rate']:.4f}")

    def update_performance(self, reward: float) -> None:
        """Update performance tracking for adaptation"""
        self.performance_history.append(reward)

    def get_params(self) -> dict:
        """Get current parameter snapshot"""
        return self.params.copy()

    def _apply_bounds(self) -> None:
        """Ensure parameters stay within safe ranges"""
        self.params['learning_rate'] = np.clip(
            self.params['learning_rate'],
            self._min_learning_rate,
            self._max_learning_rate
        )
        
    def adaptive_discount_factor(self, state_visits: int) -> float:
        """Context-aware discount factor adjustment"""
        base_gamma = self.params['discount_factor']
        # Increase planning horizon for familiar states
        if state_visits > 100:
            return min(base_gamma * 1.01, 0.999)
        # Reduce horizon for novel states
        elif state_visits < 5:  
            return max(base_gamma * 0.99, 0.8)
        return base_gamma

    def temperature_schedule(self, episode: int) -> float:
        """Curriculum learning temperature for policy sharpening"""
        initial_temp = 1.0
        final_temp = 0.1
        decay_episodes = 1000
        return max(
            final_temp, 
            initial_temp * (1 - episode/decay_episodes)
        )

    def reset(self, params_to_reset: list = None) -> None:
        """Partial reset capability for recovery mechanisms"""
        if params_to_reset is None:
            params_to_reset = ['learning_rate', 'exploration_rate']
            
        for param in params_to_reset:
            self.params[param] = self._get_default(param)
            
    def _get_default(self, param_name: str) -> float:
        """Get original parameter default"""
        defaults = {
            'learning_rate': 0.01,
            'exploration_rate': 0.3,
            'discount_factor': 0.95,
            'temperature': 1.0
        }
        return defaults.get(param_name, 0.0)
