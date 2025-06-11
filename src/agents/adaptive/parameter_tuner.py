import numpy as np

from typing import Deque
from collections import deque

from src.agents.adaptive.utils.config_loader import load_global_config, get_config_section
from src.agents.adaptive.adaptive_memory import MultiModalMemory
from src.tuning.tuner import HyperparamTuner
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Parameter Tuner")
printer = PrettyPrinter

class LearningParameterTuner:
    def __init__(self):
        self.config = load_global_config()
        self.tuner_config = get_config_section('parameter_tuner')
        
        # Load bounds from config with float conversion
        self._min_learning_rate = float(self.tuner_config.get('min_learning_rate'))
        self._max_learning_rate = float(self.tuner_config.get('max_learning_rate'))
        self._min_exploration = float(self.tuner_config.get('min_exploration'))

        # Load base parameters from config
        base_params = {
            'learning_rate': float(self.tuner_config.get('base_learning_rate')),
            'exploration_rate': float(self.tuner_config.get('base_exploration_rate')),
            'discount_factor': float(self.tuner_config.get('base_discount_factor')),
            'temperature': float(self.tuner_config.get('base_temperature'))
        }

        # Create memory instance
        self.memory = MultiModalMemory()

        # Merge with initial params
        self.initial_params = {}
        self.params = {**base_params, **(self.initial_params or {})}
        self.performance_history: Deque[float] = deque(maxlen=100)

        logger.info(f"Learning Parameter Tuner succesfully initialized with:\n {base_params}")

    def adapt(self, recent_rewards: list) -> None:
        """
        Core self-tuning algorithm based on:
        - Sutton's Automatic Step Size Adaptation
        - Darken & Moody's Search-then-converge schedule
        """
        printer.status("INIT", "Adapter succesfully initialized")

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

        # Log to memory
        self.memory.log_parameters(
            performance=np.mean(recent_rewards),
            params=self.params.copy()
        )

    def _apply_bounds(self) -> None:
        """Ensure parameters stay within safe ranges"""
        printer.status("INIT", "Bounds succesfully initialized")

        self.params['learning_rate'] = np.clip(
            self.params['learning_rate'],
            self._min_learning_rate,
            self._max_learning_rate
        )

    def run_hyperparameter_tuning(self, evaluation_function: callable):
        """Execute hyperparameter tuning session"""
        printer.status("INIT", "Tuning succesfully initialized")

        # Initialize and run tuner
        tuner = HyperparamTuner(
            config_path=self.tuner_config.get('parameter_tuner'),
            evaluation_function=evaluation_function,
            strategy=self.tuner_config.get('strategy', 'bayesian')
        )
        best_params = tuner.run_tuning_pipeline()

        # Log intervention in memory
        intervention = {
            'type': 'hyperparameter_tuning',
            'params_before': self.params.copy(),
            'params_after': best_params
        }
        effect = {
            'performance_delta': self._calculate_performance_change(best_params)
        }
        self.memory.apply_policy_intervention(intervention, effect)

        return best_params

    def _calculate_performance_change(self, new_params: dict) -> float:
        """Calculate performance impact of new parameters"""
        printer.status("INIT", "Change calculation succesfully initialized")

        old_perf = np.mean(list(self.memory.parameter_evolution['performance'][-10:]))
        self.params.update(new_params)
        new_perf = np.mean(list(self.memory.parameter_evolution['performance'][-10:]))
        return new_perf - old_perf

    def decay_exploration(self, decay_factor: float = 0.9995) -> None:
        """Exponential decay with performance-based modulation"""
        printer.status("INIT", "Decay explorer succesfully initialized")

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
        """
        Update internal performance tracking metrics.
    
        This method maintains a rolling performance history buffer,
        which is used to:
        - Influence exploration decay rate
        - Inform volatility calculations in `adapt()`
        - Track long-term reward trends
        - Provide logs to memory for longitudinal analysis
    
        Args:
            reward (float): Reward signal from environment or model performance.
        """
        if not isinstance(reward, (float, int)) or np.isnan(reward):
            logger.warning(f"Ignored invalid reward input: {reward}")
            return
    
        self.performance_history.append(float(reward))
        
        # Optionally log to memory for advanced diagnostics
        self.memory.log_parameters(
            performance=reward,
            params=self.params.copy()
        )
    
        printer.status("UPDATE", f"Recorded performance: {reward:.3f}", "info")

    def get_params(self, include_metadata: bool = False) -> dict:
        """
        Retrieve the current set of learning parameters.
    
        Args:
            include_metadata (bool): If True, includes extra diagnostics such as:
                - Bounds
                - Performance stats
                - History length
    
        Returns:
            dict: A snapshot of current parameter values, with optional metadata.
        """
        params_snapshot = self.params.copy()
    
        if include_metadata:
            params_snapshot["_metadata"] = {
                "min_learning_rate": self._min_learning_rate,
                "max_learning_rate": self._max_learning_rate,
                "min_exploration_rate": self._min_exploration,
                "performance_history_length": len(self.performance_history),
                "last_performance": (
                    self.performance_history[-1]
                    if self.performance_history else None
                ),
                "average_performance": (
                    float(np.mean(self.performance_history))
                    if self.performance_history else None
                )
            }
    
        return params_snapshot
        
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

if __name__ == "__main__":
    print("\n=== Running Learning Parameter Tuner ===\n")
    printer.status("TEST", "Starting Learning Parameter Tuner tests", "info")

    tuner = LearningParameterTuner()
    print(tuner)

    print("\n* * * * * Phase 2 * * * * *\n")
    rewards=[11, 46, 23]
    reward=50
    eval=lambda params: np.random.uniform(0.0, 1.0)
    factor=0.995
    metadata=False
    tuning = tuner.run_hyperparameter_tuning(evaluation_function=eval)

    printer.status("recent", tuner.adapt(recent_rewards=rewards), "success")
    printer.pretty("tuning", tuning, "success")
    printer.pretty("decay", tuner.decay_exploration(decay_factor=factor), "success")
    printer.pretty("update", tuner.update_performance(reward=reward), "success")
    printer.pretty("params", tuner.get_params(include_metadata=metadata), "success")

    print("All tests completed successfully!")
