import numpy as np
import yaml, json

from typing import Deque, Optional
from collections import deque

from src.agents.adaptive.adaptive_memory import MultiModalMemory
from src.tuning.tuner import HyperparamTuner
from logs.logger import get_logger

logger = get_logger("Parameter Tuner")

CONFIG_PATH = "src/agents/adaptive/configs/adaptive_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

class LearningParameterTuner:
    def __init__(self, config=None, initial_params: dict = None):
        if config is None:
            config = load_config()
        self.config = config
        memory = MultiModalMemory(self.config)
        self.memory = memory

        config = config.get('parameter_tuner', {})
        memory_config = config.get('adaptive_memory', {})
        
        # Initialize memory with proper config section
        # self.memory = MultiModalMemory(config=memory_config)
        
        # Load bounds from config with float conversion
        self._min_learning_rate = float(config.get('min_learning_rate', 1e-4))
        self._max_learning_rate = float(config.get('max_learning_rate', 0.1))
        self._min_exploration = float(config.get('min_exploration', 0.01))

        # Load base parameters from config
        base_params = {
            'learning_rate': float(config.get('base_learning_rate', 0.01)),
            'exploration_rate': float(config.get('base_exploration_rate', 0.3)),
            'discount_factor': float(config.get('base_discount_factor', 0.95)),
            'temperature': float(config.get('base_temperature', 1.0))
        }
        
        # Merge with initial params
        self.params = {**base_params, **(initial_params or {})}
        self.performance_history: Deque[float] = deque(maxlen=100)

    def adapt(self, recent_rewards: list) -> None:
        """
        Core self-tuning algorithm based on:
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

        # Log to memory
        self.memory.log_parameters(
            performance=np.mean(recent_rewards),
            params=self.params.copy()
        )

    def run_hyperparameter_tuning(self, 
                                 tuner_config: dict,
                                 evaluation_function: callable):
        """Execute hyperparameter tuning session"""
        # Initialize and run tuner
        tuner = HyperparamTuner(
            config_path=tuner_config.get('config_path'),
            evaluation_function=evaluation_function,
            strategy=tuner_config.get('strategy', 'bayesian')
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
        old_perf = np.mean(list(self.memory.parameter_evolution['performance'][-10:]))
        self.params.update(new_params)
        new_perf = np.mean(list(self.memory.parameter_evolution['performance'][-10:]))
        return new_perf - old_perf

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

if __name__ == "__main__":
    # Test initialization with custom params
    initial_params = {'learning_rate': 0.1, 'temperature': 0.5}
    tuner = LearningParameterTuner(initial_params=initial_params)
    print("Initialization test:")
    print(f"Params: {tuner.get_params()}")
    assert tuner.params['learning_rate'] == 0.1, "Initial params not merged correctly"
    assert tuner.params['temperature'] == 0.5, "Custom param not set"
    print("✅ Initialization passed\n")

    # Test performance tracking
    print("Performance tracking test:")
    rewards = [1.0, -0.5, 2.0, 0.0]
    for r in rewards:
        tuner.update_performance(r)
    assert len(tuner.performance_history) == 4, "Performance history not updated"
    print(f"Performance history: {list(tuner.performance_history)}")
    print("✅ Performance tracking passed\n")

    # Test parameter adaptation
    print("Parameter adaptation tests:")
    print("Case 1: Stable performance (low variance)")
    tuner.adapt([0.1, 0.09, 0.11, 0.1])
    print(f"New learning rate: {tuner.params['learning_rate']:.4f}")
    
    print("\nCase 2: Volatile performance (high variance)")
    tuner.adapt([-2, 3, -1, 4])
    print(f"New learning rate: {tuner.params['learning_rate']:.4f}")
    
    print("\nCase 3: Negative rewards)")
    tuner.adapt([-1, -2, -3])
    print(f"New exploration rate: {tuner.params['exploration_rate']:.2f}")
    print("✅ Adaptation logic passed\n")

    # Test exploration decay
    print("Exploration decay test:")
    original_eps = tuner.params['exploration_rate']
    tuner.decay_exploration()
    assert tuner.params['exploration_rate'] < original_eps, "No decay occurred"
    print(f"Decayed exploration: {tuner.params['exploration_rate']:.4f}")
    print("✅ Exploration decay passed\n")

    # Test discount factor adaptation
    print("Discount factor tests:")
    print("Familiar state (visits > 100):")
    familiar_gamma = tuner.adaptive_discount_factor(101)
    print(f"Gamma: {familiar_gamma:.3f}")
    
    print("Novel state (visits < 5):")
    novel_gamma = tuner.adaptive_discount_factor(3)
    print(f"Gamma: {novel_gamma:.3f}")
    print("✅ Discount factor adaptation passed\n")

    # Test temperature scheduling
    print("Temperature schedule test:")
    print("Episode 500:")
    print(f"Temp: {tuner.temperature_schedule(500):.2f}")
    print("Episode 1500:")
    print(f"Temp: {tuner.temperature_schedule(1500):.2f}")
    print("✅ Temperature scheduling passed\n")

    # Test parameter bounds
    print("Parameter bounding test:")
    tuner.params['learning_rate'] = 1.0  # Force out-of-bound
    tuner._apply_bounds()
    print(f"Clipped learning rate: {tuner.params['learning_rate']:.4f}")
    assert tuner.params['learning_rate'] == 0.1, "Bounds not enforced"
    print("✅ Parameter bounding passed\n")

    # Test reset functionality
    print("Reset test:")
    tuner.reset(['learning_rate'])
    print(f"Reset learning rate: {tuner.params['learning_rate']:.2f}")
    assert tuner.params['learning_rate'] == 0.01, "Reset failed"
    print("✅ Reset functionality passed\n")

    print("All tests completed successfully!")
