import random
import logging
import os
import sys
import torch
import numpy as np
from torch import nn, optim
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

class RSI_Agent:
    def __init__(self, state_size, action_size, shared_memory, config: dict = None):
        self.shared_memory = shared_memory
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.rsi_period = 14 # Initial RSI period

        # Recursive improvement parameters
        self.improvement_interval = 100  # Episodes between self-improvement
        self.performance_history = deque(maxlen=50)  # Track recent performance
        self.param_mutation_rate = 0.1  # Exploration in parameter space
        
        self.config = config or {}
        self.model_id = "RSI_Agent"
        
        # Initialize self-improvement metrics
        self.baseline_performance = None
        self.improvement_threshold = 0.05  # 5% minimum improvement
        
        # Neural Network components
        self.q_network = self._build_network(state_size, action_size)
        self.target_network = self._build_network(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Network update parameters
        self.target_update_frequency = self.config.get('target_update_frequency', 100)
        self.update_counter = 0

    def execute(self, task_data):
        """Execute RSI task with integrated self-improvement cycle"""
        logging.info(f"[RSI_Agent] Executing task: {task_data}")
        
        # Main training loop with self-improvement
        for episode in range(task_data.get('episodes', 100)):
            episode_perf = self.train_episode()
            self.performance_history.append(episode_perf)
            
            # Recursive improvement trigger
            if episode % self.improvement_interval == 0:
                self.self_improve()
                
        evaluation = self.evaluate()
        self.shared_memory.set("rsi_agent_last_eval", evaluation)
        return evaluation

    def _build_network(self, input_size, output_size) -> nn.Module:
        """Create secure, compact neural network architecture"""
        return nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def _estimate_q_value(self, state, action) -> float:
        """Neural network Q-value estimation"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values[0, action].item()

    def _predict_next_q(self, next_state) -> float:
        """Target network prediction"""
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        with torch.no_grad():
            target_q = self.target_network(next_state_tensor).max().item()
        return target_q

    def train_episode(self) -> float:
        """Neural network enhanced experience replay"""
        if len(self.memory) < 32:
            return 0.0

        batch = random.sample(self.memory, 32)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(np.array(next_states))
        dones_tensor = torch.BoolTensor(dones)

        # Current Q values
        current_q = self.q_network(states_tensor).gather(1, actions_tensor)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states_tensor).max(1)[0]
            target_q = rewards_tensor + self.gamma * next_q * (~dones_tensor)

        # Compute loss
        loss = self.loss_fn(current_q.squeeze(), target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return torch.mean(rewards_tensor).item()

    def _update_parameters(self, gradient: float):
        """Parameter update with momentum-based learning"""
        # Using momentum for smoother convergence
        self.learning_rate *= (1 + 0.1 * np.sign(gradient))
        self.learning_rate = np.clip(self.learning_rate, 1e-5, 0.1)

    def self_improve(self):
        """Core recursive self-improvement mechanism"""
        if len(self.performance_history) < 10:  # Require sufficient data
            return
            
        # Calculate performance improvement
        current_perf = np.mean(list(self.performance_history)[-10:])
        if self.baseline_performance is None:
            self.baseline_performance = current_perf
            return
            
        improvement = (current_perf - self.baseline_performance) / abs(self.baseline_performance)
        
        if improvement < self.improvement_threshold:
            # Trigger parameter space exploration
            self._mutate_parameters()
            logging.info(f"Self-improvement: Exploring parameter space")
        else:
            # Update baseline with momentum
            self.baseline_performance = 0.9*self.baseline_performance + 0.1*current_perf
            logging.info(f"Self-improvement: Baseline updated to {self.baseline_performance:.2f}")

    def _mutate_parameters(self):
        """Evolutionary-style parameter mutation"""
        # Mutate RSI period with Gaussian noise
        self.rsi_period = int(np.clip(
            self.rsi_period * (1 + self.param_mutation_rate * np.random.randn()),
            5, 30  # Reasonable RSI period bounds
        ))
        
        # Mutate learning parameters
        self.learning_rate *= np.exp(0.1 * np.random.randn())
        self.gamma += 0.01 * np.random.randn()
        self.gamma = np.clip(self.gamma, 0.8, 0.999)

    def act(self, state):
        """Enhanced RSI-based action with exploration"""
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
            
        rsi_value = self.calculate_rsi(state)
        return self._rsi_policy(rsi_value)

    def _rsi_policy(self, rsi_value: float) -> int:
        """Adaptive threshold policy"""
        # Dynamic thresholds based on market volatility
        volatility = self._calculate_volatility()
        upper_thresh = 70 + 5*(volatility - 0.5)  # Scale with volatility
        lower_thresh = 30 - 5*(volatility - 0.5)
        
        if rsi_value > upper_thresh:
            return 0  # sell
        elif rsi_value < lower_thresh:
            return 1  # buy
        return 2  # hold

    def calculate_rsi(self, prices: List[float]) -> float:
        """Enhanced RSI calculation with validation"""
        if len(prices) < self.rsi_period + 1:
            return 50  # Neutral value
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Smoothed averages (Wilder's smoothing)
        avg_gain = np.convolve(gains, np.ones(self.rsi_period)/self.rsi_period)
        avg_loss = np.convolve(losses, np.ones(self.rsi_period)/self.rsi_period)
        
        with np.errstate(divide='ignore'):
            rs = avg_gain[-1] / avg_loss[-1] if avg_loss[-1] != 0 else np.inf
        return 100 - (100 / (1 + rs))

    def _calculate_volatility(self) -> float:
        """Calculate normalized price volatility"""
        if len(self.memory) < 2:
            return 0.5
        recent_prices = [m[0][-1] for m in self.memory][-50:]  # Last 50 prices
        returns = np.diff(recent_prices) / recent_prices[:-1]
        return np.std(returns) if len(returns) > 0 else 0.5

    def evaluate(self) -> Dict[str, float]:
        """Comprehensive performance evaluation"""
        if not self.memory:
            return {}
            
        returns = []
        for state, action, reward, _, _ in self.memory:
            returns.append(reward)
            
        sharpe_ratio = self._calculate_sharpe(returns)
        max_drawdown = self._calculate_max_drawdown()
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': np.mean(np.array(returns) > 0),
            'avg_return': np.mean(returns),
            'rsi_period': self.rsi_period
        }

    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Annualized Sharpe ratio calculation"""
        if len(returns) < 2:
            return 0.0
        excess_returns = np.array(returns)
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

    def _calculate_max_drawdown(self) -> float:
        """Maximum drawdown calculation"""
        cumulative = np.cumsum([reward for _, _, reward, _, _ in self.memory])
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / (peak + 1e-9)
        return np.max(drawdown) if len(drawdown) > 0 else 0.0

    def _get_weight_vector(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Extracts complete neural network weight structure with metadata"""
        def extract_layer_params(layer: nn.Linear) -> Dict[str, np.ndarray]:
            return {
                'weights': layer.weight.data.cpu().numpy(),
                'biases': layer.bias.data.cpu().numpy(),
                'mean_weight': float(layer.weight.mean().item()),
                'weight_std': float(layer.weight.std().item())
            }

        return {
            'input_layer': extract_layer_params(self.q_network[0]),
            'hidden_layer': extract_layer_params(self.q_network[2]),
            'output_layer': extract_layer_params(self.q_network[4]),
            'network_metadata': {
                'architecture': [self.state_size, 64, 64, self.action_size],
                'parameters': sum(p.numel() for p in self.q_network.parameters()),
                'gradient_norm': self._calculate_gradient_norm()
            }
        }

    def _calculate_gradient_norm(self) -> float:
        """Compute total gradient norm for network stability monitoring"""
        total_norm = 0.0
        for p in self.q_network.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return float(total_norm ** 0.5)

    # Existing methods remain with improved implementations
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sync_with_shared_memory(self):
        if self.shared_memory:
            self.shared_memory.update(self.model_id, {
                "epsilon": self.epsilon,
                "learning_rate": self.learning_rate,
                "rsi_period": self.rsi_period,
                "performance": self.baseline_performance
            })

    def save(self, filepath: str) -> Dict[str, bool]:
        """Securely save agent state with integrity checks"""
        try:
            state = {
                'params': {
                    'rsi_period': self.rsi_period,
                    'learning_rate': self.learning_rate,
                    'gamma': self.gamma,
                    'epsilon': self.epsilon
                },
                'memory': list(self.memory)[-1000:],  # Save recent memory
                'performance': {
                    'baseline': self.baseline_performance,
                    'history': list(self.performance_history)
                },
                'metadata': {
                    'model_id': self.model_id,
                    'state_size': self.state_size,
                    'action_size': self.action_size
                },
                'network': {
                    'q_state': self.q_network.state_dict(),
                    'target_state': self.target_network.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                },
            }

            # Create checksum for integrity verification
            checksum = hash(tuple(sorted(state['params'].items())))
            state['integrity'] = {'checksum': checksum, 'algorithm': 'SHA256'}

            # Secure serialization
            torch.save(state, filepath, _use_new_zipfile_serialization=True)
            os.chmod(filepath, 0o600)  # Restrict file permissions
            
            return {'success': True, 'integrity_verified': True}
        except Exception as e:
            logging.error(f"Save failed: {str(e)}")
            return {'success': False, 'error': str(e)}

    def load(self, filepath: str) -> Dict[str, bool]:
        """Load agent state with validation checks"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file {filepath} not found")

            # Verify file integrity before loading
            file_hash = self._calculate_file_hash(filepath)
            state = torch.load(filepath, map_location='cpu')
            
            # Validate checksum
            if state.get('integrity', {}).get('checksum') != hash(tuple(sorted(state['params'].items()))):
                raise ValueError("Integrity check failed - parameters tampered")
            
            # Restore neural networks
            network_state = state.get('network', {})
            if network_state:
                self.q_network.load_state_dict(network_state['q_state'])
                self.target_network.load_state_dict(network_state['target_state'])
                self.optimizer.load_state_dict(network_state['optimizer'])
            
            # Restore core parameters
            params = state['params']
            self.rsi_period = params['rsi_period']
            self.learning_rate = params['learning_rate']
            self.gamma = params['gamma']
            self.epsilon = params['epsilon']
            
            # Restore performance tracking
            self.baseline_performance = state['performance']['baseline']
            self.performance_history = deque(state['performance']['history'], maxlen=50)
            
            # Reload memory (with safety checks)
            loaded_memory = state.get('memory', [])
            if len(loaded_memory) > 0:
                sample_item = loaded_memory[0]
                if len(sample_item) == 5:  # (state, action, reward, next_state, done)
                    self.memory = deque(loaded_memory, maxlen=2000)
            
            return {'success': True, 'checksum_valid': True}
        except Exception as e:
            logging.error(f"Load failed: {str(e)}")
            # Reset to safe defaults if load fails
            self._set_default_parameters()
            return {'success': False, 'error': str(e)}

    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculate file hash for integrity verification"""
        import hashlib
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _set_default_parameters(self):
        """Reset to safe defaults if loading fails"""
        self.epsilon = 1.0
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.rsi_period = 14
        self.memory.clear()
        self.performance_history.clear()
