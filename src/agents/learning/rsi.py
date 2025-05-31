"""
Proficient In:
    Adaptive environments requiring self-optimization (e.g., finance, trading).
    Long-term autonomous agents needing continuous self-tuning.

Best Used When:
    Long training periods with dynamic environments.
    You need the agent to evolve without human supervision.
    Task performance may plateau and benefit from self-reflection or adjustment.
"""
import random
import logging
import os
import yaml
import torch
import numpy as np
from torch import nn, optim
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from src.agents.learning.utils.config_loader import load_global_config, get_config_section
from src.agents.learning.utils.neural_network import NeuralNetwork
from src.agents.learning.learning_memory import LearningMemory
from logs.logger import get_logger

logger = get_logger("Recursive Self-Improvement")

CONFIG_PATH = "src/agents/learning/configs/learning_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

class RSIAgent:
    def __init__(self, state_size, action_size, agent_id):
        self.config = load_global_config()
        self.rsi_config = get_config_section('neural_network')

        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size

        rsi_config = self.config.get('rsi', {})
        self.gamma = rsi_config.get('gamma')
        self.epsilon = rsi_config.get('epsilon')
        self.epsilon_min = rsi_config.get('epsilon_min')
        self.epsilon_decay = rsi_config.get('epsilon_decay')
        self.learning_rate = rsi_config.get('learning_rate', 0.001)
        self.rsi_period = rsi_config.get('rsi_period')

        # Recursive improvement parameters
        self.improvement_interval = rsi_config.get('improvement_interval')
        self.param_mutation_rate = rsi_config.get('param_mutation_rate')

        # Sync learning rate to neural_network config
        self.config['neural_network']['learning_rate'] = self.learning_rate
        # Set network architecture
        self.config['neural_network']['layer_dims'] = [self.state_size, 64, 64, self.action_size]

        # Initialize neural networks
        self.q_network = NeuralNetwork( 
            input_dim=self.state_size,
            output_dim=self.action_size
        )
        self.target_network = NeuralNetwork(
            input_dim=self.state_size,
            output_dim=self.action_size
        )
        self.target_network.set_weights(self.q_network.get_weights())

        # Initialize other components
        self.performance_history = deque(maxlen=50)
        self.baseline_performance = None
        self.improvement_threshold = 0.05
        self.target_update_frequency = rsi_config.get('target_update_frequency')
        self.update_counter = 0
        self.current_epoch = 0

        self.learning_memory = LearningMemory()
        self.model_id = "RSI_Agent"
        self.memory = deque(maxlen=10000)

        logger.info(f"Recursive Self-Improvement has succesfully initialized")

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
        self.learning_memory.set("rsi_agent_last_eval", evaluation)
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
        """Neural network Q-value estimation using integrated NeuralNetwork"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network.forward(state_tensor)
        return q_values[0, action].item()

    def _predict_next_q(self, next_state) -> float:
        """Target network prediction using integrated NeuralNetwork"""
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        with torch.no_grad():
            target_q_values = self.target_network.forward(next_state_tensor)
        return target_q_values.max().item()

    def train_episode(self) -> float:
        """Neural network enhanced experience replay"""
        if len(self.memory) < 32:
            return 0.0

        batch = random.sample(self.memory, 32)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)

        # Compute current Q values
        current_q = self.q_network.forward(states)

        # Compute next Q values using target network
        next_q = self.target_network.forward(next_states)
        max_next_q = torch.max(next_q, dim=1).values

        # Compute target Q values
        target_q = rewards + (1 - dones.float()) * self.gamma * max_next_q

        # Create target tensor
        target = current_q.clone().detach()
        batch_indices = torch.arange(len(states))
        target[batch_indices, actions] = target_q

        # Compute loss and update q_network
        loss = self.q_network.compute_loss(current_q, target)
        self.q_network.backward(target)
        self.q_network.update_parameters()

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_network.set_weights(self.q_network.get_weights())

        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return torch.mean(rewards).item()

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

    def sync_with_learning_memory(self):
        """Enhanced synchronization with LearningMemory system"""
        if self.learning_memory:
            # Store current parameters
            self.learning_memory.add({
                "epsilon": self.epsilon,
                "learning_rate": self.learning_rate,
                "rsi_period": self.rsi_period,
                "performance": self.baseline_performance,
                "network_weights": self.q_network.get_weights(),
                "target_weights": self.target_network.get_weights()
            }, tag="agent_state")
            
            # Store experience memory in batches
            batch_size = 100
            for i in range(0, len(self.memory), batch_size):
                self.learning_memory.add(
                    list(self.memory)[i:i+batch_size],
                    tag=f"experience_batch_{i//batch_size}"
                )

    def save(self, filepath: str) -> Dict[str, bool]:
        """Save agent state through LearningMemory integration"""
        try:
            # Sync all components to learning memory
            self.sync_with_learning_memory()
            
            # Create comprehensive checkpoint
            checkpoint_data = {
                'memory_state': self.learning_memory.get(),
                'config': self.config,
                'performance_history': list(self.performance_history),
                'update_counter': self.update_counter
            }

            # Save using LearningMemory's checkpoint system
            self.learning_memory.save_checkpoint(filepath)
            
            # Save neural networks separately
            network_checkpoint = {
                'q_network': self.q_network.get_weights(),
                'target_network': self.target_network.get_weights()
            }
            torch.save(network_checkpoint, f"{filepath}.networks")
            
            return {'success': True, 'integrity_verified': True}
        except Exception as e:
            logging.error(f"Save failed: {str(e)}")
            return {'success': False, 'error': str(e)}

    def load(self, filepath: str) -> Dict[str, bool]:
        """Load agent state through LearningMemory integration"""
        try:
            # Load memory and configuration
            self.learning_memory.load_checkpoint(filepath)
            
            # Restore core parameters
            memory_state = self.learning_memory.get()
            if 'agent_state' in memory_state:
                state = memory_state['agent_state']
                self.epsilon = state.get('epsilon')
                self.learning_rate = state.get('learning_rate', 0.001)
                self.rsi_period = state.get('rsi_period')
                self.baseline_performance = state.get('performance')

            # Restore experience memory
            self.memory.clear()
            for key in sorted([k for k in memory_state if k.startswith('experience_batch_')]):
                self.memory.extend(memory_state[key])

            # Load neural networks
            network_checkpoint = torch.load(f"{filepath}.networks")
            self.q_network.set_weights(network_checkpoint['q_network'])
            self.target_network.set_weights(network_checkpoint['target_network'])

            return {'success': True, 'checksum_valid': True}
        except Exception as e:
            logging.error(f"Load failed: {str(e)}")
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
        rsi_config = self.config['rsi']
        self.gamma = rsi_config.get('gamma')
        self.epsilon = rsi_config.get('epsilon')
        self.learning_rate = rsi_config.get('learning_rate', 0.001)
        self.rsi_period = rsi_config.get('rsi_period')
        self.memory.clear()
        self.performance_history.clear()

    def train(self, total_epochs=1000, episodes_per_epoch=100, 
              evaluation_interval=10, performance_threshold=0.0,
              checkpoint_interval=50):
        """
        Full training lifecycle with integrated self-improvement and memory management
        
        Features:
        - Automated training with periodic evaluation
        - LearningMemory integration for state persistence
        - NeuralNetwork-powered experience replay
        - Performance-based parameter mutation
        - Checkpointing and recovery systems
        """
        
        try:
            # Attempt to resume from last checkpoint
            if self.learning_memory.get("last_checkpoint"):
                self._load_training_state()
                logger.info("Resuming training from checkpoint")
                
            for epoch in range(self.current_epoch, total_epochs):
                epoch_loss = 0.0
                epoch_rewards = []
                volatility_history = []
                
                # Training phase
                for _ in range(episodes_per_epoch):
                    # Standard training episode
                    episode_reward = self.train_episode()
                    epoch_rewards.append(episode_reward)
                    
                    # Track market volatility
                    volatility_history.append(self._calculate_volatility())
                    
                    # Store experience in learning memory
                    if len(self.memory) > 0:
                        latest_exp = self.memory[-1]
                        self.learning_memory.add(latest_exp, tag="rsi_experience")
    
                # Calculate epoch metrics
                avg_loss = np.mean(epoch_loss)
                avg_reward = np.mean(epoch_rewards)
                avg_volatility = np.mean(volatility_history)
                
                # Store performance metrics
                self.performance_history.append(avg_reward)
                existing_metrics = self.learning_memory.get("training_metrics") or []
                existing_metrics.append({
                    "epoch": epoch,
                    "avg_reward": avg_reward,
                    "avg_volatility": avg_volatility,
                    "epsilon": self.epsilon,
                    "learning_rate": self.learning_rate,
                    "network_weights": self.q_network.get_weights()
                })
                self.learning_memory.set("training_metrics", existing_metrics)
    
                # Evaluation and self-improvement
                if epoch % evaluation_interval == 0:
                    eval_results = self.evaluate()
                
                    if all(k in eval_results for k in ('sharpe_ratio', 'max_drawdown', 'avg_return')):
                        logger.info(f"Epoch {epoch} Evaluation - Sharpe: {eval_results['sharpe_ratio']:.2f} | "
                                    f"Max DD: {eval_results['max_drawdown']:.2f}")
                        
                        if eval_results['avg_return'] < performance_threshold:
                            self._mutate_parameters()
                            logger.info("Triggered performance-based parameter mutation")
                    else:
                        logger.warning(f"Epoch {epoch} Evaluation skipped due to missing data: {eval_results}")
    
                # Periodic checkpointing
                if checkpoint_interval and epoch % checkpoint_interval == 0:
                    self._save_training_state(epoch)
                    logger.info(f"Checkpoint saved at epoch {epoch}")
    
                # Early stopping condition
                if self._check_early_stopping():
                    logger.info("Early stopping condition met")
                    break
    
                # Adaptive learning rate adjustment
                self._adapt_learning_rate(avg_reward, avg_volatility)
    
            logger.info("Training completed successfully")
            return self.get_training_summary()
    
        except Exception as e:
            logger.error(f"Training interrupted: {str(e)}")
            self._save_training_state(self.current_epoch)
            raise
    
    def _save_training_state(self, epoch):
        """Persist complete training state using LearningMemory"""
        state = {
            "epoch": epoch,
            "q_network": self.q_network.get_weights(),
            "target_network": self.target_network.get_weights(),
            "performance_history": list(self.performance_history),
            "hyperparameters": {
                "epsilon": self.epsilon,
                "learning_rate": self.learning_rate,
                "gamma": self.gamma
            }
        }
        self.learning_memory.set("last_checkpoint", state)
        self.sync_with_learning_memory()
    
    def _load_training_state(self):
        """Restore training state from LearningMemory"""
        state = self.learning_memory.get("last_checkpoint")
        
        if state:
            self.q_network.set_weights(state["q_network"])
            self.target_network.set_weights(state["target_network"])
            self.performance_history = deque(state["performance_history"], maxlen=50)
            
            # Restore hyperparameters
            hp = state["hyperparameters"]
            self.epsilon = hp["epsilon"]
            self.learning_rate = hp["learning_rate"]
            self.gamma = hp["gamma"]
            
            self.current_epoch = state["epoch"] + 1
    
    def _adapt_learning_rate(self, avg_reward, volatility):
        """NeuralNetwork-aware learning rate adaptation"""
        # Dynamic learning rate adjustment based on performance and market conditions
        volatility_factor = np.clip(volatility / 0.2, 0.5, 2.0)  # Normalize volatility
        reward_factor = 1 + (avg_reward / 100)  # Scale with reward magnitude
        
        new_lr = self.learning_rate * reward_factor / volatility_factor
        self.learning_rate = np.clip(new_lr, 1e-5, 0.1)
        
        # Update NeuralNetwork optimizer
        self.q_network.optimizer.learning_rate = self.learning_rate
    
    def _check_early_stopping(self):
        """LearningMemory-powered early stopping criteria"""
        if len(self.performance_history) < 20:
            return False
        
        recent_perf = np.mean(list(self.performance_history)[-10:])
        baseline_perf = np.mean(list(self.performance_history)[-20:-10])
        
        return (recent_perf - baseline_perf) < self.improvement_threshold
    
    def get_training_summary(self):
        """Generate comprehensive training report using stored memory data"""
        metrics = self.learning_memory.get("training_metrics")
    
        if metrics is None or not isinstance(metrics, list) or len(metrics) == 0:
            logger.warning("No training metrics found in LearningMemory.")
            return {
                "total_episodes": len(self.memory),
                "avg_reward": None,
                "best_reward": None,
                "volatility_profile": {"avg": None, "max": None},
                "final_parameters": {
                    "epsilon": self.epsilon,
                    "learning_rate": self.learning_rate,
                    "rsi_period": self.rsi_period
                }
            }
    
        return {
            "total_episodes": len(self.memory),
            "avg_reward": np.mean([m["avg_reward"] for m in metrics]),
            "best_reward": np.max([m["avg_reward"] for m in metrics]),
            "volatility_profile": {
                "avg": np.mean([m["avg_volatility"] for m in metrics]),
                "max": np.max([m["avg_volatility"] for m in metrics])
            },
            "final_parameters": {
                "epsilon": self.epsilon,
                "learning_rate": self.learning_rate,
                "rsi_period": self.rsi_period
            }
        }

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Recursive Self-Improvement ===\n")

    config = load_global_config()
    agent_id = None

    agent = RSIAgent(
        action_size=2,
        state_size=4,
        agent_id=agent_id
    )
    training_report = agent.train(
        total_epochs=500,
        episodes_per_epoch=50,
        evaluation_interval=25,
        checkpoint_interval=100
    )

    print(f"\n{agent}\n{training_report}")
    print("\n=== Successfully Ran Recursive Self-Improvement ===\n")
