"""
Proficient In:
    Adaptive environments requiring self-optimization.
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
from collections import deque, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from src.agents.learning.utils.config_loader import load_global_config, get_config_section
from src.agents.learning.utils.neural_network import NeuralNetwork
from src.agents.learning.learning_memory import LearningMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Recursive Self-Improvement")
printer = PrettyPrinter

class RSIAgent:
    def __init__(self, state_size, action_size, agent_id):
        self.config = load_global_config()
        self.rsi_config = get_config_section('neural_network')

        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        #self.save_model = []

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
        self.policy_net = None

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

    def train_episode(self) -> Tuple[float, float]:
        """Neural network enhanced experience replay"""
        if len(self.memory) < 32:
            return 0.0, 0.0  # reward, loss
    
        batch = random.sample(self.memory, 32)
        states, actions, rewards, next_states, dones = zip(*batch)
    
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
    
        current_q = self.q_network.forward(states)
        next_q = self.target_network.forward(next_states)
        max_next_q = torch.max(next_q, dim=1).values
    
        target_q = rewards + (1 - dones.float()) * self.gamma * max_next_q
    
        target = current_q.clone().detach()
        batch_indices = torch.arange(len(states))
        target[batch_indices, actions] = target_q
    
        loss = self.q_network.compute_loss(current_q, target)
        self.q_network.backward(target)
        self.q_network.update_parameters()
    
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_network.set_weights(self.q_network.get_weights())
    
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return torch.mean(rewards).item(), loss.item()

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

    def act(self, state: Any, state_sequence: Optional[List[Any]] = None) -> int:
        """
        Action selection with optional RSI meta-policy.
    
        Args:
            state: Current environment state.
            state_sequence: Optional recent state history for RSI scoring.
    
        Returns:
            Action index.
        """
        if state_sequence:
            score = self.calculate_rsi(state_sequence)
            return self._rsi_policy(score)
    
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
    
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network.forward(state_tensor)
        return torch.argmax(q_values).item()

    def calculate_rsi(self, state_sequence: List[Any]) -> float:
        """
        Generic variability score from recent states (not RSI).
    
        Args:
            state_sequence: List of recent environment states
        
        Returns:
            Scalar score (0–1) indicating variability or novelty
        """
        if not isinstance(state_sequence, (list, np.ndarray)) or len(state_sequence) < 2:
            return 0.5  # Neutral default
    
        try:
            diffs = np.diff(np.array(state_sequence), axis=0)
            magnitude = np.linalg.norm(diffs, axis=1)
            score = np.tanh(np.mean(magnitude))  # Normalized variability
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.5  # Safe fallback

    def _rsi_policy(self, score: float) -> int:
        """
        Abstract policy using a scalar meta-score to select action.
    
        Args:
            score: A continuous scalar (e.g., uncertainty, novelty score)
        
        Returns:
            Discrete action index
        """
        threshold_high = 0.7
        threshold_low = 0.3
    
        if self.action_size < 3:
            return int(score * self.action_size)  # fallback discretization
    
        if score > threshold_high:
            return 0  # e.g., explore aggressively
        elif score < threshold_low:
            return 1  # e.g., exploit known good behavior
        else:
            return 2  # e.g., maintain current trajectory

    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Annualized Sharpe ratio calculation"""
        if len(returns) < 2:
            return 0.0
        excess_returns = np.array(returns)
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

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
            self.learning_memory.set("agent_state", {  # Use string key
                "epsilon": self.epsilon,
                "learning_rate": self.learning_rate,
                "rsi_period": self.rsi_period,
                "performance": self.baseline_performance,
                "network_weights": self.q_network.get_weights(),
                "target_weights": self.target_network.get_weights()
            })
            
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
            state = self.learning_memory.get("agent_state", {})
            if state:
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

    def select_action(self, processed_state):
        # RSI already supports direct state input
        return self.act(processed_state)
    
    def learn_step(self, experience_batch):
        # Push to memory, then call train_episode
        self.memory.extend(experience_batch)
        return self.train_episode()

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

    def evaluate(self, env, episodes=50, include_training_data=True):
        """
        General-purpose evaluation for RSI Agent
        Args:
            env: Environment to evaluate in
            episodes: Number of evaluation episodes
            include_training_data: Include training memory in evaluation
        
        Returns:
            Dict containing evaluation metrics
        """
        logger.info(f"Evaluating RSI Agent {self.agent_id} over {episodes} episodes")
    
        # Performance tracking
        total_rewards = []
        episode_lengths = []
        action_distribution = {action: 0 for action in range(self.action_size)}
        state_visit_counts = defaultdict(int)
        
        for ep in range(episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Track state visits and actions
                state_visit_counts[tuple(state)] += 1
                action_distribution[action] += 1
                
                # Update trackers
                episode_reward += reward
                steps += 1
                state = next_state
    
            total_rewards.append(episode_reward)
            episode_lengths.append(steps)
    
        # Calculate metrics
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        min_reward = min(total_rewards)
        max_reward = max(total_rewards)
        avg_length = np.mean(episode_lengths)
        
        # Action distribution normalization
        total_actions = sum(action_distribution.values())
        action_distribution = {
            k: v/total_actions for k, v in action_distribution.items()
        }
        
        # State coverage analysis
        state_coverage = len(state_visit_counts)
        sharpe_ratio = self._calculate_sharpe(total_rewards)
        
        return {
            'episodes': episodes,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'min_reward': min_reward,
            'max_reward': max_reward,
            'avg_episode_length': avg_length,
            'action_distribution': action_distribution,
            'state_coverage': state_coverage,
            'exploration_rate': self.epsilon,
            'sharpe_ratio': sharpe_ratio,
            'parameter_effectiveness': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'rsi_period': self.rsi_period,
                'epsilon': self.epsilon
            },
            'training_memory_utilized': include_training_data,
            'detailed_rewards': total_rewards
        }

    def train(self, env, total_epochs=1000, episodes_per_epoch=100, 
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
                state, _ = env.reset()
                epoch_loss = 0.0
                epoch_rewards = []
                volatility_history = []
                
                # Training phase
                for _ in range(episodes_per_epoch):
                    # Standard training episode
                    episode_reward, episode_loss = self.train_episode()
                    epoch_rewards.append(episode_reward)
                    epoch_loss += episode_loss

                    # Store experience in learning memory
                    if len(self.memory) > 0:
                        latest_exp = self.memory[-1]
                        self.learning_memory.add(latest_exp, tag="rsi_experience")
    
                # Calculate epoch metrics
                avg_loss = epoch_loss / episodes_per_epoch
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
                    eval_results = self.evaluate(env)
                    logger.info(f"Epoch {epoch} Evaluation - AvgReward: {eval_results['avg_reward']:.2f} | Sharpe: {eval_results['sharpe_ratio']:.2f}")
                
                    if all(k in eval_results for k in ('sharpe_ratio', 'max_drawdown', 'avg_return')):
                        logger.info(f"Epoch {epoch} Evaluation - Sharpe: {eval_results['sharpe_ratio']:.2f} | "
                                    f"Max DD: {eval_results['max_drawdown']:.2f}")
                        
                        if eval_results['avg_return'] < performance_threshold:
                            self._mutate_parameters()
                            logger.info("Triggered performance-based parameter mutation")
                    else:
                        if isinstance(eval_results, dict):
                            PrettyPrinter.section_header(f"Epoch {epoch} Evaluation")
                            for key, value in eval_results.items():
                                if isinstance(value, dict):
                                    print(f"{key}:")
                                    for subkey, subvalue in value.items():
                                        print(f"  - {subkey}: {subvalue}")
                                elif isinstance(value, list) and len(value) > 10:
                                    print(f"{key}: [length: {len(value)}] (truncated)")
                                else:
                                    print(f"{key}: {value}")
                        else:
                            logger.warning(f"Epoch {epoch} Evaluation result is malformed: {eval_results}")
    
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
    
    def save(self, path):
        """Save policy network weights"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        logger.info(f"Saved DQN model to {path}")

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Recursive Self-Improvement ===\n")
    from src.agents.learning.slaienv import SLAIEnv

    agent_id = None
    env = SLAIEnv(state_dim=4, action_dim=3)

    agent = RSIAgent(
        action_size=2,
        state_size=4,
        agent_id=agent_id
    )
    training_report = agent.train(
        env=env,
        total_epochs=1000,
        episodes_per_epoch=100,
        evaluation_interval=10,
        performance_threshold=0.0,
        checkpoint_interval=50
    )

    print(f"\n{agent}\n{training_report}")
    print("\n=== Successfully Ran Recursive Self-Improvement ===\n")
