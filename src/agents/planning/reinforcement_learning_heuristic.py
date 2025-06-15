"""
Reinforcement Learning Heuristic for Task Planning

This module implements a reinforcement learning–based heuristic that learns task-method selection policies over time 
using policy gradient algorithms. Unlike supervised learning heuristics, this module optimizes action choices based on 
long-term rewards gathered through interaction with the planning environment.

It maps task-method features to a probability distribution over available methods and updates its policy based on observed 
outcomes (e.g., task success/failure, time efficiency, or other reward signals).

Real-World Use Case:
1. Autonomous Robots: Learning which navigation or manipulation strategy yields the highest long-term success in dynamic environments.
2. Smart Assistants: Adapting method choices based on evolving user preferences and delayed feedback (e.g., task scheduling success).
3. Mission Planning: Optimizing sequences of actions in search-and-rescue or logistics under uncertainty, where immediate success 
   doesn't always reflect long-term effectiveness.
"""

import os
import json
import numpy as np
import torch
import joblib
from typing import List, Tuple
from datetime import datetime

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from src.agents.planning.utils.base_heuristic import BaseHeuristics
from src.agents.learning.utils.policy_network import PolicyNetwork
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Reinforcement Learning Heuristic")
printer = PrettyPrinter

class DotDict(dict):
    def __getattr__(self, item):
        return self.get(item)

class ReinforcementLearningHeuristic(BaseHeuristics):
    def __init__(self):
        self.config = load_global_config()

        self.heuristics_config = get_config_section('global_heuristic')
        self.trained = self.heuristics_config.get('trained')
        self.random_state = self.heuristics_config.get('random_state')
        self.planning_db_path = self.heuristics_config.get('planning_db_path')
        self.heuristic_model_path = self.heuristics_config.get('heuristic_model_path')
        os.makedirs(self.heuristic_model_path, exist_ok=True)
        self.model_path = os.path.join(self.heuristic_model_path, 'rl_heuristic_model.pkl')

        self.rlh_config = get_config_section('reinforcement_learning_heuristic')
        self.hidden_size = self.rlh_config.get('hidden_size')
        self.learning_rate = self.rlh_config.get('learning_rate')
        self.exploration_rate = self.rlh_config.get('initial_exploration')
        self.min_exploration = self.rlh_config.get('min_exploration')
        self.exploration_decay = self.rlh_config.get('exploration_decay')
        self.gamma = self.rlh_config.get('discount_factor')  # Future reward discount
        required_feature_keys = ['use_priority', 'use_resource_check', 'use_temporal_features']
        
        self.feature_config = self.rlh_config.get('feature_config')
        if not isinstance(self.feature_config, dict):
            raise ValueError("Expected 'feature_config' to be a dictionary in config")
        
        missing = [key for key in required_feature_keys if key not in self.feature_config]
        if missing:
            raise ValueError(f"Missing keys in 'feature_config': {missing}")

        self.policy_net = None
        self.candidate_methods = None
        self.method_to_index = None
        self.feature_names = self._get_feature_names()
        self.episode_cache = []  # Stores (state, action, reward) for each step
        
        logger.info(f"Reinforcement Learning Heuristic initialized with {len(self.feature_names)} features")

    def _get_feature_names(self) -> List[str]:
        """Define feature set based on configuration"""
        base_features = [
            'task_depth',
            'goal_overlap',
            'method_failure_rate',
            'state_diversity'
        ]
        if self.feature_config.get("use_priority"):
            base_features.append('task_priority')
        if self.feature_config.get("use_resource_check"):
            base_features.extend(['cpu_available', 'memory_available'])
        if self.feature_config.get("use_temporal_features"):
            base_features.extend(['time_since_creation', 'deadline_proximity'])
        return base_features

    def select_method(self, task, world_state, candidate_methods, method_stats) -> Tuple[str, float]:
        """
        Select a method using ε-greedy policy
        
        Returns:
            Tuple of (selected_method, selection_probability)
        """
        # Initialize network if not done
        if not self.policy_net:
            self.initialize_network(len(candidate_methods), candidate_methods)
        
        # Extract features for current state
        state_features = self.extract_features(task, world_state, method_stats)
        
        # Exploration vs exploitation
        if np.random.rand() < self.exploration_rate:
            # Random exploration
            selected_idx = np.random.choice(len(candidate_methods))
            prob = 1.0 / len(candidate_methods)
        else:
            # Policy-based selection
            features_tensor = torch.tensor(state_features.reshape(1, -1), dtype=torch.float32)
            probs = self.policy_net.forward(features_tensor).squeeze(0)
            dist = torch.distributions.Categorical(probs)
            selected_idx = dist.sample().item()
            prob = probs[selected_idx]
        
        # Update exploration rate
        self.exploration_rate = max(self.min_exploration, 
                                   self.exploration_rate * self.exploration_decay)
        
        # Cache for training
        self.current_state = state_features
        self.current_method_idx = selected_idx
        
        return candidate_methods[selected_idx], prob

    def initialize_network(self, num_actions: int, candidate_methods: list):
        """Initialize policy network with method names"""
        if not self.policy_net:
            self.policy_net = PolicyNetwork(
                state_size=len(self.feature_names),
                action_size=num_actions
            )
            self.candidate_methods = candidate_methods  # Store method names
            self.method_to_index = {m: i for i, m in enumerate(candidate_methods)}
            logger.info(f"Policy network initialized for {num_actions} methods")

    def extract_features(self, task, world_state, method_stats) -> np.ndarray:
        """Extract feature vector for current state and method"""
        printer.status("INIT", "Freature extractor succesfully initialized", "info")

        try:
            features = np.zeros(len(self.feature_names))
            feature_idx = 0
            
            # Base features
            features[feature_idx] = self._calculate_task_depth(task)
            feature_idx += 1
            
            features[feature_idx] = self._calculate_goal_overlap(task, world_state)
            feature_idx += 1
            
            features[feature_idx] = self._calculate_method_failure_rate(task, method_stats)
            feature_idx += 1
            
            features[feature_idx] = self._calculate_state_diversity(world_state)
            feature_idx += 1

            # Optional features
            if self.feature_config.get("use_priority"):
                features[feature_idx] = task.get('priority', 0.5)
                feature_idx += 1
                
            if self.feature_config.get("use_resource_check"):
                features[feature_idx] = world_state.get('cpu_available', 0)
                feature_idx += 1
                features[feature_idx] = world_state.get('memory_available', 0)
                feature_idx += 1
                
            if self.feature_config.get("use_temporal_features"):
                features[feature_idx] = self._time_since_creation(task)
                feature_idx += 1
                features[feature_idx] = self._deadline_proximity(task)
                feature_idx += 1
                
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return np.zeros(len(self.feature_names))
            
        return features.astype(np.float32)

    def predict_success_prob(self, task, world_state, method_stats, method_id: str) -> float:
        """
        Predict success probability for a specific method using value function approximation
        """
        # Save current method and set to target method
        original_method = task.get("selected_method")
        task["selected_method"] = method_id
    
        # Extract features for this method
        features = self.extract_features(task, world_state, method_stats)
    
        # Restore original method
        if original_method:
            task["selected_method"] = original_method
    
        # If network not initialized or method is unknown, return default
        if not self.policy_net or method_id not in self.method_to_index:
            return 0.5
    
        # Convert features to tensor for compatibility
        features_tensor = torch.tensor(features.reshape(1, -1), dtype=torch.float32)
    
        # Get probability distribution from the network
        probs = self.policy_net.forward(features_tensor).squeeze(0)
    
        # Get the index for the specified method using the correct mapping
        method_idx = self.method_to_index.get(method_id)

        # This check is a safeguard; the check at the top should prevent this.
        if method_idx is None:
            logger.warning(f"Method '{method_id}' not found in index map during prediction.")
            return 0.0

        # Return the specific probability if the index is valid
        if method_idx < len(probs):
            return probs[method_idx].item()
        
        logger.error(f"Method index {method_idx} for '{method_id}' is out of bounds "
                     f"for model output size {len(probs)}. This indicates a model/state mismatch.")
        return 0.0

    def save_model(self):
        """Save policy network weights and configuration"""
        if self.policy_net:
            # Use the get_weights method from PolicyNetwork for correct serialization
            model_data = {
                'weights': self.policy_net.get_weights(),
                'exploration_rate': self.exploration_rate,
                'feature_names': self.feature_names,
                'method_to_index': self.method_to_index  # Persist method mapping
            }
            joblib.dump(model_data, self.model_path)
            logger.info(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load policy network weights and configuration"""
        if os.path.exists(self.model_path):
            model_data = joblib.load(self.model_path)
            
            # Restore model configuration
            self.feature_names = model_data['feature_names']
            self.method_to_index = model_data['method_to_index']
            self.candidate_methods = sorted(self.method_to_index, key=self.method_to_index.get)
            
            state_size = len(self.feature_names)
            action_size = len(self.candidate_methods)

            # Initialize the network with the correct dimensions
            self.policy_net = PolicyNetwork(
                state_size=state_size,
                action_size=action_size
            )
            
            # Use the set_weights method for correct deserialization
            self.policy_net.set_weights(model_data['weights'])
            
            self.exploration_rate = model_data['exploration_rate']
            self.trained = True
            logger.info("Loaded pre-trained RL model")

    def record_outcome(self, reward: float):
        """Record outcome for the last selected method"""
        if hasattr(self, 'current_state') and hasattr(self, 'current_method_idx'):
            self.episode_cache.append((self.current_state, self.current_method_idx, reward))

    def update_policy(self):
        """Update policy using REINFORCE algorithm with discounted rewards"""
        if not self.episode_cache:
            return
            
        # Calculate discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        for _, _, reward in reversed(self.episode_cache):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        
        # Normalize rewards
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 0.0000001)
        
        # Update network weights
        for (state, action, _), advantage in zip(self.episode_cache, discounted_rewards):
            state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float32)
            probs = self.policy_net.forward(state_tensor).squeeze(0)
        
            one_hot = torch.zeros_like(probs)
            one_hot[action] = 1.0
        
            # Policy gradient: (probs - one_hot) * advantage
            grad_log_prob = (probs - one_hot) * advantage
        
            self.policy_net.backward(grad_log_prob.unsqueeze(0))  # Add batch dim
            self.policy_net.update_parameters()
        
        # Clear cache
        self.episode_cache = []
        logger.info("Policy updated with episode experiences")

    def load_planning_db(self, path: str = None):
        """Load task, world state, and method statistics from the planning database."""
        if path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, self.planning_db_path)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    
        tasks = data["tasks"]
        method_stats = {
            tuple(k.split(":")): v for k, v in data["method_stats"].items()
        }
        world_states = data["world_states"]
    
        return tasks, method_stats, world_states

if __name__ == "__main__":
    print("\n=== Running Reinforcement Learning Heuristic Test ===\n")
    printer.status("Init", "Reinforcement Learning Heuristic initialized", "success")
    from datetime import timedelta

    # Initialize heuristic
    rl_heuristic = ReinforcementLearningHeuristic()
    
    # Create test scenario
    task = {
        "name": "navigate_to_room",
        "priority": 0.8,
        "goal_state": {"location": "room_1"},
        "parent": {"name": "root_task", "parent": None},
        "creation_time": datetime.now().isoformat(),
        "deadline": (datetime.now() + timedelta(hours=2)).isoformat()
    }
    state = {
        "location": "corridor",
        "cpu_available": 0.7,
        "memory_available": 0.6,
        "battery_level": 0.9
    }
    stats = {
        ("navigate_to_room", "A*"): {"success": 8, "total": 10},
        ("navigate_to_room", "RRT"): {"success": 6, "total": 10}
    }
    methods = ["A*", "RRT", "D*"]

    # Test method selection
    print("\n* * * * * Phase 1 - Method Selection * * * * *\n")
    selected_method, prob = rl_heuristic.select_method(task, state, methods, stats)
    printer.pretty(f"Selected method: {selected_method} (prob: {prob:.3f})", "", "Success")
    
    # Simulate execution outcome
    success = selected_method == "A*"  # Assume A* always succeeds in this test
    rl_heuristic.record_outcome(1.0 if success else -0.5)
    
    # Update policy
    print("\n* * * * * Phase 2 - Policy Update * * * * *\n")
    rl_heuristic.update_policy()
    
    # Test probability prediction
    print("\n* * * * * Phase 3 - Success Prediction * * * * *\n")
    for method in methods:
        prob = rl_heuristic.predict_success_prob(task, state, stats, method)
        printer.pretty(f"Success probability for {method}: {prob:.3f}", "", "Info")
    
    # Save model
    print("\n* * * * * Phase 4 - Model Saving * * * * *\n")
    rl_heuristic.save_model()
    
    print("\n=== Successfully Ran Reinforcement Learning Heuristic ===\n")
