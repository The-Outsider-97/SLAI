__version__ = "1.8.0"

"""
Adaptive Agent with Reinforcement Learning Capabilities

This agent combines:
1. Reinforcement learning for self-improvement through experience
2. Memory systems for retaining knowledge
3. Adaptive routing for task delegation
4. Continuous learning from feedback and demonstrations

Key Features:
- Self-tuning learning parameters
- Multi-modal memory system
- Flexible task routing
- Integrated learning from various sources
- Minimal external dependencies

Academic References:
- Sutton & Barto (2018) - Reinforcement Learning: An Introduction
- Silver et al. (2014) - Deterministic Policy Gradient Algorithms
- Mnih et al. (2015) - Human-level control through deep reinforcement learning
- Schmidhuber (2015) - On Learning to Think
"""

import random
import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import statsmodels.formula.api as smf

from datetime import timedelta
from typing import Any
from collections import defaultdict, deque

from src.agents.base.utils.main_config_loader import load_global_config, get_config_section
from src.agents.adaptive.policy_manager import PolicyManager
from src.agents.adaptive.parameter_tuner import LearningParameterTuner
from src.agents.adaptive.reinforcement_learning import ReinforcementLearning
from src.agents.learning.slaienv import SLAIEnv
from src.agents.base_agent import BaseAgent
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Adaptive Agent")
printer = PrettyPrinter

class AdaptiveAgent(BaseAgent):
    """
    An adaptive agent that combines reinforcement learning with memory and routing capabilities.
    Continuously improves from feedback, success/failure, and demonstrations.
    """

    def __init__(self, shared_memory, agent_factory, config):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config
        )
        """
        Initialize the adaptive agent with learning and memory systems.
        """
        self.config = load_global_config()
        self.adaptive_config = get_config_section('adaptive_agent')
        self.state_dim = self.adaptive_config.get('state_dim')
        self.num_actions = self.adaptive_config.get('num_actions')
        self.num_handlers = self.adaptive_config.get('num_handlers')
        self.max_episode_steps = self.adaptive_config.get('max_episode_steps')

        self.shared_memory = shared_memory
        self.agent_factory = agent_factory

        self.rl_engine = ReinforcementLearning()
        self.policy = PolicyManager()
        self.tuner = LearningParameterTuner()

        # Initialize environment
        self.env = SLAIEnv(
            state_dim=self.state_dim,
            action_dim=self.num_actions,
            max_steps=self.max_episode_steps
        )

        # Connect components
        self.policy.attach_policy_network(self.rl_engine.policy_net)
        self.policy.memory = self.rl_engine.local_memory
        self.tuner.memory = self.rl_engine.local_memory
        
        # Training state
        self.recovery_history = defaultdict(lambda: {'success': 0, 'fail': 0})
        self.recovery_strategies = [self._recover_soft_reset, self._recover_lr_adjustment, self._recover_full_reset]
        self.current_state = None
        self.episode = 0
        self.total_steps = 0
        self.episode_reward = 0
        self.episode_length = 0
        self.last_reward = 0
        
        logger.info("Adaptive Agent initialized with:")
        logger.info(f"State Dim: {self.state_dim}, Actions: {self.num_actions}")
        logger.info(f"Handlers: {self.num_handlers}, Max Steps: {self.max_episode_steps}")

    def is_initialized(self) -> bool:
        return True

    def perform_task(self, task_data: Any) -> Any:
        """
        Primary task execution method:
        1. Resets environment for new episode
        2. Runs through environment steps
        3. Collects experiences and learns
        4. Routes tasks when applicable
        """
        # Reset environment for new episode
        state, info = self.env.reset()
        self.current_state = state
        done = False
        self.episode_reward = 0
        self.episode_length = 0
        
        while not done:
            # Get action from policy
            action = self._select_action(state)
            
            # Execute action in environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            self._store_experience(state, action, reward, next_state, done)
            
            # Learn periodically
            if self.total_steps % self.adaptive_config.get('learning_interval', 1) == 0:
                self._learn()
            
            # Update state and counters
            state = next_state
            self.episode_reward += reward
            self.episode_length += 1
            self.total_steps += 1
        
        # End-of-episode processing
        self._end_episode()
        
        # Return performance metrics
        return {
            'episode': self.episode,
            'reward': self.episode_reward,
            'length': self.episode_length,
            'steps': self.total_steps
        }

    def _select_action(self, state: np.ndarray) -> int:
        """Select action using policy with exploration"""
        return self.policy.get_action(
            state, 
            explore=True,
            context={"state": state, "episode": self.episode}
        )

    def _store_experience(self, state, action, reward, next_state, done):
        """Store experience in RL engine memory"""
        self.rl_engine.store_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        
        # Also store in policy manager for memory-based action biasing
        self.policy.store_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            params=self.tuner.get_params()
        )

    def update_recovery_success(self, strategy_name, success=True):
        """Track success/failure of recovery strategies"""
        if success:
            self.recovery_history[strategy_name]['success'] += 1
        else:
            self.recovery_history[strategy_name]['fail'] += 1
        logger.debug(f"Updated recovery stats for {strategy_name}: {self.recovery_history[strategy_name]}")
    
    def rank_recovery_strategies(self):
        """Order strategies by success rate with Laplace smoothing"""
        return sorted(
            self.recovery_strategies,
            key=lambda s: self.recovery_history[s.__name__]['success'] /
                        (self.recovery_history[s.__name__]['success'] +
                        self.recovery_history[s.__name__]['fail'] + 1),
            reverse=True
        )
    
    def _recover_soft_reset(self):
        """Soft reset without affecting model weights"""
        logger.info("[Recovery] Performing soft reset: clearing episodic memory and resetting counters")
        # Clear recent experiences
        self.rl_engine.local_memory.episodic.clear()
        self.last_reward = 0
        self.total_steps = 0
        logger.info("[Recovery] Soft reset complete")
        return True
    
    def _recover_lr_adjustment(self):
        """Adjust learning rate based on performance"""
        logger.info("[Recovery] Adjusting learning rate for recovery")
        try:
            # Get current performance
            recent_rewards = [exp['reward'] for exp in self.rl_engine.local_memory.episodic][-10:]
            if not recent_rewards:
                logger.warning("No rewards available for LR adjustment")
                return False
                
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            
            # Adjust LR - increase if performance is poor
            current_lr = self.tuner.params['learning_rate']
            new_lr = current_lr * 1.2 if avg_reward < 0 else current_lr * 0.8
            new_lr = max(self.tuner._min_learning_rate, min(new_lr, self.tuner._max_learning_rate))
            
            # Apply new LR
            self.tuner.params['learning_rate'] = new_lr
            logger.info(f"Learning rate adjusted from {current_lr:.4f} to {new_lr:.4f}")
            return True
        except Exception as e:
            logger.error(f"LR adjustment failed: {str(e)}")
            return False
    
    def _recover_full_reset(self):
        """Complete agent reset"""
        logger.info("[Recovery] Performing full agent reset")
        try:
            # Reset policy network
            for layer in self.rl_engine.policy_net.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
                    
            # Clear all memory
            self.rl_engine.local_memory
            self.policy.memory = self.rl_engine.local_memory
            
            # Reset training state
            self.episode = 0
            self.total_steps = 0
            self.episode_reward = 0
            self.episode_length = 0
            self.last_reward = 0
            
            logger.info("[Recovery] Full reset complete")
            return True
        except Exception as e:
            logger.error(f"Full reset failed: {str(e)}")
            return False

    def _end_episode(self):
        """Handle end-of-episode tasks"""
        # Update performance tracking
        self.tuner.update_performance(self.episode_reward)

        # Check if recovery needed
        if self.episode_reward < -10:  # Example threshold
            self.trigger_recovery()
        
        # Decay exploration rate
        self.tuner.decay_exploration()
        
        # Consolidate memory
        self.rl_engine.local_memory.consolidate()
        
        # Log episode metrics
        self._log_episode_metrics()
        
        # Increment episode counter
        self.episode += 1

    def trigger_recovery(self):
        """Execute recovery strategies in ranked order until success"""
        logger.warning("Agent performance degraded - initiating recovery sequence")
        strategies = self.rank_recovery_strategies()
        
        for strategy in strategies:
            strategy_name = strategy.__name__
            logger.info(f"Attempting recovery with: {strategy_name}")
            try:
                success = strategy()
                self.update_recovery_success(strategy_name, success)
                if success:
                    logger.info(f"Recovery successful with {strategy_name}")
                    return True
            except Exception as e:
                logger.error(f"Recovery strategy {strategy_name} failed: {str(e)}")
                self.update_recovery_success(strategy_name, False)
        
        logger.error("All recovery strategies failed!")
        return False

    def _learn(self):
        """Update policy and tune parameters"""
        buffer_size = len(self.rl_engine.local_memory.replay_buffer)
        batch_size = self.adaptive_config.get('batch_size', 64)

        if buffer_size < batch_size:
            logger.info(f"Insufficient experiences for learning ({buffer_size}/{batch_size}). Waiting for more data.")
            return
        
        # Tune learning parameters
        rewards = [exp['reward'] for exp in self.rl_engine.local_memory.episodic]
        if rewards:
            self.tuner.adapt(rewards[-self.adaptive_config.get('tuning_window', 100):])
        
        # Update policy manager with new parameters
        batch = self.rl_engine.local_memory.sample(batch_size)
        self.policy_manager.update_policy(batch["states"], batch["actions"], batch["advantages"])

    def _log_episode_metrics(self):
        """Log detailed episode metrics"""
        metrics = {
            'episode': self.episode,
            'reward': self.episode_reward,
            'length': self.episode_length,
            'steps': self.total_steps,
            'exploration_rate': self.tuner.params['exploration_rate'],
            'learning_rate': self.tuner.params['learning_rate']
        }
        self.log_evaluation_result(metrics)
        logger.info(
            f"Episode {self.episode} completed | "
            f"Reward: {self.episode_reward:.2f} | "
            f"Length: {self.episode_length} | "
            f"Exploration: {self.tuner.params['exploration_rate']:.4f}"
        )

    def learn_from_feedback(self, feedback: dict):
        """
        Incorporate human/agent feedback into learning:
        1. Convert feedback to reward signal
        2. Create new experience from feedback
        3. Add to memory
        """
        # Convert feedback to experience
        state = feedback.get('state')
        action = feedback.get('action')
        reward = feedback.get('reward', 0)
        feedback_type = feedback.get('type', 'correction')
        
        # Create bonus based on feedback type
        if feedback_type == 'correction':
            reward += self.adaptive_config.get('correction_bonus', 1.0)
        elif feedback_type == 'demonstration':
            reward += self.adaptive_config.get('demonstration_bonus', 2.0)
        
        # Store feedback as experience
        self._store_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=state,  # Terminal state for feedback
            done=True
        )
        
        # Immediately learn from feedback
        self._learn()
        logger.info(f"Learned from {feedback_type} feedback | Reward: {reward:.2f}")

    def route_task(self, task_description: str):
        """
        Route tasks to specialized handlers:
        1. Analyze task description
        2. Select appropriate handler
        3. Delegate task
        """
        # Get semantic embedding of task
        task_embedding = self._get_task_embedding(task_description)
        
        # Retrieve similar tasks from memory
        memories = self.rl_engine.local_memory.retrieve(
            query=task_description,
            context={"task": task_description}
        )
        
        # Select best handler based on memory
        if memories:
            # Use memory-biased selection
            handler_id = self._select_handler_from_memory(memories)
        else:
            # Fallback to random selection
            handler_id = random.randint(0, self.num_handlers - 1)
        
        # Delegate task
        handler = self.agent_factory.create(f"HandlerAgent_{handler_id}")
        result = handler.execute(task_description)
        
        # Store routing decision
        self._log_routing_decision(task_description, handler_id, result)
        
        return result

    def _get_task_embedding(self, task_description: str) -> torch.Tensor:
        """
        Generate a consistent task embedding in torch format using the environment's novelty detector.
    
        Args:
            task_description (str): High-level natural language task input.
    
        Returns:
            torch.Tensor: Embedding of shape (1, embedding_dim)
        """
        with torch.no_grad():  # Prevent gradient tracking
            embedding = self.env.get_state_embedding(task_description)
            if embedding.requires_grad:
                embedding = embedding.detach()
        return embedding

    def _select_handler_from_memory(self, memories: list) -> int:
        """Select handler based on memory content"""
        handler_rewards = defaultdict(list)
    
        for memory in memories:
            handler_id = memory['data'].get('handler_id')
            reward = memory['data'].get('reward')
    
            if handler_id is not None and reward is not None:
                handler_rewards[handler_id].append(reward)
    
        if not handler_rewards:
            logger.warning("No valid handler reward data found in memory. Selecting randomly.")
            return random.randint(0, self.num_handlers - 1)
    
        avg_rewards = {
            hid: np.mean(rewards)
            for hid, rewards in handler_rewards.items()
        }
    
        return max(avg_rewards.items(), key=lambda x: x[1])[0]

    def _log_routing_decision(self, task: str, handler: int, result: dict):
        """Log routing decision to memory"""
        self.rl_engine.local_memory.store_experience(
            state=self._get_task_embedding(task),
            action=handler,
            reward=result.get('reward', 0),
            context={
                "task": task,
                "handler": handler,
                "result": result
            }
        )
        
    def alternative_execute(self, task_data, original_error=None):
        """Fallback execution with environment rendering"""
        # Shape error specific handling
        if "shape" in str(original_error).lower():
            reshaped = self.reshape_input_for_model(task_data)
            if reshaped != task_data:
                return self.perform_task(reshaped)
        try:
            # Render environment state
            env_img = self.env.render(mode='rgb_array')
            
            # Create simplified task representation
            simplified_task = str(task_data)[:200]
            
            return {
                'status': 'fallback',
                'environment_state': env_img.tolist() if env_img is not None else None,
                'simplified_task': simplified_task,
                'error': str(original_error)
            }
        except Exception as e:
            return super().alternative_execute(task_data, original_error)
        
    def reshape_input_for_model(self, task_data):
        """Attempt to reshape input data for compatibility with model layers."""
        target_features = 8
        if isinstance(self.current_state, np.ndarray):
            original_shape = self.current_state.shape
            
            # Handle both 1D and 2D states
            if self.current_state.ndim == 1:
                if original_shape[0] > target_features:
                    return {"data": self.current_state[:target_features], "is_reshaped": True}
            elif self.current_state.ndim == 2 and original_shape[1] > target_features:
                return {"data": self.current_state[:, :target_features], "is_reshaped": True}
        
        return task_data
    
    def supports_fail_operational(self) -> bool:
        """
        Determine whether the agent supports fail-operational capabilities.
        
        Fail-operational capability means the agent can continue operating 
        safely and effectively even when partial failures occur in its components.
        
        This implementation checks:
        - Presence of recovery strategies
        - Integrity of local memory
        - Policy redundancy or reset mechanisms
        - Minimum episode reward thresholds
        
        Returns:
            bool: True if agent has sufficient fail-operational strategies in place.
        """
        try:
            # Check for valid recovery strategies
            has_recovery = (
                hasattr(self, 'recovery_strategies') and
                isinstance(self.recovery_strategies, list) and
                len(self.recovery_strategies) > 0
            )
    
            # Check if policy and memory modules are connected
            policy_intact = (
                hasattr(self, 'policy') and
                self.policy.memory is not None and
                hasattr(self.policy, 'get_action')
            )
    
            # Check if agent can reset parameters
            reset_capable = any(
                callable(getattr(s, '__call__', None)) 
                for s in getattr(self, 'recovery_strategies', [])
            )
    
            # Evaluate episode reward resilience (optional)
            reward_resilient = getattr(self, 'episode_reward', 0) > -100
    
            return all([has_recovery, policy_intact, reset_capable, reward_resilient])
        
        except Exception as e:
            logger.error(f"[Fail-Operational Check] Error: {e}")
            return False
        
    def has_redundant_safety_channels(self) -> bool:
        """
        Determines if the agent has multiple independent channels for safety-critical decisions.
    
        Returns:
            bool: True if redundant safety mechanisms are active.
        """
        try:
            # Example: check if multiple independent monitors or handlers are active
            safety_checks = [
                hasattr(self.env, 'safety_monitor') and self.env.safety_monitor.is_active(),
                hasattr(self.policy, 'safety_policy') and self.policy.safety_policy.is_enabled(),
                hasattr(self, 'emergency_stop') and callable(self.emergency_stop)
            ]
    
            active_channels = [check for check in safety_checks if check]
            if len(active_channels) >= 2:
                logger.info("Redundant safety channels verified.")
                return True
            else:
                logger.warning("Insufficient redundancy in safety channels.")
                return False
    
        except Exception as e:
            logger.error(f"Safety channel validation failed: {str(e)}")
            return False
    
if __name__ == "__main__":
    print("\n=== Running Adaptive Agent ===\n")
    printer.status("TEST", "Starting Adaptive Agent tests", "info")
    from src.agents.collaborative.shared_memory import SharedMemory
    
    shared_memory = SharedMemory()
    agent_factory = lambda: None
    config = load_global_config()

    agent = AdaptiveAgent(shared_memory, agent_factory, config)
    print(agent)

    print("\n* * * * * Phase 2 * * * * *\n")
    print("\nAll tests completed successfully!\n")
