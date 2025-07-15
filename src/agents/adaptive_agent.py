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
from typing import Any, Callable, Dict
from collections import defaultdict, deque

from src.agents.base.utils.main_config_loader import load_global_config, get_config_section
from src.agents.adaptive.policy_manager import PolicyManager
from src.agents.adaptive.parameter_tuner import LearningParameterTuner
from src.agents.adaptive.imitation_learning_worker import ImitationLearningWorker
from src.agents.adaptive.meta_learning_worker import MetaLearningWorker
from src.agents.adaptive.reinforcement_learning import SkillWorker
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

    def __init__(self, shared_memory,
                 agent_factory, config=None,
                 args=(), kwargs={}):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config)
        """
        Initialize the adaptive agent with learning and memory systems.
        """
        self.config = load_global_config()
        self.adaptive_config = get_config_section('adaptive_agent')
        self.state_dim = self.adaptive_config.get('state_dim')
        self.num_actions = self.adaptive_config.get('num_actions')
        self.num_handlers = self.adaptive_config.get('num_handlers')
        self.max_episode_steps = self.adaptive_config.get('max_episode_steps')
        self.base_learning_interval = self.adaptive_config.get("base_learning_interval", 10)

        self.shared_memory = shared_memory
        self.agent_factory = agent_factory

        self.tuner = LearningParameterTuner()
        self.skills = self._initialize_skills()
        if not self.skills:
            logger.error("No skills initialized! Creating fallback skill")
            self.skills = self._create_fallback_skill()
        self.policy_manager = PolicyManager()
        self.policy_manager.initialize_skills(self.skills)

        self.meta_learning = MetaLearningWorker()
        self.rl_engine = SkillWorker.create_worker(
            skill_id=-1, 
            skill_metadata={
                'name': 'global_rl_engine',
                'state_dim': self.state_dim,
                'action_dim': self.num_actions
            }
        )
        self.imitation_worker = ImitationLearningWorker(
            action_dim=self.num_actions,
            state_dim=self.state_dim,
            policy_network=self.rl_engine.actor_critic.actor
        )
        
        self._connect_imitation_learning()
        self._connect_meta_learning()

        # Initialize environment
        self.env = SLAIEnv(
            state_dim=self.state_dim,
            action_dim=self.num_actions,
            max_steps=self.max_episode_steps
        )

        self._load_agent_state()
        self._load_recovery_history()
        
        # Recovery strategies
        self.recovery_strategies = [self._recover_soft_reset, self._recover_lr_adjustment, self._recover_full_reset]

        logger.info("Adaptive Agent initialized with hierarchical architecture")
        logger.info(f"State Dim: {self.state_dim}, Skills: {len(self.skills)}")

    def _initialize_skills(self) -> Dict[int, SkillWorker]:
        """Create skill workers from configuration"""
        skills_config = self.adaptive_config.get('skills', {})
        skills = {}
        for skill_id, skill_meta in skills_config.items():
            # Ensure all skills use the agent's state_dim
            skills[skill_id] = SkillWorker.create_worker(
                skill_id, 
                {
                    **skill_meta,
                    'state_dim': self.state_dim
                }
            )
            logger.debug(f"Initialized skill {skill_id} with state_dim={self.state_dim}")
        return skills

    def _create_fallback_skill(self) -> Dict[int, SkillWorker]:
        """Create a fallback skill if configuration is empty"""
        logger.warning("Creating fallback navigation skill")
        return {
            0: SkillWorker.create_worker(0, {
                'name': 'fallback_navigation',
                'state_dim': self.state_dim,
                'action_dim': self.num_actions
            })
        }

    def _connect_imitation_learning(self):
        """Connect imitation learning to skill workers"""
        for skill_id, worker in self.skills.items():
            worker.attach_imitation_learning(self.imitation_worker)
        self.rl_engine.attach_imitation_learning(self.imitation_worker)

    def _connect_meta_learning(self):
        """Connect meta-learning to skill workers"""
        self.meta_learning.skill_worker_registry = self.skills
        for skill_id, worker in self.skills.items():
            worker.attach_meta_learning(self.meta_learning)

    def _load_agent_state(self):
        """Load agent state from shared memory if available"""
        state_key = f"agent_state:{self.name}"
        cached_state = self.shared_memory.get(state_key)
        
        if cached_state and isinstance(cached_state, dict):
            self.episode = cached_state.get('episode', 0)
            self.total_steps = cached_state.get('total_steps', 0)
            self.episode_reward = cached_state.get('episode_reward', 0)
            self.episode_length = cached_state.get('episode_length', 0)
            self.last_reward = cached_state.get('last_reward', 0)
            logger.info(f"Loaded agent state from shared memory: {cached_state}")
        else:
            # Initialize fresh state
            self.current_state = None
            self.episode = 0
            self.total_steps = 0
            self.episode_reward = 0
            self.episode_length = 0
            self.last_reward = 0

    def _load_recovery_history(self):
        """Load recovery history from shared memory"""
        recovery_key = f"recovery_history:{self.name}"
        self.recovery_history = defaultdict(lambda: {'success': 0, 'fail': 0})
        
        history_data = self.shared_memory.get(recovery_key)
        if history_data and isinstance(history_data, dict):
            for strategy, counts in history_data.items():
                self.recovery_history[strategy] = counts
            logger.info(f"Loaded recovery history from shared memory")
        else:
            logger.info("No recovery history found in shared memory, starting fresh")

    def _save_agent_state(self):
        """Save agent state to shared memory"""
        state_data = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
            'last_reward': self.last_reward
        }
        state_key = f"agent_state:{self.name}"
        self.shared_memory.set(state_key, state_data)

    def is_initialized(self) -> bool:
        return True

    def perform_task(self, task_data: Any) -> Any:
        """Primary task execution with hierarchical structure"""

        # Step 1: Preprocess task_data
        if isinstance(task_data, dict):
            context = task_data.get('context', {})
            task_goal = task_data.get('goal')
            task_type = task_data.get('type')
            self.env.set_task_context(context) if hasattr(self.env, 'set_task_context') else None
            self.policy_manager.set_task_goal(task_goal) if hasattr(self.policy_manager, 'set_task_goal') else None
            self.policy_manager.set_task_type(task_type) if hasattr(self.policy_manager, 'set_task_type') else None
            logger.info(f"Task context loaded: {context}, goal: {task_goal}, type: {task_type}")
        else:
            logger.warning("Task data not a recognized dict format")
    
        # Step 2: Standard episode loop
        state, info = self.env.reset()
        state = self._validate_state(state)
        self.current_state = state
        state_dim = len(state)
        if state_dim != self.policy_manager.state_dim:
            logger.warning(f"Updating PolicyManager state_dim from {self.policy_manager.state_dim} to {state_dim}")
            self.policy_manager.state_dim = state_dim
            self.policy_manager.initialize_skills(self.skills)
        done = False
        self.episode_reward = 0
        self.episode_length = 0

        while not done:
            # 1. Select skill using policy manager
            skill_id = self.policy_manager.select_skill(state, explore=True)
            skill = self.skills[skill_id]
            
            # 2. Execute selected skill
            skill_done = False
            skill_reward = 0
            while not skill_done and not done:
                try:
                    # Select primitive action
                    action, log_prob, entropy = skill.select_action(state)
                except Exception as e:
                    logger.error(f"Action selection failed: {e}")
                    action = random.randint(0, self.num_actions - 1)
                    log_prob, entropy = 0.0, 0.0

                # Execute action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state = self._validate_state(next_state)
                done = terminated or truncated
                
                # Store experience
                skill.store_experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    log_prob=log_prob,
                    entropy=entropy
                )
                
                # Update state and counters
                state = next_state
                skill_reward += reward
                self.episode_reward += reward
                self.episode_length += 1
                self.total_steps += 1
                
                # Check skill completion (simple step-based completion)
                skill_done = self.episode_length % self.adaptive_config.get('skill_max_steps', 10) == 0

            # 3. Finalize skill execution
            success = skill_reward > 0  # Simple success heuristic
            self.policy_manager.finalize_skill(skill_reward, success)
            
            # 4. Update skill policy
            if skill.learner_memory.size() >= self.adaptive_config.get('skill_batch_size', 32):
                skill.update_policy()
        
        # End-of-episode processing
        self._end_episode()
        return self._generate_performance_report()

    def _validate_state(self, state):
        """Ensure state is a properly formatted numeric array"""
        if isinstance(state, str):
            logger.warning(f"Invalid string state received: '{state}'. Using zero state.")
            return np.zeros(self.state_dim, dtype=np.float32)
        elif not isinstance(state, np.ndarray):
            try:
                return np.array(state, dtype=np.float32)
            except:
                logger.error("Failed to convert state to array. Using zeros.")
                return np.zeros(self.state_dim, dtype=np.float32)
        
        # Ensure correct shape
        if len(state) != self.state_dim:
            logger.warning(f"State dimension mismatch: expected {self.state_dim}, got {len(state)}")
            return state[:self.state_dim] if len(state) > self.state_dim else np.pad(
                state, 
                (0, self.state_dim - len(state)), 
                'constant'
            )
        return state

    def _end_episode(self):
        """Handle end-of-episode tasks"""
        # Update performance tracking
        self.tuner.update_performance(self.episode_reward)

        # Check if recovery needed
        if self.episode_reward < -10:  # Example threshold
            self.trigger_recovery()
        
    
        self.tuner.decay_exploration()              # Decay exploration rate
        self.rl_engine.local_memory.consolidate()   # Consolidate memory
        self._log_episode_metrics()                 # Log episode metrics
        
        # Save state to shared memory
        self._save_agent_state()
        self._save_recovery_history()
        
        # Increment episode counter
        self.episode += 1

    def _generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        skill_report = self.policy_manager.get_skill_report()
        optimization_report = self.meta_learning.get_optimization_report()
        
        return {
            'episode': self.episode,
            'total_reward': self.episode_reward,
            'steps': self.episode_length,
            'skills': skill_report,
            'hyperparameters': optimization_report
        }

    def _log_episode_metrics(self):
        """Log detailed episode metrics"""
        metrics = {
            'episode': self.episode,
            'reward': self.episode_reward,
            'length': self.episode_length,
            'steps': self.total_steps
        }
        self.log_evaluation_result(metrics)

    def _select_action(self, state: np.ndarray) -> int:
        """Select action using policy with exploration"""
        return self.policy_manager.get_action(
            state,
            context={"state": state, "episode": self.episode}
        )

    def _store_experience(self, state, action, reward, next_state, done):
        """Store experience in RL engine memory"""
        self.rl_engine.store_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=0.0,
            entropy=0.0
        )
        
        # Also store in policy manager for memory-based action biasing
        self.policy_manager.store_experience(
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
            
        # Immediately update shared memory
        self._save_recovery_history()
        
        logger.debug(f"Updated recovery stats for {strategy_name}: {self.recovery_history[strategy_name]}")
    
    def _save_recovery_history(self):
        """Save recovery history to shared memory"""
        recovery_key = f"recovery_history:{self.name}"
        # Convert defaultdict to regular dict for serialization
        history_data = dict(self.recovery_history)
        self.shared_memory.set(recovery_key, history_data)

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
            self.policy_manager.memory = self.rl_engine.local_memory
            
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

    def learn_from_demonstration(self, demo_data: dict):
        """Learn from demonstration data"""
        state = demo_data['state']
        action = demo_data['action']
        self.imitation_worker.add_demonstration(state, action)
        
        # Also store in memory for reinforcement learning
        self._store_experience(
            state=state,
            action=action,
            reward=2.0,  # Bonus for demonstrations
            next_state=state,
            done=True
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
        handler = self.agent_factory.create(
            "planning", 
            self.shared_memory,
            env=self.env,
            config= {} # {'handler_id': handler_id}
        )
        result = handler.execute(task_description)
        
        # Store routing decision in shared memory
        self._log_routing_decision(task_description, handler_id, result)
        
        return result

    def _log_routing_decision(self, task: str, handler: int, result: dict):
        """Log routing decision to memory"""
        
        # Additional shared memory logging
        routing_key = f"routing_decisions:{self.name}"
        routing_data = {
            'timestamp': time.time(),
            'task': task,
            'handler': handler,
            'result': result
        }
        
        # Get existing logs or initialize new list
        routing_logs = self.shared_memory.get(routing_key) or []
        routing_logs.append(routing_data)
        
        # Limit log size to last 100 entries
        if len(routing_logs) > 100:
            routing_logs = routing_logs[-100:]
        
        self.shared_memory.set(routing_key, routing_logs)

    def _get_task_embedding(self, task_description: str) -> torch.Tensor:
        """
        Generate a consistent task embedding in torch format using the environment's novelty detector.

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

    def predict(self, state: Any = None) -> Dict[str, Any]:
        """
        Predicts an action using the most appropriate skill and returns structured output.
    
        This is used in evaluation and testing, so it avoids exploration and randomness.

        """
        if state is None:
            state = self.current_state
        state = self._validate_state(state)

        try:
            # Select the most suitable skill using policy manager (no exploration)
            skill_id = self.policy_manager.select_skill(state, explore=False)
            skill = self.skills.get(skill_id)
    
            if skill is None:
                raise ValueError(f"Skill {skill_id} not found in initialized skills.")
    
            # Select action with deterministic policy (no entropy-driven exploration)
            action, log_prob, entropy = skill.select_action(state, explore=False)
    
            return {
                "selected_skill": skill_id,
                "action": action,
                "confidence_score": float(torch.exp(log_prob).item()) if isinstance(log_prob, torch.Tensor) else float(log_prob),
                "policy_entropy": float(entropy) if isinstance(entropy, (int, float)) else float(entropy.item()),
                "log_probability": float(log_prob) if isinstance(log_prob, (int, float)) else float(log_prob.item())
            }
    
        except Exception as e:
            logger.exception(f"[Prediction Error] AdaptiveAgent failed to predict: {e}")
            raise RuntimeError(f"AdaptiveAgent prediction failed: {str(e)}")

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
    
    def analyze_task(self, task):
        """
        Analyze an incoming task and return context dictionary.
        """
        return {
            "task_name": getattr(task, "name", "unknown"),
            "parameters": getattr(task, "parameters", {}),
            "task_type": getattr(task, "task_type", "abstract")
        }

    def calculate_learning_interval(self) -> int:
        """
        Calculate adaptive learning interval based on episode reward.
        Avoids zero or negative intervals.
        """
        base_interval = max(1, self.base_learning_interval)  # Ensure base is at least 1
        if self.episode_reward < -10:
            return max(1, base_interval // 4)
        elif self.episode_reward < 0:
            return max(1, base_interval // 2)
        elif self.episode_reward > 100:
            return base_interval * 2
        return base_interval

    def register_utility(self, name: str, utility: Any) -> None:
        """
        Register a utility object (e.g., map, database) by name.
        These utilities can be accessed later during agent operations.

        """
        if not hasattr(self, '_utilities'):
            self._utilities = {}
        self._utilities[name] = utility
        logger.debug(f"Registered utility '{name}'")

    def register_callback(self, event_name: str, callback: Callable) -> None:
        """
        Register a callback function for a specific event.
        The callback will be triggered when the event occurs.

        """
        if not hasattr(self, '_callbacks'):
            self._callbacks = defaultdict(list)
        self._callbacks[event_name].append(callback)
        logger.debug(f"Registered callback for event '{event_name}'")

    def connect_planner(self, planner: Any) -> None:
        """
        Connect to a planning agent for coordination
        """
        self.planner = planner
        logger.info(f"Connected to planning agent: {type(planner).__name__}")

    def supports_fail_operational(self) -> bool:
        """
        Determine whether the agent supports fail-operational capabilities.

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
                self.policy_manager.memory is not None and
                hasattr(self.policy_manager, 'get_action')
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

        """
        try:
            # Example: check if multiple independent monitors or handlers are active
            safety_checks = [
                hasattr(self.env, 'safety_monitor') and self.env.safety_monitor.is_active(),
                hasattr(self.policy_manager, 'safety_policy') and self.policy_manager.safety_policy.is_enabled(),
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
    
    def log_evaluation_result(self, metrics):
        return metrics

if __name__ == "__main__":
    print("\n=== Running Adaptive Agent ===\n")
    printer.status("TEST", "Starting Adaptive Agent tests", "info")
    from src.agents.collaborative.shared_memory import SharedMemory
    from src.agents.agent_factory import AgentFactory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()

    agent = AdaptiveAgent(shared_memory, agent_factory, config=None)
    print(agent)

    print("\n* * * * * Phase 2 - Save * * * * *\n")
    state = agent._save_agent_state()
    recover = agent._save_recovery_history

    printer.pretty("STATE", state, "success" if state else "error")
    printer.pretty("RECOVER", recover, "success" if recover else "error")

    print("\n* * * * * Phase 3 - Task * * * * *\n")
    task={
        "type": "navigation",
        "goal": "reach_target",
        "context": {
            "target_position": [5.0, 3.0],
            "obstacles": [[1.0, 1.0], [2.0, 2.0]],
            "start_position": [0.0, 0.0]
        }
    }
    state=np.random.rand(agent.state_dim).astype(np.float32)

    perform = agent.perform_task(task_data=task)
    action = agent._select_action(state=state)
    execute = agent.alternative_execute(task_data=task, original_error=None)

    printer.pretty("TASK", perform, "success" if perform else "error")
    printer.pretty("ACTION", action, "success" if action else "error")
    printer.status("EXECUTE", execute, "success" if execute else "error")

    print("\n* * * * * Phase 4 - Learn * * * * *\n")
    printer.pretty("LEARNER", agent._learn(), "success")
    feeedback={
        "state": state,                  # reuse state from earlier
        "action": action if isinstance(action, int) else 0,
        "reward": 1.0,
        "type": "correction"
    }
    description="Find the safest route to the target avoiding all obstacles."

    learn=agent.learn_from_feedback(feedback=feeedback)
    task2=agent.route_task(task_description=description)
    embedding = agent._get_task_embedding(task_description=description)

    printer.pretty("learn", learn, "success" if learn else "error")
    printer.pretty("TASK2", task2, "success" if task2 else "error")
    printer.pretty("TASK3", embedding, "success" if embedding is not None and embedding.any() else "error")
    

    print("\n* * * * * Phase 5 - support * * * * *\n")
    support = agent.supports_fail_operational()
    channels = agent.has_redundant_safety_channels()

    printer.pretty("Support", support, "success" if support else "error")
    printer.pretty("Channels", channels, "success" if channels else "error")
    print("\nAll tests completed successfully!\n")
