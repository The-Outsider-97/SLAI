from datetime import datetime
from typing import Dict
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.adaptive.utils.config_loader import load_global_config, get_config_section
from src.agents.learning.utils.policy_network import PolicyNetwork
from src.agents.adaptive.adaptive_memory import MultiModalMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Policy Manager")
printer = PrettyPrinter

class PolicyManager:
    """
    Manager for hierarchical skill selection.
    """
    _manager_instance = None

    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.manager_config = get_config_section('policy_manager')
        self.skills = None
        self.num_skills = 0
        self.state_dim = self.manager_config.get('state_dim', 32)
        self.hidden_layers = self.manager_config.get('hidden_layers', [64, 32])
        self.activation = self.manager_config.get('activation', 'tanh')
        self.explore=self.manager_config.get('explore', True)
        
        # Memory and tracking
        self.memory = MultiModalMemory()
        self.active_skill = None
        self.skill_start_state = None
        self.skill_history = []
        
        # Policy network for skill selection
        self.policy_network = None
        
        # Learning parameters
        self.skill_gamma = self.manager_config.get('skill_discount_factor', 0.99)
        self.learning_rate = self.manager_config.get('manager_learning_rate', 0.001)
        self._steps = 0
        
        logger.info("Policy Manager base initialized")
        logger.info(f"State Dim: {self.state_dim}")

    def set_task_goal(self, goal: str):
        self.current_goal = goal
        logger.info(f"Task goal set to: {goal}")
    
    def set_task_type(self, task_type: str):
        self.current_task_type = task_type
        logger.info(f"Task type set to: {task_type}")

    def initialize_skills(self, skills: dict):
        """Initialize the manager with specific skills"""
        if not skills:
            raise ValueError("Cannot initialize PolicyManager with empty skills dictionary")
        
        self.skills = skills
        self.num_skills = len(skills)
        
        # Always reinitialize policy network to match current state_dim
        if hasattr(self, 'policy_network'):
            del self.policy_network
            
        self.policy_network = PolicyNetwork(
            state_size=self.state_dim,
            action_size=self.num_skills
        )
        
        logger.info(f"Policy Manager skills initialized with {self.num_skills} skills")
        logger.info(f"Skill Space: {self.num_skills}")

    def select_skill(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select a skill based on current high-level state.
        
        Args:
            state: High-level state representation
            explore: Whether to allow exploration
            
        Returns:
            skill_id: Selected skill identifier
        """
        if self.policy_network is None:
            raise RuntimeError("Policy network is not initialized. Call `initialize_skills()` first.")
    
        # Convert string states to zero arrays with warning
        if isinstance(state, str):
            logger.error(f"Invalid string state received: '{state}'. Using zero state.")
            state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Ensure state is numeric array
        if not isinstance(state, (np.ndarray, list)):
            try:
                state = np.array(state, dtype=np.float32)
            except:
                logger.error("Failed to convert state to array. Using zeros.")
                state = np.zeros(self.state_dim, dtype=np.float32)
    
        # Validate state dimensions
        if len(state) != self.state_dim:
            logger.warning(f"State dimension mismatch: expected {self.state_dim}, got {len(state)}")
            state = state[:self.state_dim]  # Truncate to expected dimensions
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Retrieve relevant memories
        memory_context = {"state": state, "type": "skill_selection"}
        memories = self.memory.retrieve(
            query="policy_decision",
            context=memory_context
        )
        
        # Get skill probabilities
        with torch.no_grad():
            probs = self.policy_network(state_tensor).numpy()[0]
        
        # Exploration adjustment based on memory
        if explore and memories:
            memory_bias = self.memory._generate_memory_bias(memories)
            adjusted_probs = self._adjust_probs(probs, memory_bias)
            skill_id = np.random.choice(self.num_skills, p=adjusted_probs)
        else:
            skill_id = np.argmax(probs) if not explore else np.random.choice(self.num_skills, p=probs)
        
        # Track skill activation
        self.active_skill = skill_id
        self.skill_start_state = state
        logger.debug(f"Selected skill: {skill_id}")
        
        return skill_id

    def finalize_skill(self, final_reward: float, success: bool = False):
        """
        Complete skill execution and update manager policy.
        
        Args:
            final_reward: Cumulative reward from skill execution
            success: Whether skill achieved its objective
        """
        if self.active_skill is None:
            logger.warning("Skill finalized with no active skill")
            return
            
        # Manager reward based on skill success
        manager_reward = 1.0 if success else -0.1
        
        # Store experience
        self.store_experience(
            state=self.skill_start_state,
            action=self.active_skill,
            reward=manager_reward
        )
        
        # Add to skill history
        self.skill_history.append({
            "skill": self.active_skill,
            "reward": final_reward,
            "success": success,
            "steps": self._steps
        })
        
        # Reset tracking
        self.active_skill = None
        self.skill_start_state = None
        self._steps += 1
        
        # Update policy periodically
        if len(self.skill_history) % self.manager_config.get('update_frequency', 10) == 0:
            self.update_policy()

    def update_policy(self):
        """Update manager policy based on stored experiences"""
        if self.memory.size()['episodic'] < self.manager_config.get('min_batch_size', 16):
            return
            
        # Sample experiences
        batch_size = min(self.manager_config.get('batch_size', 32), self.memory.size()['episodic'])
        experiences = self.memory.retrieve("", {}, limit=batch_size)
        
        # Prepare training data
        states, skills, rewards = [], [], []
        for exp in experiences:
            if exp['type'] == 'episodic' and 'state' in exp['data'] and 'action' in exp['data']:
                states.append(exp['data']['state'])
                skills.append(exp['data']['action'])
                rewards.append(exp['data']['reward'])
        
        if len(states) == 0:
            return
            
        # Calculate advantages
        advantages = self.advantage_calculation(rewards)
        
        # Update policy network
        states_t = torch.FloatTensor(np.array(states))
        skills_t = torch.LongTensor(np.array(skills))
        advantages_t = torch.FloatTensor(np.array(advantages))
        
        # Forward pass
        probs = self.policy_network(states_t)
        log_probs = torch.log(probs.gather(1, skills_t.unsqueeze(1)))
        
        # Policy gradient loss
        loss = -(log_probs * advantages_t).mean()
        
        # Backpropagate and update
        self.policy_network.backward(loss)
        self.policy_network.update_parameters()
        
        logger.info(f"Manager policy updated | Loss: {loss.item():.4f}")

    def advantage_calculation(self, rewards, gamma=0.99, normalize=True):
        """
        Calculate advantages from a list of rewards using discounted return.
        """
        R = 0
        advantages = []
        
        # Calculate discounted returns in reverse
        for r in reversed(rewards):
            R = r + gamma * R
            advantages.insert(0, R)
        
        advantages = np.array(advantages)
        
        if normalize:
            mean = advantages.mean()
            std = advantages.std() + 1e-8
            advantages = (advantages - mean) / std
        
        return advantages

    def _adjust_probs(self, probs: np.ndarray, memory_bias: np.ndarray) -> np.ndarray:
        """
        Adjust skill probabilities using memory-based bias.
        """
        probs = probs.flatten()
        memory_bias = memory_bias.flatten()
        # Ensure bias matches number of skills
        if len(memory_bias) != len(probs):
            logger.warning(f"Memory bias dimension mismatch: expected {len(probs)}, got {len(memory_bias)}")
            return probs
        
        # Ensure valid probability distribution
        if not np.isclose(np.sum(probs), 1.0) or np.any(probs < 0):
            logger.warning("Invalid probability distribution in bias adjustment")
            return probs
        
        # Apply exponential scaling
        adjustment_factor = np.exp(memory_bias)
        adjusted_probs = probs * adjustment_factor
        
        # Normalize
        total = np.sum(adjusted_probs)
        if total < 1e-8:
            logger.warning("Adjusted probabilities sum to zero, using uniform distribution")
            return np.ones_like(probs) / len(probs)
        
        return adjusted_probs / total

    def store_experience(self, state, action, reward, next_state=None, done=True, context=None, params=None):
        """
        Store manager-level experience in memory.
        """
        experience = {
            'state': state,
            'action': action,  # skill_id
            'reward': reward,
            'timestamp': datetime.now(),
            'type': 'manager_decision'
        }
        
        # Store in memory
        self.memory.store_experience(
            state=state,
            action=action,
            reward=reward,
            context=context,
            params=params
        )
        
        # Log parameters if provided
        if params:
            log_params = {
                'learning_rate': params.get('learning_rate', np.nan),
                'temperature': params.get('temperature', np.nan)
            }
            self.memory.log_parameters(reward, log_params)

    def get_action(self, state, context: Dict) -> dict:
        """
        Select a high-level skill (action) based on the current state and optional task context.
    
        Args:
            state (np.ndarray): Current environment state vector
            context (Dict): Optional metadata (e.g., task goal, type, position info)
    
        Returns:
            dict: {
                'skill_id': int,
                'skill_name': str,
                'probabilities': List[float],
                'context_hash': str,
                'raw_state': np.ndarray
            }
        """
        if self.policy_network is None:
            raise RuntimeError("Policy network not initialized. Call `initialize_skills()`.")
    
        # Ensure correct shape
        if len(state) != self.state_dim:
            logger.warning(f"State dimension mismatch: expected {self.state_dim}, got {len(state)}")
            state = state[:self.state_dim]
    
        # Convert to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
        # Memory context
        memory_context = {
            "state": state.tolist(),  # for hashable memory indexing
            "goal": context.get("goal", None),
            "type": context.get("type", None),
            "episode": context.get("episode", -1)
        }
        def _sanitize_context(obj):
            if isinstance(obj, dict):
                return {k: _sanitize_context(v) for k, v in obj.items()}
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        clean_context = _sanitize_context(context)
        context_hash = self.memory._generate_context_hash(clean_context)
    
        # Skill probabilities
        with torch.no_grad():
            raw_probs = self.policy_network(state_tensor).numpy()[0]
    
        # Memory-based bias adjustment
        memory_bias = None
        retrieved = self.memory.retrieve(query="skill", context=memory_context)
        if retrieved:
            memory_bias = self.memory._generate_memory_bias(memories=retrieved)
            if memory_bias.size == raw_probs.size:
                memory_bias = memory_bias.flatten()
                probs = self._adjust_probs(raw_probs.flatten(), memory_bias)
            else:
                logger.warning(f"Memory bias shape mismatch: {memory_bias.shape} != {raw_probs.shape}")
                probs = raw_probs
        else:
            probs = raw_probs
    
        # Skill selection
        if self.explore:
            skill_id = np.random.choice(self.num_skills, p=probs)
        else:
            skill_id = np.argmax(probs)
    
        # Now skill_id is defined, so we can use it
        skill_worker = self.skills.get(skill_id)
        if skill_worker is not None:
            skill_name = skill_worker.name
        else:
            skill_name = f"skill_{skill_id}"
    
        self.active_skill = skill_id
        self.skill_start_state = state

        logger.debug(f"Selected skill {skill_id} ({skill_name}) with probs: {np.round(probs, 3).tolist()}")
    
        return {
            "skill_id": skill_id,
            "skill_name": skill_name,
            "probabilities": probs.tolist(),
            "context_hash": context_hash,
            "raw_state": state
        }

    def get_skill_report(self) -> dict:
        """Generate performance report for skills"""
        if not self.skill_history:
            return {}
            
        skill_stats = {}
        for skill_id in range(self.num_skills):
            skill_data = [s for s in self.skill_history if s['skill'] == skill_id]
            if not skill_data:
                continue
                
            skill_stats[skill_id] = {
                'usage_count': len(skill_data),
                'success_rate': sum(s['success'] for s in skill_data) / len(skill_data),
                'avg_reward': sum(s['reward'] for s in skill_data) / len(skill_data),
                'last_used': max(s['steps'] for s in skill_data)
            }
        
        return {
            'total_skill_invocations': len(self.skill_history),
            'skill_stats': skill_stats,
            'recent_success_rate': sum(s['success'] for s in self.skill_history[-10:]) / min(10, len(self.skill_history))
        }

    def consolidate_memory(self):
        """Apply memory consolidation and forgetting mechanisms"""
        self.memory.consolidate()

    def save_checkpoint(self, path):
        """Save manager state"""
        torch.save({
            'policy_state': self.policy_network.get_weights(),
            'memory_state': self.memory.state_dict(),
            'skill_history': self.skill_history,
            'steps': self._steps
        }, path)

    def load_checkpoint(self, path):
        """Load manager state"""
        checkpoint = torch.load(path)
        self.policy_network.set_weights(checkpoint['policy_state'])
        self.memory.load_state_dict(checkpoint['memory_state'])
        self.skill_history = checkpoint.get('skill_history', [])
        self._steps = checkpoint.get('steps', 0)

if __name__ == "__main__":
    # Example skills (in real system these would be RL workers)
    skills = {
        0: {"name": "navigate_to_A", "state_dim": 8},
        1: {"name": "collect_item_B", "state_dim": 6},
        2: {"name": "avoid_obstacles", "state_dim": 10}
    }
    
    print("\n=== Testing Policy Manager ===\n")
    printer.status("TEST", "Starting Policy Manager tests", "info")

    manager = PolicyManager()
    manager.initialize_skills(skills)
    print(manager)

    # Test skill selection
    state = np.random.randn(manager.state_dim)
    skill_id = manager.select_skill(state, explore=True)
    printer.pretty("Selected skill", skill_id, "success")

    # Simulate skill completion
    manager.finalize_skill(final_reward=2.5, success=True)
    printer.pretty("Skill finalized", manager.skill_history[-1], "success")

    # Test reporting
    report = manager.get_skill_report()
    printer.pretty("Skill report", report, "success")

    print("\nAll tests completed successfully!\n")
