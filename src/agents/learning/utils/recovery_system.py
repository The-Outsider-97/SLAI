
import copy
import time

from src.agents.base.utils.main_config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Learning Agent")
printer = PrettyPrinter

class RecoverySystem:
    def __init__(self, learning_agent):
        self.learning_agent = learning_agent
        self.config = learning_agent.config.get('recovery_system', {})
        self.error_count = 0
        self.last_error_time = time.time()
        
        # Define recovery strategies with increasing severity
        self.recovery_strategies = [
            self._recover_soft_reset,
            self._recover_learning_rate_adjustment,
            self._recover_architecture_rollback,
            self._recover_strategy_switch,
            self._recover_full_reset
        ]
        
        # Configured thresholds from config
        self.error_decay_time = self.config.get('error_decay_time', 3600)  # 1 hour
        self.error_thresholds = self.config.get('error_thresholds', [3, 6, 9, 12])
        
    def decay_error_count(self):
        """Gradually reduce error count over time"""
        current_time = time.time()
        if current_time - self.last_error_time > self.error_decay_time:
            decay_factor = self.config.get('error_decay_factor', 0.5)
            self.error_count = max(0, int(self.error_count * decay_factor))
    
    def increment_error_count(self):
        """Track and update error state"""
        self.error_count += 1
        self.last_error_time = time.time()
        logger.warning(f"Error count increased to {self.error_count}")
    
    def execute_recovery(self):
        """Execute appropriate recovery strategy based on error severity"""
        self.decay_error_count()
        
        # Determine strategy level
        strategy_level = 0
        for i, threshold in enumerate(self.error_thresholds):
            if self.error_count >= threshold:
                strategy_level = i + 1
        strategy_level = min(strategy_level, len(self.recovery_strategies) - 1)
        
        logger.warning(f"Executing recovery level {strategy_level+1}")
        return self.recovery_strategies[strategy_level]()
    
    def reset_error_count(self):
        """Reset error tracking after successful recovery"""
        self.error_count = 0
        logger.info("Error count reset after successful recovery")

    # Recovery strategies implementation
    def _recover_soft_reset(self):
        """Level 1: Reset network weights and clear buffers"""
        logger.warning("Performing soft reset")
        
        # Reset neural network weights
        for agent in self.learning_agent.agents.values():
            if hasattr(agent, 'policy_net'):
                for layer in agent.policy_net.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
        
        # Clear experience buffers
        if hasattr(self.learning_agent, 'state_history'):
            self.learning_agent.state_history.clear()
        if hasattr(self.learning_agent, 'action_history'):
            self.learning_agent.action_history.clear()
        
        # Reset exploration parameters
        for agent in self.learning_agent.agents.values():
            if hasattr(agent, 'epsilon'):
                agent.epsilon = min(1.0, agent.epsilon * 1.5)
        
        return {'status': 'recovered', 'strategy': 'soft_reset'}

    def _recover_learning_rate_adjustment(self):
        """Level 2: Adaptive learning rate scaling"""
        logger.warning("Adjusting learning rates")
        lr_reduction = self.config.get('lr_reduction_factor', 0.5)
        
        for agent in self.learning_agent.agents.values():
            if hasattr(agent, 'learning_rate'):
                agent.learning_rate *= lr_reduction
                agent.learning_rate = max(agent.learning_rate, 1e-6)
        
        return {'status': 'recovered', 'strategy': 'lr_adjustment'}

    def _recover_strategy_switch(self):
        """Level 3: Fallback to safe strategy"""
        logger.warning("Switching to safe strategy")
        
        # Use basic RL as safe fallback
        self.learning_agent.active_strategy = 'rl'
        
        # If safety system exists, engage it
        if hasattr(self.learning_agent, 'safety_guard'):
            return self.learning_agent.safety_guard.execute(
                {'task': 'emergency_override'}
            )
        
        return {'status': 'recovered', 'strategy': 'strategy_switch'}

    def _recover_full_reset(self):
        """Level 4: Complete system reset"""
        logger.critical("Performing full reset!")
        
        # Preserve essential configuration
        env = self.learning_agent.env
        config = self.learning_agent.config
        shared_memory = self.learning_agent.shared_memory
        
        # Reinitialize learning agent
        self.learning_agent.__init__(
            shared_memory=shared_memory,
            agent_factory=self.learning_agent.agent_factory,
            env=env,
            config=config
        )
        
        return {'status': 'recovered', 'strategy': 'full_reset'}
            
    def _recover_architecture_rollback(self):
        """Level 3: Rollback to the last known stable architecture"""
        logger.warning("Performing architecture rollback")
        
        # Check if we have any saved architectures
        if hasattr(self.learning_agent, 'architecture_history') and self.learning_agent.architecture_history:
            # Get the last stable architecture snapshot
            last_stable_architecture = self.learning_agent.architecture_history[-1]
            
            # Rollback the policy network for each agent
            for agent_id, agent_snapshot in last_stable_architecture.items():
                if agent_id in self.learning_agent.agents:
                    agent = self.learning_agent.agents[agent_id]
                    if hasattr(agent, 'policy_net'):
                        agent.policy_net = copy.deepcopy(agent_snapshot['policy_net'])
                    if hasattr(agent, 'target_net'):
                        agent.target_net = copy.deepcopy(agent_snapshot['target_net'])
                    logger.info(f"Rolled back architecture for {agent_id}")
            
            return {'status': 'recovered', 'strategy': 'architecture_rollback'}
        else:
            logger.warning("No saved architecture found, falling back to strategy switch")
            return self._recover_strategy_switch()