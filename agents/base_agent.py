"""
Comprehensive Base Agent for Multi-Agent Systems

This implementation provides a complete foundation for all agent types with:
- Core agent lifecycle management
- Shared memory integration
- Standardized communication protocols
- Performance tracking and evaluation
- Model persistence
- Configuration management

Academic References:
- Russell & Norvig (2020) "Artificial Intelligence: A Modern Approach"
- Sutton & Barto (2018) "Reinforcement Learning: An Introduction"
- Wooldridge (2009) "An Introduction to MultiAgent Systems"
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List, Tuple
import numpy as np
import torch
import pickle
import os
from datetime import datetime
from collections import deque
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BaseAgent(ABC):
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 shared_memory: Optional[Any] = None):
        """
        Initialize the base agent with configuration and shared memory.
        
        Args:
            config: Dictionary containing agent configuration parameters
            shared_memory: Reference to shared memory object for inter-agent communication
        """
        self.config = self._validate_config(config or {})
        self.shared_memory = shared_memory
        self.performance_history = deque(maxlen=1000)
        self.training_steps = 0
        self.episodes_completed = 0
        self._model = None
        self._optimizer = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialized = False
        
        # Communication properties
        self.communication_protocol = {
            'message_format': 'json',
            'supported_actions': [],
            'response_timeout': 5.0
        }
        
        logger.info(f"Initializing {self.__class__.__name__} with config: {self.config}")

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set default configuration parameters."""
        defaults = {
            'learning_rate': 0.001,
            'exploration_rate': 0.1,
            'gamma': 0.99,
            'batch_size': 32,
            'memory_capacity': 10000,
            'update_frequency': 100,
            'log_level': 'INFO'
        }
        
        # Set defaults for missing values
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
                
        # Validate critical parameters
        if config['learning_rate'] <= 0:
            raise ValueError("Learning rate must be positive")
            
        return config

    @abstractmethod
    def build_model(self) -> None:
        """
        Construct the agent's internal model architecture.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def select_action(self, 
                     state: Union[np.ndarray, torch.Tensor, Dict[str, Any]],
                     explore: bool = True) -> Any:
        """
        Select an action based on the current state.
        
        Args:
            state: Current observation/state from environment
            explore: Whether to include exploration in action selection
            
        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def train(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Train the agent on given data or through environment interaction.
        Must be implemented by subclasses.
        
        Returns:
            Dictionary containing training metrics
        """
        pass

    @abstractmethod
    def evaluate(self, 
                eval_env: Optional[Any] = None,
                n_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate agent performance.
        
        Args:
            eval_env: Environment to evaluate in (if different from training)
            n_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass

    def initialize(self) -> None:
        """
        Complete initialization after all dependencies are set.
        """
        if not self._initialized:
            self.build_model()
            self._setup_optimizer()
            self._initialized = True
            logger.info(f"{self.__class__.__name__} initialization complete")

    def _setup_optimizer(self) -> None:
        """Initialize the optimizer based on config."""
        if self._model is not None:
            self._optimizer = torch.optim.Adam(
                self._model.parameters(), 
                lr=self.config['learning_rate']
            )

    def save_model(self, path: str, include_config: bool = True) -> None:
        """
        Save the agent's model and state to disk.
        
        Args:
            path: File path to save model
            include_config: Whether to save configuration
        """
        if self._model is not None:
            state = {
                'model_state': self._model.state_dict(),
                'optimizer_state': self._optimizer.state_dict() if self._optimizer else None,
                'performance_history': list(self.performance_history),
                'training_steps': self.training_steps,
                'episodes_completed': self.episodes_completed
            }
            
            if include_config:
                state['config'] = self.config
                
            torch.save(state, path)
            logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load a saved model from disk.
        
        Args:
            path: File path to load model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model file found at {path}")
            
        state = torch.load(path, map_location=self._device)
        
        if self._model is not None:
            self._model.load_state_dict(state['model_state'])
            
        if self._optimizer and state['optimizer_state']:
            self._optimizer.load_state_dict(state['optimizer_state'])
            
        self.performance_history = deque(state.get('performance_history', []), maxlen=1000)
        self.training_steps = state.get('training_steps', 0)
        self.episodes_completed = state.get('episodes_completed', 0)
        
        if 'config' in state:
            self.config.update(state['config'])
            
        logger.info(f"Model loaded from {path}")

    def save_state(self, path: str) -> None:
        """
        Save complete agent state including memory and configuration.
        
        Args:
            path: Directory path to save state
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model weights
        if self._model is not None:
            torch.save(self._model.state_dict(), os.path.join(path, 'model.pt'))
        
        # Save optimizer state
        if self._optimizer:
            torch.save(self._optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
        
        # Save other state components
        state = {
            'config': self.config,
            'performance_history': list(self.performance_history),
            'training_steps': self.training_steps,
            'episodes_completed': self.episodes_completed,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(path, 'state.json'), 'w') as f:
            json.dump(state, f)
            
        logger.info(f"Full agent state saved to {path}")

    def load_state(self, path: str) -> None:
        """
        Load complete agent state from directory.
        
        Args:
            path: Directory path containing saved state
        """
        # Load model weights
        model_path = os.path.join(path, 'model.pt')
        if os.path.exists(model_path) and self._model is not None:
            self._model.load_state_dict(torch.load(model_path))
        
        # Load optimizer state
        optimizer_path = os.path.join(path, 'optimizer.pt')
        if os.path.exists(optimizer_path) and self._optimizer:
            self._optimizer.load_state_dict(torch.load(optimizer_path))
        
        # Load other state components
        state_path = os.path.join(path, 'state.json')
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
                
            self.config.update(state.get('config', {}))
            self.performance_history = deque(state.get('performance_history', []), maxlen=1000)
            self.training_steps = state.get('training_steps', 0)
            self.episodes_completed = state.get('episodes_completed', 0)
            
        logger.info(f"Agent state loaded from {path}")

    def get_performance_metrics(self, window: int = 100) -> Dict[str, Any]:
        """
        Get current performance metrics with moving averages.
        
        Args:
            window: Window size for moving averages
            
        Returns:
            Dictionary of performance metrics
        """
        if not self.performance_history:
            return {}
            
        recent = list(self.performance_history)[-window:]
        
        return {
            'current': recent[-1] if recent else None,
            'average_reward': np.mean([m['reward'] for m in recent]),
            'best_reward': max(m['reward'] for m in recent),
            'moving_average': self._calculate_moving_average(window),
            'success_rate': np.mean([m.get('success', 0) for m in recent]),
            'steps_per_episode': np.mean([m.get('steps', 0) for m in recent])
        }

    def _calculate_moving_average(self, window: int) -> float:
        """
        Calculate moving average of rewards.
        
        Args:
            window: Size of the moving window
            
        Returns:
            Moving average value
        """
        if len(self.performance_history) < window:
            window = len(self.performance_history)
            
        if window == 0:
            return 0.0
            
        return np.mean([m['reward'] for m in list(self.performance_history)[-window:]])

    def execute(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standard interface for task execution.
        
        Args:
            task_data: Dictionary containing task specifications
            
        Returns:
            Dictionary containing task results
        """
        if not self._initialized:
            self.initialize()
            
        task_type = task_data.get('type', 'evaluate')
        
        try:
            if task_type == 'train':
                return self._execute_train(task_data)
            elif task_type == 'evaluate':
                return self._execute_evaluate(task_data)
            elif task_type == 'predict':
                return self._execute_predict(task_data)
            elif task_type == 'communicate':
                return self._execute_communication(task_data)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'task_type': task_type
            }

    def _execute_train(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle training tasks."""
        params = task_data.get('params', {})
        result = self.train(**params)
        
        # Update performance history
        if 'reward' in result:
            self.performance_history.append({
                'reward': result['reward'],
                'steps': result.get('steps', 0),
                'success': result.get('success', False),
                'timestamp': datetime.now().isoformat()
            })
            
        return result

    def _execute_evaluate(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle evaluation tasks."""
        params = task_data.get('params', {})
        result = self.evaluate(**params)
        
        # Update performance history
        if 'reward' in result:
            self.performance_history.append({
                'reward': result['reward'],
                'steps': result.get('steps', 0),
                'success': result.get('success', True),
                'timestamp': datetime.now().isoformat(),
                'eval': True
            })
            
        return result

    def _execute_predict(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction tasks."""
        state = task_data['state']
        explore = task_data.get('explore', False)
        
        action = self.select_action(state, explore=explore)
        return {
            'status': 'success',
            'action': action,
            'explore': explore
        }

    def _execute_communication(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle inter-agent communication."""
        message = task_data['message']
        sender = task_data.get('sender', 'unknown')
        
        logger.info(f"Received message from {sender}: {message}")
        return {
            'status': 'received',
            'sender': self.__class__.__name__,
            'timestamp': datetime.now().isoformat()
        }

    def update_shared_memory(self, key: str, value: Any) -> None:
        """
        Update shared memory with a key-value pair.
        
        Args:
            key: Memory key
            value: Value to store
        """
        if self.shared_memory:
            self.shared_memory.set(key, value)
        else:
            logger.warning("Shared memory not initialized - update ignored")

    def retrieve_shared_memory(self, key: str, default: Any = None) -> Any:
        """
        Retrieve value from shared memory.
        
        Args:
            key: Memory key to retrieve
            default: Default value if key not found
            
        Returns:
            Retrieved value or default if not found
        """
        if self.shared_memory:
            return self.shared_memory.get(key, default)
            
        logger.warning("Shared memory not initialized - returning default")
        return default

    def reset(self) -> None:
        """
        Reset agent state while maintaining learned parameters.
        """
        self.performance_history.clear()
        self.training_steps = 0
        self.episodes_completed = 0
        logger.info("Agent reset - performance history cleared")

    def communicate(self, message: Dict[str, Any], recipient: Any) -> Dict[str, Any]:
        """
        Standardized communication interface between agents.
        
        Args:
            message: Dictionary containing message content
            recipient: Target agent to receive message
            
        Returns:
            Response from recipient agent
        """
        if not isinstance(message, dict):
            raise ValueError("Message must be a dictionary")
            
        # Add standard metadata
        message.update({
            'sender': self.__class__.__name__,
            'timestamp': datetime.now().isoformat(),
            'protocol': self.communication_protocol['message_format']
        })
        
        try:
            response = recipient.execute({
                'type': 'communicate',
                'message': message
            })
            return response
        except Exception as e:
            logger.error(f"Communication failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def __str__(self) -> str:
        """
        String representation of the agent.
        """
        return (f"{self.__class__.__name__}(initialized={self._initialized}, "
                f"device={self._device}, "
                f"training_steps={self.training_steps})")

    def __repr__(self) -> str:
        return self.__str__()

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_model'):
            del self._model
        if hasattr(self, '_optimizer'):
            del self._optimizer
