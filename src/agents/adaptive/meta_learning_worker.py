
import torch
import random
import numpy as np

from scipy.stats import norm
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.agents.adaptive.utils.config_loader import load_global_config, get_config_section
from src.agents.adaptive.adaptive_memory import MultiModalMemory
from src.tuning.utils.bayesian_neural_network import BayesianNeuralNetwork
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Meta Learning Worker")
printer = PrettyPrinter

class MetaLearningWorker:
    """
    Meta-Learning Worker for hyperparameter optimization across skills
    - Uses Bayesian Neural Network for uncertainty-aware hyperparameter tuning
    - Optimizes hyperparameters using expected improvement acquisition
    - Maintains memory of hyperparameter-performance relationships
    - Coordinates with SkillWorkers for parameter updates
    """
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.meta_config = get_config_section('meta_learning')
        self.training_epochs = self.meta_config.get('training_epochs')
        self.skill_worker_registry = {}
        self.worker_registry_name = self.meta_config.get('skill_workers', 'SkillWorkerRegistry')
        
        # Hyperparameter space definition
        self.hyperparameter_space = {
            'learning_rate': (0.00001, 0.01),
            'exploration_rate': (0.01, 0.3),
            'entropy_coef': (0.001, 0.1),
            'discount_factor': (0.9, 0.999)
        }
        self.hyperparam_names = list(self.hyperparameter_space.keys())
        self.num_hyperparams = len(self.hyperparameter_space)
        
        # Bayesian Neural Network for performance prediction
        self.bnn = BayesianNeuralNetwork(
            layer_sizes=[self.num_hyperparams] + 
                        self.meta_config.get('hidden_layers', [64, 64]) + 
                        [1]  # Single output (performance)
        )
        
        # Memory for hyperparameter-performance pairs
        self.memory = MultiModalMemory()
        self.performance_history = []

        self.exploration_factor = self.meta_config.get('exploration_factor', 0.1)
        self.update_frequency = self.meta_config.get('update_frequency', 100)
        self.batch_size = self.meta_config.get('batch_size', 32)
        self.num_candidates = self.meta_config.get('num_candidates', 100)
        
        logger.info("Meta Learning Worker initialized")
        logger.info(f"Optimizing hyperparameters: {', '.join(self.hyperparam_names)}")
        logger.info(f"Using worker registry: {self.worker_registry_name}")

    def get_worker_registry(self):
        """Retrieve worker registry from memory"""
        # Access the semantic memory store directly
        registry_key = f"ctx_{self.worker_registry_name[:6]}"
        
        if registry_key in self.memory.semantic:
            registry_data = self.memory.semantic[registry_key]['data']
            if isinstance(registry_data, dict):
                return registry_data
            else:
                logger.warning(f"Registry data is not a dictionary: {type(registry_data)}")
        else:
            logger.warning(f"Worker registry '{self.worker_registry_name}' not found in memory")
        
        return {}

    def collect_performance_metrics(self) -> Dict[int, Dict]:
        """Collect performance metrics from all registered skill workers"""
        metrics = {}
        registry = self.get_worker_registry()
        
        for worker_id, worker in registry.items():
            # Skip if worker doesn't have performance metrics method
            if not hasattr(worker, 'get_performance_metrics'):
                logger.warning(f"Worker {worker_id} missing performance metrics method")
                continue
                
            try:
                metrics[worker_id] = worker.get_performance_metrics()
            except Exception as e:
                logger.error(f"Error collecting metrics from worker {worker_id}: {str(e)}")
        return metrics

    def store_hyperparameter_experience(self, hyperparams: Dict, performance: float):
        """Store hyperparameter configuration and its performance"""
        # Create context for semantic memory
        context = {
            'type': 'hyperparameter_config',
            'params': hyperparams
        }
        
        # Store in memory
        self.memory.store_experience(
            state=None,
            action=None,
            reward=performance,
            context=context,
            params=hyperparams
        )
        
        # Add to performance history
        self.performance_history.append({
            'hyperparams': hyperparams,
            'performance': performance,
            'timestamp': datetime.now()
        })

    def suggest_hyperparameters(self) -> Dict[str, float]:
        """
        Suggest new hyperparameters using Bayesian optimization
        with Expected Improvement acquisition function
        """
        # If no history, return random configuration
        if not self.performance_history:
            return self._random_hyperparameters()
        
        # Prepare training data
        X, y = self._prepare_training_data()
        
        # Train BNN on historical data
        self._train_bnn(X, y)
        
        # Generate candidate configurations
        candidates = [self._random_hyperparameters() for _ in range(self.num_candidates)]
        candidate_array = np.array([self._hyperparams_to_array(c) for c in candidates])
        
        # Predict performance and uncertainty
        means, stds = self.bnn.predict(candidate_array, num_samples=100)
        means = means.squeeze()
        stds = stds.squeeze()
        
        # Calculate Expected Improvement
        best_observed = max(y)
        with np.errstate(divide='ignore'):
            z = (means - best_observed - self.exploration_factor) / stds
            ei = (means - best_observed - self.exploration_factor) * norm.cdf(z) + stds * norm.pdf(z)
        ei[stds == 0] = 0  # Handle zero std cases
        
        # Select candidate with highest EI
        best_idx = np.argmax(ei)
        return candidates[best_idx]

    def update_skill_hyperparameters(self, hyperparams: Dict[str, float]):
        """Update hyperparameters for all skill workers"""
        for worker in self.skill_worker_registry.values():
            # Update learning parameters
            worker.learning_rate = hyperparams['learning_rate']
            worker.exploration_rate = hyperparams['exploration_rate']
            worker.entropy_coef = hyperparams['entropy_coef']
            worker.gamma = hyperparams['discount_factor']
            
            # Update optimizers with new learning rates
            for param_group in worker.actor_optimizer.param_groups:
                param_group['lr'] = hyperparams['learning_rate']
            for param_group in worker.critic_optimizer.param_groups:
                param_group['lr'] = hyperparams['learning_rate']
        
        logger.info("Updated skill hyperparameters")

    def optimization_step(self):
        """Perform one optimization cycle"""
        # Collect current performance
        metrics = self.collect_performance_metrics()
        avg_performance = np.mean([m['recent_reward'] for m in metrics.values()])
        
        # Collect current hyperparameters (using first worker as representative)
        worker = next(iter(self.skill_worker_registry.values()))
        current_hyperparams = {
            'learning_rate': worker.learning_rate,
            'exploration_rate': worker.exploration_rate,
            'entropy_coef': worker.entropy_coef,
            'discount_factor': worker.gamma
        }
        
        # Store current configuration
        self.store_hyperparameter_experience(current_hyperparams, avg_performance)
        
        # Suggest and apply new hyperparameters
        new_hyperparams = self.suggest_hyperparameters()
        self.update_skill_hyperparameters(new_hyperparams)
        
        logger.info(f"Meta-optimization step completed | "
                    f"Performance: {avg_performance:.4f}")

    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare hyperparameter-performance data for BNN training"""
        X = []
        y = []
        
        for experience in self.performance_history:
            hyperparams = experience['hyperparams']
            X.append(self._hyperparams_to_array(hyperparams))
            y.append(experience['performance'])
        
        return np.array(X), np.array(y)

    def _train_bnn(self, X: np.ndarray, y: np.ndarray):
        """Train Bayesian Neural Network on historical data"""
        # Normalize targets
        y_mean, y_std = np.mean(y), np.std(y)
        y_normalized = (y - y_mean) / (y_std + 1e-8)
        
        # Train in mini-batches
        num_batches = int(np.ceil(len(X) / self.batch_size))
        for _ in range(self.training_epochs):
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(X))
                
                batch_X = X[indices[start_idx:end_idx]]
                batch_y = y_normalized[indices[start_idx:end_idx]]
                
                # Train step
                self.bnn.train_step(batch_X, batch_y.reshape(-1, 1))

    def _hyperparams_to_array(self, hyperparams: Dict) -> np.ndarray:
        """Convert hyperparameter dictionary to normalized array"""
        array = []
        for param in self.hyperparam_names:
            # Normalize to [0, 1] based on defined ranges
            low, high = self.hyperparameter_space[param]
            value = hyperparams[param]
            normalized = (value - low) / (high - low)
            array.append(normalized)
        return np.array(array)

    def _array_to_hyperparams(self, array: np.ndarray) -> Dict:
        """Convert normalized array back to hyperparameters"""
        hyperparams = {}
        for i, param in enumerate(self.hyperparam_names):
            low, high = self.hyperparameter_space[param]
            value = array[i] * (high - low) + low
            hyperparams[param] = value
        return hyperparams

    def _random_hyperparameters(self) -> Dict[str, float]:
        """Generate random hyperparameters within defined ranges"""
        return {
            param: random.uniform(low, high)
            for param, (low, high) in self.hyperparameter_space.items()
        }

    def get_optimization_report(self) -> Dict:
        """Generate optimization progress report"""
        if not self.performance_history:
            return {}
            
        performances = [e['performance'] for e in self.performance_history]
        best_idx = np.argmax(performances)
        
        return {
            'total_configurations': len(self.performance_history),
            'best_performance': performances[best_idx],
            'best_hyperparams': self.performance_history[best_idx]['hyperparams'],
            'recent_performance': performances[-1] if performances else 0.0,
            'performance_history': performances
        }

    def save_checkpoint(self, path: str):
        """Save meta-learning state"""
        checkpoint = {
            'performance_history': self.performance_history,
            'bnn_state': self.bnn.save_state(),  # Requires save_state method in BNN
            'optimization_step': self.optimization_step_count
        }
        torch.save(checkpoint, path)
        logger.info(f"Meta-learning checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load meta-learning state"""
        checkpoint = torch.load(path)
        self.performance_history = checkpoint['performance_history']
        self.bnn.load_state(checkpoint['bnn_state'])  # Requires load_state method in BNN
        self.optimization_step_count = checkpoint['optimization_step']
        logger.info(f"Meta-learning checkpoint loaded from {path}")

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Meta Learning Worker ===\n")
    printer.status("TEST", "Starting Meta Learning Worker tests", "info")
    skill_workers = {}

    worker = MetaLearningWorker()
    collector = worker.collect_performance_metrics()
    class SkillWorker:
        def get_performance_metrics(self):
            return {'recent_reward': 0.85}
    
    registry = {1: SkillWorker()}
    registry_key = f"ctx_{worker.worker_registry_name[:6]}"
    
    # Store registry in semantic memory
    worker.memory.semantic[registry_key] = {
        'data': registry,
        'strength': 1.0,
        'last_accessed': datetime.now(),
        'context_hash': worker.worker_registry_name
    }
    
    collector = worker.collect_performance_metrics()
    printer.pretty("Collect", collector, "success" if collector else "error")

    print("\n* * * * * Phase 2 - Hyperparam * * * * *\n")
    hyperparams ={
        'learning_rate': 0.001,
        'exploration_rate': 0.1,
        'entropy_coef': 0.01,
        'discount_factor': 0.95
    }
    performance =0.78

    worker.store_hyperparameter_experience(hyperparams=hyperparams, performance=performance)
    printer.pretty("Store", "Experience stored", "success" if "Experience stored" else "error")
    
    suggest = worker.suggest_hyperparameters()
    printer.pretty("Suggest", suggest, "success" if suggest else "error")
    
    worker.update_skill_hyperparameters(hyperparams=suggest)
    printer.pretty("Update", "Hyperparameters updated", "success")
    print("\nAll tests completed successfully!\n")