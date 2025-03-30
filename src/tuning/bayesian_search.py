"""
Enhanced Bayesian Hyperparameter Optimization with Reasoning Agent Integration
"""

import logging
import os
import json
import yaml
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from typing import Dict, List, Tuple, Callable, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BayesianSearch:
    """
    Advanced Bayesian hyperparameter optimization with reasoning capabilities.
    
    Features:
    - Robust search space configuration
    - Adaptive optimization strategies
    - Reasoning agent integration
    - Enhanced logging and result analysis
    """
    
    def __init__(self, 
                 config_file: Union[str, Path],
                 evaluation_function: Callable[[Dict], float],
                 n_calls: int = 20,
                 n_random_starts: int = 5,
                 reasoning_agent: Any = None):
        """
        Initialize Bayesian search with optional reasoning agent.
        
        Args:
            config_file: Path to YAML/JSON config file
            evaluation_function: Function that evaluates hyperparameters
            n_calls: Total optimization iterations
            n_random_starts: Random exploration steps
            reasoning_agent: Optional reasoning agent for guidance
        """
        self.config_file = Path(config_file)
        self.evaluation_function = evaluation_function
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts
        self.reasoning_agent = reasoning_agent
        
        # Initialize search space
        self.hyperparam_space, self.dimensions = self._load_search_space()
        
        # Track optimization history
        self.optimization_history = []
        self.best_score = -np.inf
        self.best_params = None
        
        # Create output directory
        self.output_dir = Path("hyperparam_tuning/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_search_space(self) -> Tuple[List[Dict], List]:
        """
        Load and validate search space configuration.
        
        Returns:
            Tuple of (space_definition, skopt_dimensions)
            
        Raises:
            ValueError: If config is invalid
            FileNotFoundError: If config file doesn't exist
        """
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
            
        logger.info(f"Loading search space from: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            if self.config_file.suffix == '.yaml':
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
                
        if 'hyperparameters' not in config:
            raise ValueError("Config must contain 'hyperparameters' section")
            
        space = []
        dimensions = []
        
        for param in config['hyperparameters']:
            if not all(k in param for k in ['name', 'type']):
                raise ValueError("Each parameter must have 'name' and 'type'")
                
            name = param['name']
            param_type = param['type']
            
            try:
                if param_type == 'int':
                    dim = Integer(param['min'], param['max'], name=name)
                elif param_type == 'float':
                    prior = param.get('prior', 'uniform')
                    dim = Real(param['min'], param['max'], prior=prior, name=name)
                elif param_type == 'categorical':
                    dim = Categorical(param['choices'], name=name)
                else:
                    raise ValueError(f"Unknown parameter type: {param_type}")
                    
                dimensions.append(dim)
                space.append({
                    'name': name,
                    'type': param_type,
                    'specs': param
                })
                
            except KeyError as e:
                raise ValueError(f"Missing required field for {name}: {e}")
                
        logger.info(f"Loaded {len(space)} hyperparameters")
        return space, dimensions

    def _log_iteration(self, params: Dict, score: float) -> None:
        """Record optimization progress."""
        self.optimization_history.append({
            'params': params,
            'score': score
        })
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = params
            logger.info(f"New best score: {score:.4f} with params: {params}")

    def _ask_reasoning_agent(self, current_state: Dict) -> Dict:
        """
        Query reasoning agent for optimization guidance.
        
        Args:
            current_state: Current optimization state
            
        Returns:
            Suggested parameter adjustments
        """
        if self.reasoning_agent is None:
            return {}
            
        try:
            # Prepare query for reasoning agent
            query = {
                'current_params': current_state['params'],
                'history': self.optimization_history,
                'search_space': self.hyperparam_space
            }
            
            # Get recommendations (format depends on your reasoning agent)
            recommendations = self.reasoning_agent.query(
                "Suggest hyperparameter adjustments for optimization"
            )
            
            return recommendations or {}
            
        except Exception as e:
            logger.warning(f"Reasoning agent query failed: {e}")
            return {}

    def run_search(self) -> Tuple[Dict, float, Any]:
        """
        Execute Bayesian optimization with optional reasoning.
        
        Returns:
            Tuple of (best_params, best_score, optimization_result)
        """
        logger.info("Starting Bayesian optimization...")
        
        @use_named_args(self.dimensions)
        def objective(**params):
            # Get reasoning agent suggestions if available
            if self.reasoning_agent:
                suggestions = self._ask_reasoning_agent({
                    'params': params,
                    'iteration': len(self.optimization_history)
                })
                
                # Blend suggestions with current params
                for k, v in suggestions.items():
                    if k in params:
                        # Simple blending - can be enhanced
                        params[k] = 0.7 * params[k] + 0.3 * v
            
            # Evaluate parameters
            score = self.evaluation_function(params)
            self._log_iteration(params, score)
            
            # We minimize, so return negative score
            return -score

        # Run optimization
        result = gp_minimize(
            func=objective,
            dimensions=self.dimensions,
            n_calls=self.n_calls,
            n_random_starts=self.n_random_starts,
            random_state=42,
            verbose=True
        )

        # Process results
        best_params = dict(zip([dim.name for dim in self.dimensions], result.x))
        best_score = -result.fun
        
        # Save results
        self._save_results(best_params, best_score, result)
        
        return best_params, best_score, result

    def _save_results(self, 
                     best_params: Dict, 
                     best_score: float,
                     result: Any) -> None:
        """
        Save optimization results to files.
        
        Args:
            best_params: Best parameters found
            best_score: Best score achieved
            result: Full optimization result
        """
        # Save best params
        best_output = {
            'best_parameters': best_params,
            'best_score': best_score,
            'config_file': str(self.config_file)
        }
        
        best_file = self.output_dir / f"best_{self.config_file.stem}.json"
        with open(best_file, 'w') as f:
            json.dump(self._make_json_serializable(best_output), f, indent=2)
            
        # Save full history
        history_file = self.output_dir / f"history_{self.config_file.stem}.json"
        with open(history_file, 'w') as f:
            json.dump(self._make_json_serializable({
                'history': self.optimization_history,
                'config': self.hyperparam_space
            }), f, indent=2)
            
        logger.info(f"Results saved to {best_file} and {history_file}")

    def _make_json_serializable(self, data: Any) -> Any:
        """
        Convert numpy types to native Python types for JSON serialization.
        """
        if isinstance(data, (np.integer, np.floating)):
            return int(data) if isinstance(data, np.integer) else float(data)
        elif isinstance(data, dict):
            return {k: self._make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(i) for i in data]
        return data


# Reasoning Agent Placeholder Integration
class ReasoningAgentPlaceholder:
    """
    Placeholder for reasoning agent integration.
    Replace with actual ReasoningAgent class from agents module.
    """
    
    def __init__(self):
        self.knowledge = {}
        
    def query(self, question: str) -> Dict:
        """
        Simulate reasoning agent response.
        
        Args:
            question: Query string
            
        Returns:
            Dictionary with suggestions
        """
        # This would be replaced with actual reasoning logic
        return {
            'learning_rate': np.random.uniform(0.0001, 0.1),
            'num_layers': np.random.randint(1, 5)
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy config
    config = {
        "hyperparameters": [
            {
                "name": "learning_rate",
                "type": "float",
                "min": 0.0001,
                "max": 0.1,
                "prior": "log-uniform"
            },
            {
                "name": "num_layers",
                "type": "int",
                "min": 1,
                "max": 5
            }
        ]
    }
    
    # Save dummy config
    config_file = Path("hyperparam_tuning/example_config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    # Dummy evaluation function
    def dummy_eval(params):
        return -((params['learning_rate'] - 0.01)**2 + (params['num_layers'] - 2)**2)
    
    # Run with reasoning agent placeholder
    bayes_opt = BayesianSearch(
        config_file=config_file,
        evaluation_function=dummy_eval,
        n_calls=10,
        n_random_starts=2,
        reasoning_agent=ReasoningAgentPlaceholder()
    )
    
    best_params, best_score, _ = bayes_opt.run_search()
    print(f"\nOptimization complete. Best score: {best_score:.4f}")
    print("Best parameters:", best_params)
