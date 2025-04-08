"""
Enhanced Bayesian Hyperparameter Optimization with Integrated Reasoning Agent
Combines features from both implementations with improved integration
"""

import logging
import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from typing import Dict, List, Tuple, Callable, Any, Union

from src.agents.reasoning_agent import ReasoningAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BayesianSearch:
    """
    Advanced Bayesian hyperparameter optimization with full ReasoningAgent integration
    
    Features:
    - Full integration of ReasoningAgent's knowledge base
    - Adaptive parameter adjustment based on semantic reasoning
    - Continuous learning from optimization outcomes
    - Enhanced visualization and result tracking
    """
    
    def __init__(self, 
                 config_file: Union[str, Path],
                 evaluation_function: Callable[[Dict], float],
                 n_calls: int = 20,
                 n_random_starts: int = 5,
                 reasoning_agent: ReasoningAgent = None):
        """
        Initialize Bayesian search with integrated reasoning agent
        
        Args:
            config_file: Path to YAML/JSON config file
            evaluation_function: Function that evaluates hyperparameters
            n_calls: Total optimization iterations
            n_random_starts: Random exploration steps
            reasoning_agent: Pre-initialized ReasoningAgent instance
        """
        self.config_file = Path(config_file)
        self.evaluation_function = evaluation_function
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts
        self.reasoning_agent = reasoning_agent or ReasoningAgent()
        
        # Initialize search space
        self.hyperparam_space, self.dimensions = self._load_search_space()
        self.param_names = [dim.name for dim in self.dimensions]
        
        # Optimization tracking
        self.optimization_history = []
        self.best_score = -np.inf
        self.best_params = None
        self.rewards = []
        
        # Create output directory
        self.output_dir = Path("report/tuning_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_search_space(self) -> Tuple[List[Dict], List]:
        """Load and validate search space configuration"""
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

    def _register_trial_with_agent(self, params: Dict) -> None:
        """Record trial parameters with ReasoningAgent"""
        structured_facts = [(k, 'has_value', str(v)) for k, v in params.items()]
        for fact in structured_facts:
            self.reasoning_agent.add_fact(fact, confidence=0.85)

    def _analyze_trial_with_agent(self, params: Dict, score: float) -> None:
        """Update ReasoningAgent with trial results"""
        feedback = {
            tuple([k, 'has_value', str(v)]): score > self.best_score
            for k, v in params.items()
        }
        self.reasoning_agent.learn_from_interaction(feedback)

    def _ask_reasoning_agent(self, current_params: Dict) -> Dict:
        """Get parameter suggestions from ReasoningAgent"""
        try:
            # Convert parameters to semantic query
            param_facts = [f"{k} is {v}" for k, v in current_params.items()]
            query = "Suggest improvements for: " + ", ".join(param_facts)
            
            # Get recommendations from agent
            recommendations = self.reasoning_agent.react_loop(query, max_steps=3)
            
            # Convert recommendations to parameter adjustments
            adjustments = {}
            for k, v in recommendations.items():
                if k in current_params:
                    # Convert suggested values to appropriate types
                    current_type = type(current_params[k])
                    try:
                        adjustments[k] = current_type(v)
                    except ValueError:
                        logger.warning(f"Couldn't convert {v} to {current_type}")
            return adjustments
            
        except Exception as e:
            logger.warning(f"Reasoning agent query failed: {e}")
            return {}

    def _log_iteration(self, params: Dict, score: float) -> None:
        """Record optimization progress"""
        self.optimization_history.append({
            'params': params,
            'score': score
        })
        self.rewards.append(score)
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = params
            logger.info(f"New best score: {score:.4f} with params: {params}")

    def run_search(self) -> Tuple[Dict, float, Any]:
        """Execute Bayesian optimization with integrated reasoning"""
        logger.info("Starting Bayesian optimization with ReasoningAgent...")
        
        @use_named_args(self.dimensions)
        def objective(**params):
            # Register parameters with ReasoningAgent
            self._register_trial_with_agent(params)
            
            # Get reasoning agent suggestions
            suggestions = self._ask_reasoning_agent(params)
            
            # Blend suggestions with current parameters
            if suggestions:
                logger.debug(f"Applying suggestions: {suggestions}")
                for k, v in suggestions.items():
                    if isinstance(params[k], (int, float)):
                        # Weighted average for numerical parameters
                        params[k] = 0.7 * params[k] + 0.3 * v
                    else:
                        # Direct replacement for categoricals
                        params[k] = v
            
            # Evaluate parameters
            score = self.evaluation_function(params)
            self._log_iteration(params, score)
            self._analyze_trial_with_agent(params, score)
            
            return -score  # Minimize negative score

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
        best_params = dict(zip(self.param_names, result.x))
        best_score = -result.fun
        
        # Save and visualize results
        self._save_results(best_params, best_score, result)
        self._plot_rewards()
        
        return best_params, best_score, result

    def _save_results(self, 
                     best_params: Dict, 
                     best_score: float,
                     result: Any) -> None:
        """Save optimization results to files"""
        best_output = {
            'best_parameters': best_params,
            'best_score': best_score,
            'config_file': str(self.config_file)
        }
        
        best_file = self.output_dir / f"best_{self.config_file.stem}.json"
        with open(best_file, 'w') as f:
            json.dump(self._make_json_serializable(best_output), f, indent=2)
            
        history_file = self.output_dir / f"history_{self.config_file.stem}.json"
        with open(history_file, 'w') as f:
            json.dump(self._make_json_serializable({
                'history': self.optimization_history,
                'config': self.hyperparam_space
            }), f, indent=2)
            
        logger.info(f"Results saved to {best_file} and {history_file}")

    def _plot_rewards(self):
        """Visualize optimization progress"""
        plt.figure(figsize=(8, 4))
        plt.plot(self.rewards, marker='o', color='#2c7bb6')
        plt.title("Bayesian Optimization Progress with Reasoning Agent")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = self.output_dir / "optimization_progress.png"
        plt.savefig(plot_path)
        logger.info(f"Progress plot saved to {plot_path}")

    def _make_json_serializable(self, data: Any) -> Any:
        """Convert numpy types to native Python types"""
        if isinstance(data, (np.integer, np.floating)):
            return int(data) if isinstance(data, np.integer) else float(data)
        elif isinstance(data, dict):
            return {k: self._make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(i) for i in data]
        return data
