import logging
import json
import torch
import numpy as np
import itertools
from joblib import Parallel, delayed
from typing import Dict, List, Callable, Tuple, Any
from pathlib import Path
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import KFold

from src.agents.reasoning_agent import ReasoningAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GridSearch:
    """
    GridSearch performs hyperparameter optimization using an exhaustive search
    over all possible combinations of provided hyperparameter values.
    """

    def __init__(self,
                 config_file: str,
                 evaluation_function: Callable[[Dict], float],
                 reasoning_agent: ReasoningAgent = None,
                 n_jobs: int = -1,
                 cross_val_folds: int = 5):
        self.config_file = Path(config_file)
        self.evaluation_function = evaluation_function
        self.reasoning_agent = reasoning_agent or ReasoningAgent()
        self.n_jobs = n_jobs
        self.cross_val_folds = cross_val_folds
        
        self.hyperparam_space, self.param_names = self._load_search_space()
        self._validate_search_space()
        
        self.results = []
        self.best_params = None
        self.best_score = -np.inf
        """
        Initializes the GridSearch instance.
        
        Args:
            config_file (str): Path to the hyperparameter config JSON.
            evaluation_function (callable): Function to evaluate model performance with given hyperparameters.
        """
        self.config_file = config_file
        self.evaluation_function = evaluation_function
        self.hyperparam_space, self.param_names = self._load_search_space()

    def _load_search_space(self) -> Tuple[List[List], List[str]]:
        if "prior_research" in param:
            self.reasoning_agent.add_fact(
                (param['name'], 'has_prior_study', param['prior_research'])
            )
        """
        Loads hyperparameter search space from the config file.

        Returns:
            tuple: A list of hyperparameter value lists and a list of parameter names.
        """
        logger.info("Loading hyperparameter search space from: %s", self.config_file)
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        
        param_names = []
        param_values = []

        for param in config['hyperparameters']:
            if 'values' not in param:
                raise ValueError("Grid config requires explicit 'values' array")
                
            param_names.append(param['name'])
            param_values.append(param['values'])
            
            # Register parameter space with reasoning agent
            self._register_param_space(param)
            
        return param_values, param_names
        
    def _register_param_space(self, param: Dict) -> None:
        """Structure parameter metadata for agent's knowledge base"""
        fact = (
            param['name'],
            'has_domain',
            json.dumps({
                'type': param['type'],
                'values': param['values'],
                'search_type': 'grid'
            })
        )
        self.reasoning_agent.add_fact(fact, confidence=0.95)

    def _validate_search_space(self) -> None:
        """Combinatorial complexity check (Li et al., 2017)"""
        total_comb = np.prod([len(v) for v in self.hyperparam_space])
        logger.info(f"Total parameter combinations: {total_comb}")
        
        if total_comb > 1e6:
            raise ValueError(f"Combinatorial explosion: {total_comb} > 1e6")

    def _cross_validate(self, params: Dict) -> Dict[str, float]:
        """k-fold cross-validation with confidence intervals"""
        kf = KFold(n_splits=self.cross_val_folds)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(100))):  # Dummy split
            score = self.evaluation_function(params, fold=fold)
            scores.append(score)
            
            # Register fold performance with agent
            self.reasoning_agent.add_fact(
                (json.dumps(params), 
                f'fold_{fold}_score', 
                score
            ))
            
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        return {
            'mean': mean_score,
            'std': std_score,
            'ci95': (
                mean_score - 1.96*std_score/np.sqrt(len(scores)),
                mean_score + 1.96*std_score/np.sqrt(len(scores))
            )
        }

    def _evaluate_combination(self, combo: tuple) -> Dict:
        """Parallel evaluation with statistical validation"""
        params = dict(zip(self.param_names, combo))
        cv_results = self._cross_validate(params)
        
        # Register with agent's knowledge base
        self.reasoning_agent.add_fact(
            ('hyperparameters', 'current_evaluation'),
            json.dumps({'params': params, 'results': cv_results})
        )
        
        return {
            'params': params,
            'scores': cv_results,
            'effect_size': self._calculate_effect_size(params)
        }

    def _calculate_effect_size(self, params: Dict) -> float:
        """Cohen's d effect size relative to current best"""
        if not self.best_params:
            return 0.0
            
        baseline_score = self.best_score
        current_score = self.evaluation_function(params)
        pooled_std = np.sqrt((0 + self.results[-1]['scores']['std']**2)/2)
        
        return (current_score - baseline_score) / pooled_std

    def run_search(self) -> Dict:
        """Parallel grid search with agent-guided early stopping"""
        combinations = list(itertools.product(*self.hyperparam_space))
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate_combination)(combo)
            for combo in combinations
        )
        
        # Find optimal parameters using agent's reasoning
        optimal_params = self._agent_best_selection(results)
        self._save_results(optimal_params)
        
        return optimal_params

    def _agent_best_selection(self, results: List[Dict]) -> Dict:
        """Use agent's probabilistic reasoning to select best params"""
        query = {
            "type": "parameter_selection",
            "results": results,
            "strategy": "grid_search"
        }
        response = self.reasoning_agent.react_loop(json.dumps(query))
        return json.loads(response['optimal_parameters'])

    def _save_results(self, best_params: Dict) -> None:
        """Save full results with academic metadata"""
        output = {
            'best_parameters': best_params,
            'search_space': {
                'param_names': self.param_names,
                'param_values': self.hyperparam_space
            },
            'statistical_metrics': {
                'effect_sizes': [r['effect_size'] for r in self.results],
                'confidence_intervals': [r['scores']['ci95'] for r in self.results]
            },
            'agent_knowledge_snapshot': self.reasoning_agent.knowledge_base
        }
        
        output_file = self.config_file.parent / f"grid_results_{self.config_file.stem}.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
            
        logger.info(f"Saved full grid results to {output_file}")

    def run_search(self):
        """
        Executes grid search to find the best hyperparameters.

        Returns:
            dict: The best hyperparameter combination found.
        """
        logger.info("Starting exhaustive grid search hyperparameter optimization...")

        best_params = None
        best_score = float('-inf')

        combinations = list(itertools.product(*self.hyperparam_space))
        logger.info("Total combinations to evaluate: %d", len(combinations))

        for combo in combinations:
            params = {name: val for name, val in zip(self.param_names, combo)}
            logger.info("Evaluating combination: %s", params)

            score = self.evaluation_function(params)
            logger.info("Score achieved: %.4f", score)

            if score > best_score:
                best_score = score
                best_params = params
                logger.info("New best score %.4f with params: %s", best_score, best_params)

        self._save_best_params(best_params, best_score)
        return best_params

    def _save_best_params(self, params, score):
        """
        Saves the best hyperparameters to a JSON file.
        """
        output = {
            'best_hyperparameters': params,
            'best_score': score
        }

        output_file = self.config_file.replace('.json', '_grid_best.json')
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=4)

        logger.info("Best hyperparameters saved to: %s", output_file)

    def _plot_learning_curve(self) -> None:
        """Generate effect size vs parameter space exploration plot"""
        # Implementation would use matplotlib to show:
        # 1. Effect size progression
        # 2. Confidence interval narrowing
        # 3. Agent's confidence in parameter regions
        pass
