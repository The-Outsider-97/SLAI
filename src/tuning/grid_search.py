import logging
import json
import numpy as np
import itertools
from joblib import Parallel, delayed
from typing import Dict, List, Callable, Tuple, Any
from pathlib import Path
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from src.agents.reasoning_agent import ReasoningAgent
from logs.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

class GridSearch:
    """
    Enhanced Grid Search with cross-validation, agent integration, and visualization
    Features:
    - k-fold cross-validation with confidence intervals
    - Reasoning agent integration for parameter selection
    - Effect size analysis and learning curve visualization
    """
    
    def __init__(self,
                 config_file: str,
                 evaluation_function: Callable[[Dict], float],
                 reasoning_agent: ReasoningAgent = None,
                 n_jobs: int = -1,
                 cross_val_folds: int = 5):
        """
        Initialize grid search with full agent integration
        """
        self.config_file = Path(config_file)
        self.evaluation_function = evaluation_function
        self.reasoning_agent = reasoning_agent or ReasoningAgent()
        self.n_jobs = n_jobs
        self.cross_val_folds = cross_val_folds
        
        # Initialize search space
        self.hyperparam_space, self.param_names = self._load_search_space()
        self._validate_search_space()
        
        # Tracking
        self.results = []
        self.best_score = -np.inf
        self.best_params = None
        self.output_dir = Path("reports/grid_search")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_search_space(self) -> Tuple[List[List], List[str]]:
        """Load and validate search space with agent registration"""
        logger.info(f"Loading search space from: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            config = json.load(f)
            
        param_names = []
        param_values = []
        
        for param in config['hyperparameters']:
            if 'values' not in param:
                raise ValueError("Grid config requires explicit 'values' array")
                
            param_names.append(param['name'])
            param_values.append(param['values'])
            
            # Register parameter metadata with agent
            if "prior_research" in param:
                self.reasoning_agent.add_fact(
                    (param['name'], 'has_prior_study', json.dumps(param['prior_research'])),
                    confidence=0.9
                )
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
        """Combinatorial complexity check"""
        total_comb = np.prod([len(v) for v in self.hyperparam_space])
        logger.info(f"Total parameter combinations: {total_comb}")
        
        if total_comb > 1e6:
            raise ValueError(f"Combinatorial explosion: {total_comb} > 1e6")

    def _cross_validate(self, params: Dict) -> Dict[str, float]:
        """Full k-fold cross-validation with statistical reporting"""
        kf = KFold(n_splits=self.cross_val_folds, shuffle=True)
        scores = []
        
        # Dummy data - replace with actual dataset
        X = np.random.rand(100, 4)  # Example feature matrix
        y = np.random.randint(0, 2, 100)  # Example labels
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            score = self.evaluation_function(
                params,
                X_train=X[train_idx], y_train=y[train_idx],
                X_val=X[val_idx], y_val=y[val_idx]
            )
            scores.append(score)
            
            self.reasoning_agent.add_fact(
                (json.dumps(params), f'fold_{fold}_score', score),
                confidence=0.8
            )
            
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

    def _calculate_effect_size(self, current_score: float) -> float:
        """Cohen's d effect size relative to current best"""
        if not self.results:
            return 0.0
            
        pooled_std = np.sqrt(
            (self.results[-1]['scores']['std']**2 + 
             np.mean([r['scores']['std'] for r in self.results[-3:]])**2)/2
        )
        return (current_score - self.best_score) / pooled_std

    def _evaluate_combination(self, combo: tuple) -> Dict:
        """Parallel evaluation with full statistical reporting"""
        params = dict(zip(self.param_names, combo))
        cv_results = self._cross_validate(params)
        
        result = {
            'params': params,
            'scores': cv_results,
            'effect_size': self._calculate_effect_size(cv_results['mean'])
        }
        
        self.results.append(result)
        if cv_results['mean'] > self.best_score:
            self.best_score = cv_results['mean']
            self.best_params = params
            
        return result

    def run_search(self) -> Dict:
        """Parallel grid search with agent-guided analysis"""
        combinations = list(itertools.product(*self.hyperparam_space))
        
        logger.info(f"Starting grid search with {len(combinations)} combinations")
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate_combination)(combo)
            for combo in combinations
        )
        
        optimal_params = self._agent_best_selection(results)
        self._save_results(optimal_params)
        self._plot_learning_curve()
        
        return optimal_params

    def _agent_best_selection(self, results: List[Dict]) -> Dict:
        """Agent-based selection with multi-criteria optimization"""
        query = {
            "type": "parameter_selection",
            "results": results,
            "strategy": "grid_search",
            "constraints": {
                "max_complexity": 1e6,
                "hardware_limits": {"vram": 16}  # GB
            }
        }
        response = self.reasoning_agent.react_loop(json.dumps(query), max_steps=3)
        return json.loads(response['optimal_parameters'])

    def _save_results(self, best_params: Dict) -> None:
        """Save comprehensive results with metadata"""
        output = {
            'best_parameters': best_params,
            'search_space': dict(zip(self.param_names, self.hyperparam_space)),
            'statistical_metrics': {
                'effect_sizes': [r['effect_size'] for r in self.results],
                'confidence_intervals': [r['scores']['ci95'] for r in self.results]
            },
            'agent_insights': self.reasoning_agent.export_knowledge()
        }
        
        output_file = self.output_dir / f"grid_results_{self.config_file.stem}.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
            
        logger.info(f"Saved full results to {output_file}")

    def _plot_learning_curve(self) -> None:
        """Visualize effect size progression and confidence intervals"""
        plt.figure(figsize=(10, 5))
        
        # Effect Size Plot
        plt.subplot(1, 2, 1)
        plt.plot([r['effect_size'] for r in self.results], marker='o', color='#2c7bb6')
        plt.title("Effect Size Progression")
        plt.xlabel("Iteration")
        plt.ylabel("Cohen's d Effect Size")
        plt.grid(True, alpha=0.3)
        
        # Confidence Interval Plot
        plt.subplot(1, 2, 2)
        means = [r['scores']['mean'] for r in self.results]
        lower = [r['scores']['ci95'][0] for r in self.results]
        upper = [r['scores']['ci95'][1] for r in self.results]
        plt.fill_between(range(len(means)), lower, upper, alpha=0.2, color='#d7191c')
        plt.plot(means, color='#d7191c', marker='o')
        plt.title("Score Confidence Intervals")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / "grid_search_metrics.png"
        plt.savefig(plot_path)
        logger.info(f"Saved learning curve to {plot_path}")
