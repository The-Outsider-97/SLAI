
import yaml, json
import numpy as np
import itertools
import matplotlib.pyplot as plt

from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from typing import Dict, List, Callable, Tuple, Any, Optional

from logs.logger import get_logger 

logger = get_logger("GridSearch")

CONFIG_PATH = "src/tuning/configs/hyperparam.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

class GridSearch:
    """
    Comprehensive Grid Search with k-fold cross-validation, parallel execution,
    statistical analysis, and visualization.
    
    Features:
    - Configuration via YAML file for hyperparameter search space.
    - Parallel evaluation of hyperparameter combinations using joblib.
    - k-fold cross-validation for robust performance estimation.
    - Calculation of mean scores, standard deviations, and 95% confidence intervals.
    - Effect size (Cohen's d) calculation for comparing parameter sets.
    - Visualization of score progression, confidence intervals, and effect sizes.
    - Detailed JSON output of all results and best parameters.
    """
    
    def __init__(self, config,
             evaluation_function: Callable[[Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]):
        """
        Initialize the GridSearch instance.

        Args:
            config_file (str): Path to the YAML configuration file defining the hyperparameter search space.
            evaluation_function (Callable): The function used to evaluate a given set of hyperparameters.
                It must accept: (params: Dict, X_train: np.ndarray, y_train: np.ndarray, 
                                 X_val: np.ndarray, y_val: np.ndarray)
                and return a single float score (higher is better).
            n_jobs (int): Number of CPU cores to use for parallel execution. 
                          -1 means use all available cores. Defaults to -1.
            cross_val_folds (int): Number of folds for k-fold cross-validation. Defaults to 5.
            random_state (Optional[int]): Seed for the random number generator used in KFold shuffling,
                                          ensuring reproducibility. Defaults to 42.
        """
        config = load_config() or {}
        
        # Extract grid_search-specific parameters
        self.grid_search_config = config.get('grid_search', {})
        self.n_jobs = self.grid_search_config.get('n_jobs', -1)
        self.cross_val_folds = self.grid_search_config.get('cross_val_folds', 5)
        self.random_state = self.grid_search_config.get('random_state', 42)
        
        # Load hyperparameters from the root of the config
        self.hyperparam_space, self.param_names = self._load_search_space(config)
        
        # Rest of the initialization remains unchanged
        self.evaluation_function = evaluation_function
        self.results: List[Dict] = []
        self.best_score: float = -np.inf
        self.best_params: Optional[Dict] = None
        self.best_score_std: float = 0.0
        self.output_dir = Path("reports/grid_search")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.X_data: Optional[np.ndarray] = None
        self.y_data: Optional[np.ndarray] = None
        logger.info("Grid Search successfully initialized")

    def _load_search_space(self, full_config: Dict) -> Tuple[List[List[Any]], List[str]]:
        """Load hyperparameter search space from the full configuration."""
        logger.info("Loading search space from config")
        param_configs = full_config.get('hyperparameters', [])
        
        param_names = []
        param_values_list = []
        
        for param_config in param_configs:
            if not isinstance(param_config, dict) or 'name' not in param_config or 'values' not in param_config:
                raise ValueError("Invalid hyperparameter configuration")
            param_names.append(param_config['name'])
            param_values_list.append(param_config['values'])
            
        return param_values_list, param_names

    def _validate_search_space(self) -> None:
        """Validate the search space, checking for combinatorial explosion."""
        if not self.hyperparam_space: # Corresponds to empty hyperparameters list in config
            total_combinations = 1 # Will evaluate one default/empty param set
        else:
            # np.prod returns 1.0 for an empty list of lengths (if hyperparam_space was [[],[]...])
            # but _load_search_space ensures lists in param_values_list are non-empty
            list_of_lengths = [len(v_list) for v_list in self.hyperparam_space]
            total_combinations = np.prod(list_of_lengths) if list_of_lengths else 1
            
        logger.info(f"Total parameter combinations to evaluate: {total_combinations}")
        
        # This limit is arbitrary and can be adjusted based on computational resources.
        if total_combinations > 10000:
            logger.warning(f"High number of combinations ({total_combinations}). "
                           "Grid search may take a significant amount of time.")
        if total_combinations == 0 and self.hyperparam_space:
            # This case should be caught by _load_search_space if a value list is empty.
            # Defensive check.
             raise ValueError("Hyperparameter space definition results in zero combinations. "
                              "Ensure all parameter value lists are non-empty.")


    def _cross_validate(self, params: Dict) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation for a given set of parameters.
        Uses data stored in self.X_data and self.y_data.
        """
        if self.X_data is None or self.y_data is None:
            raise RuntimeError("X_data and y_data must be provided to run_search() before cross-validation.")

        kf = KFold(n_splits=self.cross_val_folds, shuffle=True, random_state=self.random_state)
        fold_scores: List[float] = []
        
        for fold_num, (train_idx, val_idx) in enumerate(kf.split(self.X_data, self.y_data)):
            X_train_fold, X_val_fold = self.X_data[train_idx], self.X_data[val_idx]
            y_train_fold, y_val_fold = self.y_data[train_idx], self.y_data[val_idx]
            
            try:
                score = self.evaluation_function(params, X_train_fold, y_train_fold, X_val_fold, y_val_fold)
                fold_scores.append(score)
            except Exception as e:
                logger.error(f"Evaluation failed for parameters {params} on fold {fold_num + 1}: {e}", exc_info=True)
                fold_scores.append(-np.inf) # Assign a very low score for failures
        
        if not fold_scores: # Should only happen if self.cross_val_folds is 0 or less.
            logger.error(f"No scores recorded for params {params}. Check cross_val_folds value.")
            return {'mean': -np.inf, 'std': 0.0, 'ci95': (-np.inf, -np.inf), 'raw_scores': []}

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        
        # Calculate 95% confidence interval for the mean score
        # Using 1.96 (Z-value for 95% CI with normal approximation)
        # More accurate for small samples would be t-distribution, but 1.96 is common.
        ci_margin = 1.96 * std_score / np.sqrt(len(fold_scores)) if len(fold_scores) > 0 and std_score > 0 else 0.0
        
        return {
            'mean': mean_score,
            'std': std_score,
            'ci95': (mean_score - ci_margin, mean_score + ci_margin),
            'raw_scores': fold_scores
        }

    def _evaluate_combination_core(self, combo: Tuple) -> Dict:
        """
        Core evaluation logic for a single hyperparameter combination.
        This method is designed to be called in parallel.
        """
        params = dict(zip(self.param_names, combo))
        cv_results = self._cross_validate(params)
        return {'params': params, 'scores': cv_results}

    def run_search(self, X_data: np.ndarray, y_data: np.ndarray) -> Optional[Dict]:
        """
        Execute the grid search over the defined hyperparameter space.

        Args:
            X_data (np.ndarray): The feature matrix for the dataset.
            y_data (np.ndarray): The target vector for the dataset.

        Returns:
            Optional[Dict]: The best hyperparameter combination found, or None if the search fails
                            or no valid combinations are found.
        """
        self.X_data = X_data
        self.y_data = y_data

        # Reset state for a new search run
        self.results = []
        self.best_score = -np.inf
        self.best_params = None
        self.best_score_std = 0.0

        if not self.param_names and not self.hyperparam_space:
            # Case: No hyperparameters specified in config, evaluate with empty params (default model)
            combinations_to_evaluate = [()] 
            logger.info("No hyperparameters specified. Evaluating a single default parameter set.")
        elif self.param_names and not self.hyperparam_space:
            # This case should ideally be caught earlier, but defensive check.
            logger.error("Parameter names are defined, but no hyperparameter values lists provided. Cannot run search.")
            return None
        else:
            combinations_to_evaluate = list(itertools.product(*self.hyperparam_space))

        if not combinations_to_evaluate:
            logger.warning("No hyperparameter combinations to evaluate. Check configuration.")
            return None
            
        num_combinations = len(combinations_to_evaluate)
        logger.info(f"Starting grid search: {num_combinations} combinations, {self.cross_val_folds} CV folds each, "
                    f"using {self.n_jobs if self.n_jobs != -1 else 'all available'} parallel jobs.")

        # Parallel execution of _evaluate_combination_core for all combinations
        all_combination_results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate_combination_core)(combo)
            for combo in combinations_to_evaluate
        )

        # Process results sequentially to calculate effect sizes and identify the best parameters
        for i, core_result in enumerate(all_combination_results):
            current_mean_score = core_result['scores']['mean']
            current_std_dev = core_result['scores']['std']

            effect_size = 0.0  # Default for the first combination or if no prior best
            if self.best_params is not None and self.best_score > -np.inf: # If a best has been established
                # Calculate Cohen's d effect size: (M1 - M2) / SD_pooled
                # SD_pooled = sqrt((SD1^2 + SD2^2) / 2) for equal group sizes (CV folds)
                # Compares current params to the best params found *so far*.
                # self.best_score_std is the std_dev of CV scores for the current self.best_params.
                
                # Ensure variance is positive before sqrt
                variance_current = current_std_dev**2
                variance_best_so_far = self.best_score_std**2
                
                pooled_std_denominator = np.sqrt((variance_current + variance_best_so_far) / 2.0)
                
                if pooled_std_denominator > 1e-9: # Avoid division by zero or tiny numbers
                    effect_size = (current_mean_score - self.best_score) / pooled_std_denominator
                else: # Denominator is zero or near-zero (e.g., both std_devs are zero)
                    if np.isclose(current_mean_score, self.best_score):
                        effect_size = 0.0
                    else: # Scores differ, but stds are zero (implies deterministic scores within CV)
                        effect_size = np.inf * np.sign(current_mean_score - self.best_score)
            
            full_result_entry = {
                'id': i, # Iteration index
                'params': core_result['params'],
                'scores': core_result['scores'], # Contains mean, std, ci95, raw_scores
                'effect_size': effect_size # Cohen's d vs. best_so_far
            }
            self.results.append(full_result_entry)

            if current_mean_score > self.best_score:
                self.best_score = current_mean_score
                self.best_params = core_result['params']
                self.best_score_std = current_std_dev # Update std for the new best
        
        if self.best_params is not None:
            logger.info(f"Grid search completed. Best parameters: {self.best_params} "
                        f"achieved score (mean): {self.best_score:.4f} (std: {self.best_score_std:.4f})")
            self._save_results_to_file()
            self.plot_search_performance()
        else:
            logger.warning("Grid search completed, but no best parameters were identified. "
                           "This might happen if all evaluations failed or returned non-finite scores.")

        return self.best_params

    def _save_results_to_file(self) -> None:
        """Save all search results, including best parameters and scores, to a JSON file."""
        if not self.results:
            logger.warning("No results available to save.")
            return

        output_content = {
            'search_configuration': {
                'config_file': str(self.config),
                'cross_validation_folds': self.cross_val_folds,
                'random_state': self.random_state,
                'parameter_names': self.param_names,
                'parameter_value_options': self.hyperparam_space
            },
            'best_result': {
                'parameters': self.best_params,
                'mean_score': self.best_score,
                'score_std_dev': self.best_score_std,
                # Find full details of the best result from self.results
                'best_result_details': next((r for r in self.results if r['params'] == self.best_params), None)
            },
            'all_evaluated_combinations': self.results 
        }
        
        results_file_path = self.output_dir / f"grid_search_results_{self.config.stem}.json"
        try:
            with open(results_file_path, 'w') as f:
                # Custom JSON encoder for numpy types and other non-serializable objects if needed
                json.dump(output_content, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
            logger.info(f"Full grid search results saved to: {results_file_path}")
        except Exception as e:
            logger.error(f"Failed to save results to {results_file_path}: {e}", exc_info=True)


    def plot_search_performance(self) -> None:
        """Generate and save plots visualizing search performance metrics."""
        if not self.results:
            logger.warning("No results available to plot.")
            return

        num_combinations_evaluated = len(self.results)
        iteration_indices = range(num_combinations_evaluated)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True) # Two plots vertically, sharing x-axis
        
        # Plot 1: Score Confidence Intervals
        mean_scores = np.array([r['scores']['mean'] for r in self.results])
        lower_ci_bounds = np.array([r['scores']['ci95'][0] for r in self.results])
        upper_ci_bounds = np.array([r['scores']['ci95'][1] for r in self.results])

        # Handle non-finite values for plotting
        finite_means_mask = np.isfinite(mean_scores)
        ax1.plot(np.array(iteration_indices)[finite_means_mask], mean_scores[finite_means_mask], 
                 color='#d7191c', marker='o', linestyle='-', markersize=5, label="Mean Score")
        
        finite_ci_mask = np.isfinite(lower_ci_bounds) & np.isfinite(upper_ci_bounds)
        ax1.fill_between(np.array(iteration_indices)[finite_ci_mask], 
                         lower_ci_bounds[finite_ci_mask], 
                         upper_ci_bounds[finite_ci_mask], 
                         alpha=0.2, color='#fdae61', label="95% CI")
        
        # Highlight the best score
        if self.best_params is not None:
            try:
                # Find index of the best parameters in the results list
                best_param_id = next(r['id'] for r in self.results if r['params'] == self.best_params)
                if np.isfinite(self.best_score):
                    ax1.scatter([best_param_id], [self.best_score], marker='*', s=150, 
                                color='gold', edgecolor='black', zorder=5, label="Best Score")
            except StopIteration:
                logger.warning("Could not locate best parameters in results for plotting highlight.")

        ax1.set_title("Score per Hyperparameter Combination (with 95% CI)")
        ax1.set_ylabel("Evaluation Score")
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        # Plot 2: Effect Size Progression
        effect_sizes = np.array([r['effect_size'] for r in self.results])
        finite_effect_sizes_mask = np.isfinite(effect_sizes)

        ax2.plot(np.array(iteration_indices)[finite_effect_sizes_mask], 
                 effect_sizes[finite_effect_sizes_mask], 
                 marker='.', linestyle='-', color='#2c7bb6', label="Cohen's d")
        ax2.axhline(0, color='grey', linestyle=':', linewidth=0.8) # Zero line for reference
        ax2.set_title("Effect Size Progression (Cohen's d vs. Best So Far)")
        ax2.set_xlabel("Parameter Combination Index")
        ax2.set_ylabel("Cohen's d Effect Size")
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout(pad=2.0)
        fig.suptitle(f"Grid Search Performance Metrics ({self.config.stem})", fontsize=14, y=1.02)
        
        plot_file_path = self.output_dir / f"grid_search_performance_{self.config.stem}.png"
        try:
            plt.savefig(plot_file_path)
            logger.info(f"Performance plots saved to: {plot_file_path}")
        except Exception as e:
            logger.error(f"Failed to save plots to {plot_file_path}: {e}", exc_info=True)
        finally:
            plt.close(fig) # Close the figure to free up memory


# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Grid Search ===\n")
    config = load_config()
    evaluation_function=None
    grid1 = GridSearch(config, evaluation_function)

    print(f"{grid1}")
    print(f"\n* * * * * Phase 2 * * * * *\n")
    def dummy_evaluation(params, X_train, y_train, X_val, y_val):
        return np.random.rand()
    grid2 = GridSearch(config, evaluation_function=dummy_evaluation)
    X_dummy = np.random.rand(100, 5)  # 100 samples, 5 features
    y_dummy = np.random.randint(0, 2, 100)  # Binary target
    
    # Execute the grid search
    best_params = grid2.run_search(X_dummy, y_dummy)
    print(f"\nBest parameters: {best_params}")
    print(f"\n* * * * * Phase 3 * * * * *\n")

    print("\n=== Successfully Ran Grid Search ===\n")
