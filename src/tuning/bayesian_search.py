"""
Bayesian Hyperparameter Optimization with Integrated Reasoning Agent
Combines features from both implementations with improved integration
"""


from datetime import datetime
import yaml, json
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args, OptimizeResult
from typing import Dict, List, Tuple, Callable, Any, Union, Optional

from src.tuning.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("BayesianSearch")
printer = PrettyPrinter

class BayesianSearch:
    """
    Advanced Bayesian hyperparameter optimization with full ReasoningAgent integration
    
    Features:
    - Full integration of ReasoningAgent's knowledge base
    - Adaptive parameter adjustment based on semantic reasoning
    - Continuous learning from optimization outcomes
    - Enhanced visualization and result tracking
    """
    
    def __init__(self, evaluation_function: Callable[[Dict[str, Any]], float], model_type=None):
        """
        Initialize the BayesianSearch instance.

        Args:
            config (Union[str, Path]): Path to the YAML or JSON configuration file 
                                            defining the hyperparameter search space.
            evaluation_function (Callable[[Dict[str, Any]], float]): 
                The function to be minimized. It takes a dictionary of 
                hyperparameters as input and returns a single float score.
                Note: Bayesian optimization typically minimizes the objective, 
                so if your function returns a score where higher is better, 
                you should return its negative.
            n_calls (int): Total number of evaluations (calls to the evaluation_function).
                           This includes `n_initial_points`.
            n_initial_points (int): Number of evaluations of `evaluation_function` with
                                   randomly D-chosen points before Gaussian Process fitting.
            random_state (Optional[int]): Seed for reproducible results.
            output_dir_name (str): Name of the directory to save reports and plots.
                                   Will be created under a 'reports' parent directory.
        """
        self.config = load_global_config()
        self.bayesian_config = get_config_section('bayesian_search')
        self.n_calls = self.bayesian_config.get('n_calls')
        self.output_dir = self.bayesian_config.get('output_dir')
        self.summary_dir = self.bayesian_config.get('summary_dir')
        self.random_state = self.bayesian_config.get('random_state')
        self.n_initial_points = self.bayesian_config.get('n_initial_points')
        self.model_type = model_type if model_type is not None else self.bayesian_config.get('model_type', 'GradientBoosting')
        if not self.model_type:
            self.model_type = 'GradientBoosting'

        output_dir_name: str = "bayesian_search"

        self.evaluation_function = evaluation_function
        self.search_space_config, self.dimensions = self._load_search_space()
        self.param_names: List[str] = [dim.name for dim in self.dimensions]

        self.optimization_history: List[Dict[str, Any]] = []
        self.best_score_so_far: float = np.inf # Assuming minimization; use -np.inf for maximization
        self.best_params_so_far: Optional[Dict[str, Any]] = None

        self.output_dir = Path() / output_dir_name # Standardize reports location
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"BayesianSearch initialized. Results will be saved to: {self.output_dir.resolve()}")

    def _load_search_space(self) -> Tuple[List[Dict[str, Any]], List[Union[Real, Integer, Categorical]]]:
        logger.info("Loading search space from config")
        
        if 'hyperparameters' not in self.config:
            raise ValueError("Config missing 'hyperparameters' section")
        
        config_data = self.config
        
        # Use the instance model_type
        model_type = self.model_type
        if not model_type:
            model_type = 'GradientBoosting'  # Ensure we have a value
        
        model_params = config_data['hyperparameters'].get(model_type, [])
        if not model_params:
            # Try with capitalized first letter
            model_type_cap = model_type.capitalize()
            model_params = config_data['hyperparameters'].get(model_type_cap, [])
            if not model_params:
                # Try with lowercase
                model_type_lower = model_type.lower()
                model_params = config_data['hyperparameters'].get(model_type_lower, [])
                if not model_params:
                    raise ValueError(f"No hyperparameters defined for model type: {self.model_type}")
        
        space_definitions = []
        skopt_dimensions = []
        
        for param_spec in model_params:
            if not isinstance(param_spec, dict) or 'name' not in param_spec or 'type' not in param_spec:
                raise ValueError("Invalid parameter specification")
                
            name = param_spec['name']
            param_type = param_spec['type'].lower()
            
            # Map grid-style types to Bayesian-compatible types
            if param_type == 'float': param_type = 'real'
            if param_type == 'int': param_type = 'integer'
            
            try:
                if param_type == 'integer':
                    # Get values from list if no explicit min/max
                    if 'values' in param_spec:
                        vals = [v for v in param_spec['values'] if v is not None]  # Filter out None
                        if not vals:
                            raise ValueError(f"Parameter {name} has no valid integer values after removing None.")
                        param_min, param_max = min(vals), max(vals)
                    else:
                        param_min = param_spec['min']
                        param_max = param_spec['max']
                    skopt_dimensions.append(Integer(param_min, param_max, name=name))
                    
                elif param_type == 'real':
                    # Handle log-uniform prior for learning rates
                    prior = param_spec.get('prior', 'uniform').lower()
                    if 'values' in param_spec:
                        vals = param_spec['values']
                        param_min, param_max = min(vals), max(vals)
                    else:
                        param_min = param_spec['min']
                        param_max = param_spec['max']
                    skopt_dimensions.append(Real(param_min, param_max, prior=prior, name=name))
                    
                elif param_type == 'categorical':
                    # Directly use values list as categories
                    choices = param_spec['values']
                    skopt_dimensions.append(Categorical(choices, name=name))
                    
                else:
                    raise ValueError(f"Unsupported type: {param_type}")
                    
                space_definitions.append(param_spec)
                
            except Exception as e:
                raise ValueError(f"Error processing {name}: {str(e)}")
        
        return space_definitions, skopt_dimensions

    def _log_iteration_progress(self, params: Dict[str, Any], score: float, iteration: int) -> None:
        """
        Record the parameters and score for the current optimization iteration.
        Updates the best score and parameters if the current score is an improvement.
        (Assumes minimization, so a lower score is better).
        """
        self.optimization_history.append({
            'iteration': iteration,
            'parameters': params,
            'score': score  # This is the value returned by evaluation_function (to be minimized)
        })
        
        # gp_minimize minimizes the function, so lower score is better.
        if score < self.best_score_so_far:
            self.best_score_so_far = score
            self.best_params_so_far = params
            logger.info(f"Iteration {iteration}: New best score: {score:.6f} with params: {params}")
        else:
            logger.info(f"Iteration {iteration}: Score: {score:.6f} with params: {params}")

    def run_search(self) -> Tuple[Optional[Dict[str, Any]], float, OptimizeResult]:
        """
        Execute the Bayesian optimization process.

        Returns:
            Tuple[Optional[Dict[str, Any]], float, OptimizeResult]:
                - best_parameters (Dict): The best set of hyperparameters found.
                - best_score (float): The score achieved by the best parameters.
                                      (This is the minimized value from the objective function).
                - result (OptimizeResult): The full result object from `skopt.gp_minimize`.
        """
        logger.info(f"Starting Bayesian optimization: {self.n_calls} total calls, "
                    f"{self.n_initial_points} initial random points.")
        
        iteration_count = 0

        # The objective function to be minimized by gp_minimize
        @use_named_args(self.dimensions)
        def objective_function(**params: Any) -> float:
            nonlocal iteration_count
            iteration_count += 1
            
            # Ensure correct types from skopt (e.g., numpy int to python int)
            processed_params = {name: params[name] for name in self.param_names}
            
            current_score = self.evaluation_function(processed_params)
            
            if not isinstance(current_score, (int, float)) or np.isnan(current_score) or np.isinf(current_score):
                logger.warning(f"Evaluation function for params {processed_params} returned non-finite "
                               f"score: {current_score}. Assigning a high penalty.")
                # Assign a high penalty for bad evaluations to guide optimizer away
                current_score = float(np.finfo(np.float64).max / (self.n_calls * 10)) # Large but finite penalty
            
            self._log_iteration_progress(processed_params, current_score, iteration_count)
            
            # gp_minimize aims to minimize this returned value
            return current_score 

        # Execute the Gaussian Process minimization
        try:
            result: OptimizeResult = gp_minimize(
                func=objective_function,
                dimensions=self.dimensions,
                n_calls=self.n_calls,
                n_initial_points=self.n_initial_points,
                random_state=self.random_state,
                verbose=False # We handle logging per iteration
            )
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}", exc_info=True)
            # In case of failure, we might not have a proper result.
            # Return current best if any, or None.
            self._save_results_to_file(self.best_params_so_far, self.best_score_so_far, None)
            self._plot_optimization_progress()
            return self.best_params_so_far, self.best_score_so_far, None


        # The best parameters are in result.x, and the best score (minimized value) is result.fun
        final_best_params = dict(zip(self.param_names, result.x))
        final_best_score = float(result.fun) # This is the minimized value.
        
        logger.info(f"Bayesian optimization completed.")
        logger.info(f"Best parameters found: {final_best_params}")
        logger.info(f"Best score (minimized objective): {final_best_score:.6f}")
        
        self._save_results_to_file(final_best_params, final_best_score, result)
        self._plot_optimization_progress()
        
        return final_best_params, final_best_score, result

    def _save_results_to_file(self, best_params, best_score, skopt_result):
        """Save comprehensive optimization results to JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_summary = {
            'search_configuration': {
                'model_type': self.model_type,
                'n_calls': self.n_calls,
                'n_initial_points': self.n_initial_points,
                'random_state': self.random_state,
                'search_space_definition': self.search_space_config
            },
            'best_result_found': {
                'parameters': best_params,
                'minimized_score': best_score # Explicitly state it's the minimized score
            },
            'optimization_history': self.optimization_history
        }
        summary_file_path = self.output_dir / f"bayesian_search_summary_{self.model_type}_{timestamp}.json"
        
        # If skopt_result is available, add more details if serializable
        if skopt_result:
            # Storing the entire OptimizeResult can be large and complex.
            # Extract key, serializable information.
            output_summary['skopt_summary'] = {
                'x_iters': skopt_result.x_iters, # List of parameter points evaluated
                'func_vals': skopt_result.func_vals.tolist(), # List of objective values
                'space_details': str(skopt_result.space) # String representation of the space
            }

        # Ensure all parts of the output are JSON serializable
        serializable_output = self._make_dict_json_serializable(output_summary)
        summary_file_path = self.output_dir / f"bayesian_search_summary_{self.model_type}.json"
        try:
            with open(summary_file_path, 'w') as f:
                json.dump(serializable_output, f, indent=2)
            logger.info(f"Optimization summary saved to: {summary_file_path.resolve()}")
        except Exception as e:
            logger.error(f"Failed to save summary to {summary_file_path.resolve()}: {e}", exc_info=True)

    def _plot_optimization_progress(self) -> None:
        """Visualize the scores obtained at each iteration of the optimization."""
        if not self.optimization_history:
            logger.info("No optimization history to plot.")
            return

        iterations = [item['iteration'] for item in self.optimization_history]
        scores = [item['score'] for item in self.optimization_history] # These are the raw scores (to be minimized)

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, scores, marker='o', linestyle='-', color='#2c7bb6', markersize=5, label="Objective Score per Iteration")
        
        # Plot a running minimum to show convergence
        running_min_scores = np.minimum.accumulate(scores)
        plt.plot(iterations, running_min_scores, marker='.', linestyle='--', color='#d7191c', label="Best Score So Far")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file_path = self.output_dir / f"bayesian_optimization_progress_{self.model_type}_{timestamp}.png"
        plt.title(f"Bayesian Optimization Progress ({self.model_type})\nObjective: Minimize Score")
        plt.xlabel("Iteration Number")
        plt.ylabel("Objective Function Score")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        try:
            plt.savefig(plot_file_path)
            logger.info(f"Optimization progress plot saved to: {plot_file_path.resolve()}")
        except Exception as e:
            logger.error(f"Failed to save plot to {plot_file_path.resolve()}: {e}", exc_info=True)
        finally:
            plt.close() # Close the figure to free memory

    def _convert_numpy_types(self, obj: Any) -> Any:
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def _make_dict_json_serializable(self, data: Any) -> Dict:
        """
        Recursively process data to convert numpy types to native Python types.
        Handles dictionaries, lists, and nested structures.
        """
        if isinstance(data, dict):
            return {key: self._make_dict_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._make_dict_json_serializable(item) for item in data]
        else:
            return self._convert_numpy_types(data)

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Bayesian Search ===\n")
    evaluation_function=None
    output_dir_name = "bayesian_search"
    
    search1 = BayesianSearch(evaluation_function)

    print(f"{search1}")

    print(f"\n* * * * * Phase 2 * * * * *\n")
    def evaluation(params: Dict) -> float:
        print(f"Evaluating parameters: {params}")
        return np.random.rand()  # Example score
    
    search2 = BayesianSearch(evaluation)
    best_params, best_score, _ = search2.run_search()

    print(f"Best params: {best_params}")
    print(f"Best score: {best_score:.4f}")
    print(f"\n* * * * * Phase 3 * * * * *\n")

    print("\n=== Successfully Ran Bayesian Search ===\n")
