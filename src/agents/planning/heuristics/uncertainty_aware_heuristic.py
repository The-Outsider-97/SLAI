"""
Uncertainty-Aware Heuristic for Risk-Aware Task Planning

This module implements a Bayesian machine learning–based heuristic that not only predicts 
the success probability of task-method pairs but also quantifies uncertainty in those predictions. 
It enables risk-sensitive planning by using probabilistic models (e.g., Gaussian Processes, Bayesian Neural Nets) 
to decide *when* a prediction is confident enough to act upon.

By expressing both prediction and confidence, this heuristic helps agents avoid over-committing to uncertain actions, 
making it especially valuable in safety-critical or information-sparse domains.

Real-World Use Case:
- In healthcare robotics, it helps decide whether the confidence in a recommended action (e.g., assistive maneuver) 
  is sufficient, otherwise escalating to human supervision.
- In autonomous exploration (e.g., drones, planetary rovers), it delays decisions or gathers more information 
  when prediction confidence is low, reducing mission risk.
- In any adaptive planning system, it allows fallback strategies or data gathering when uncertainty is too high.
"""

import os
import json
import threading
import numpy as np
import joblib

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from typing import List, Any, Dict, Tuple, Optional

import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.base_heuristic import BaseHeuristics
from ....tuning.utils.bayesian_neural_network import BayesianNeuralNetwork
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Uncertainty-Aware Heuristic")
printer = PrettyPrinter

class DotDict(dict):
    def __getattr__(self, item):
        return self.get(item)

class UncertaintyAwareHeuristic(BaseHeuristics):
    def __init__(self):
        self.config = load_global_config()
        self.heuristics_config = get_config_section('global_heuristic')
        self.ua_config = get_config_section('uncertainty_aware_heuristic')

        self.model_type = self.ua_config.get('model_type', 'GP')
        self.uncertainty_threshold = self.ua_config.get('uncertainty_threshold', 0.15)
        self.feature_config = self.ua_config.get('feature_config', {})
        self.bnn_config = self.ua_config.get('bnn_config', {})

        # Paths
        self.planning_db_path = self.heuristics_config.get('planning_db_path')
        self.heuristic_model_path = self.heuristics_config.get('heuristic_model_path')
        self.random_state = self.heuristics_config.get('random_state', 42)
        os.makedirs(self.heuristic_model_path, exist_ok=True)
        self.model_path = os.path.join(self.heuristic_model_path, 'uah_model.pkl')

        self.feature_names = self._get_feature_names()
        self.scaler = StandardScaler()
        self.model = None
        self.trained = False
        self._lock = threading.RLock()

        self._load_model()

    def _get_feature_names(self) -> List[str]:
        base = ['task_depth', 'goal_overlap', 'method_failure_rate', 'state_diversity']
        if self.feature_config.get("use_priority"):
            base.append('task_priority')
        if self.feature_config.get("use_resource_check"):
            base.extend(['cpu_available', 'memory_available'])
        if self.feature_config.get("use_temporal_features"):
            base.extend(['time_since_creation', 'deadline_proximity'])
        return base

    def _load_model(self):
        with self._lock:
            if os.path.exists(self.model_path):
                try:
                    data = joblib.load(self.model_path)
                    self.scaler = data['scaler']
                    self.feature_names = data['feature_names']
                    self.model_type = data['model_type']
                    self.uncertainty_threshold = data['uncertainty_threshold']
                    if self.model_type == 'GP':
                        self.model = data['gp_model']
                    elif self.model_type == 'BNN':
                        bnn_params = data['bnn_params']
                        self.model = BayesianNeuralNetwork(
                            bnn_params['layer_sizes'],
                            bnn_params['learning_rate'],
                            random_state=self.random_state
                        )
                        self.model.weights_mu = [np.array(w) for w in bnn_params['weights_mu']]
                        self.model.weights_logvar = [np.array(w) for w in bnn_params['weights_logvar']]
                        self.model.biases_mu = [np.array(b) for b in bnn_params['biases_mu']]
                        self.model.biases_logvar = [np.array(b) for b in bnn_params['biases_logvar']]
                    self.trained = True
                    logger.info("Loaded pre-trained Uncertainty-Aware model")
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")

    def _init_bnn(self):
        """Initialize Bayesian Neural Network with config parameters"""
        if self.bnn is not None:
            return
            
        # Get layer sizes from config or use default
        layer_sizes = self.bnn_config.get(
            'layer_sizes', 
            [len(self.feature_names), 64, 32, 1]  # Input, hidden1, hidden2, output
        )
        
        learning_rate = self.bnn_config.get('learning_rate', 0.01)
        
        self.bnn = BayesianNeuralNetwork(layer_sizes, learning_rate)
        logger.info(f"Initialized BNN with layers: {layer_sizes}")

    def _init_model(self):
        if self.model_type == 'GP':
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            return GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-3,
                n_restarts_optimizer=5,
                random_state=self.random_state,
                normalize_y=True
            )
        elif self.model_type == 'BNN':
            layer_sizes = self.bnn_config.get(
                'layer_sizes',
                [len(self.feature_names), 64, 32, 1]
            )
            learning_rate = self.bnn_config.get('learning_rate', 0.01)
            return BayesianNeuralNetwork(layer_sizes, learning_rate, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _ensure_model_ready(self) -> bool:
        if self.trained:
            return True
        with self._lock:
            if self.trained:
                return True
            try:
                X, y = self.load_training_data()
                if len(X) < 10:
                    logger.warning("Insufficient training data, using fallback.")
                    return False
                if len(X) > 0 and len(np.unique(y)) >= 2:
                    self.train(X, y)
                    return self.trained
            except Exception as e:
                logger.error(f"Auto‑training failed: {e}")
        return False

    def predict_with_uncertainty(
        self,
        task: Dict[str, Any],
        world_state: Dict[str, Any],
        method_stats: Dict,
        method_id: str
    ) -> Tuple[float, float]:
        if not self._ensure_model_ready():
            return 0.5, 0.5

        original_method = task.get("selected_method")
        task["selected_method"] = method_id
        try:
            features = self.extract_features(task, world_state, method_stats)
            scaled = self.scaler.transform(features.reshape(1, -1))

            if self.model_type == 'GP':
                pred, std = self.model.predict(scaled, return_std=True)
                prob = np.clip(pred[0], 0, 1)
                uncertainty = std[0]
            elif self.model_type == 'BNN':
                mean, std = self.model.predict(scaled, num_samples=100)
                prob = np.clip(sigmoid(mean[0, 0]), 0, 1)   # apply sigmoid
                uncertainty = std[0, 0]                     # keep raw uncertainty
            else:
                prob, uncertainty = 0.5, 0.5
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            prob, uncertainty = 0.5, 0.5
        finally:
            if original_method is not None:
                task["selected_method"] = original_method
            else:
                task.pop("selected_method", None)
        return prob, uncertainty

    def predict_success_prob(
        self,
        task: Dict[str, Any],
        world_state: Dict[str, Any],
        method_stats: Dict[Tuple[str, str], Dict[str, int]],
        method_id: str
    ) -> float:
        prob, _ = self.predict_with_uncertainty(task, world_state, method_stats, method_id)
        return prob

    def is_confidence_sufficient(self, uncertainty: float) -> bool:
        """Determine if prediction confidence meets required threshold"""
        printer.status("UAH", "determine prediction confidence", "info")

        return uncertainty <= self.uncertainty_threshold

    def select_best_method(
        self,
        task: Dict[str, Any],
        world_state: Dict[str, Any],
        candidate_methods: List[str],
        method_stats: Dict[Tuple[str, str], Dict[str, int]]
    ) -> Tuple[Optional[str], float]:
        best_method = None
        best_prob = -1.0
        best_uncertainty = float('inf')
        confident = []

        for method in candidate_methods:
            prob, unc = self.predict_with_uncertainty(task, world_state, method_stats, method)
            if prob > best_prob:
                best_prob = prob
                best_method = method
                best_uncertainty = unc
            if unc <= self.uncertainty_threshold:
                confident.append((prob, unc, method))

        if confident:
            confident.sort(key=lambda x: (-x[0], x[1]))
            best_method, best_prob, _ = confident[0]
        elif best_method:
            logger.warning(f"Using uncertain method {best_method} (unc={best_uncertainty:.2f})")
        return best_method, best_prob

    def extract_features(self, task, world_state, method_stats) -> np.ndarray:
        # Similar to DecisionTreeHeuristic but with method_id passed
        method_id = task.get("selected_method", "unknown")
        features = np.zeros(len(self.feature_names))
        idx = 0

        features[idx] = self._calculate_task_depth(task)
        idx += 1
        features[idx] = self._calculate_goal_overlap(task, world_state)
        idx += 1
        features[idx] = self._calculate_method_failure_rate(task, method_stats, method_id)
        idx += 1
        features[idx] = self._calculate_state_diversity(world_state)
        idx += 1

        if self.feature_config.get("use_priority"):
            features[idx] = task.get("priority", 0.5)
            idx += 1

        if self.feature_config.get("use_resource_check"):
            features[idx] = world_state.get("cpu_available", 0.0)
            idx += 1
            features[idx] = world_state.get("memory_available", 0.0)
            idx += 1

        if self.feature_config.get("use_temporal_features"):
            features[idx] = self._time_since_creation(task)
            idx += 1
            features[idx] = self._deadline_proximity(task)
            idx += 1

        return features

    def train(self, X: np.ndarray, y: np.ndarray):
        with self._lock:
            if len(np.unique(y)) < 2:
                logger.error("Need at least two classes for training")
                return

            X_scaled = self.scaler.fit_transform(X)

            if self.model_type == 'GP':
                kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
                self.model = GaussianProcessRegressor(
                    kernel=kernel, alpha=1e-3, n_restarts_optimizer=5,
                    random_state=self.heuristics_config.get('random_state', 42),
                    normalize_y=True
                )
                self.model.fit(X_scaled, y)
            elif self.model_type == 'BNN':
                # BNN training
                layer_sizes = self.bnn_config.get('layer_sizes', [len(self.feature_names), 64, 32, 1])
                learning_rate = self.bnn_config.get('learning_rate', 0.001)
                self.model = BayesianNeuralNetwork(layer_sizes, learning_rate)
                # Training loop (simplified)
                epochs = self.bnn_config.get('epochs', 20)
                batch_size = self.bnn_config.get('batch_size', 32)
                n_samples = X_scaled.shape[0]
                n_batches = int(np.ceil(n_samples / batch_size))
                for epoch in range(epochs):
                    indices = np.random.permutation(n_samples)
                    total_elbo = 0
                    for batch_idx in range(n_batches):
                        start = batch_idx * batch_size
                        end = min((batch_idx+1)*batch_size, n_samples)
                        batch_idx = indices[start:end]
                        x_batch = X_scaled[batch_idx]
                        y_batch = y[batch_idx].reshape(-1, 1)
                        elbo, _ = self.model.train_step(x_batch, y_batch, num_samples=5)
                        total_elbo += elbo
                    if epoch % 10 == 0:
                        logger.info(f"Epoch {epoch}: ELBO {total_elbo/n_batches:.4f}")

            self.trained = True
            self._save_model()

    def _save_model(self):
        with self._lock:
            data = {
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'uncertainty_threshold': self.uncertainty_threshold,
                'trained': self.trained,
            }
            if self.model_type == 'GP':
                data['gp_model'] = self.model
            elif self.model_type == 'BNN':
                data['bnn_params'] = {
                    'layer_sizes': self.model.layer_sizes,
                    'learning_rate': self.model.learning_rate,
                    'weights_mu': [w.tolist() for w in self.model.weights_mu],
                    'weights_logvar': [w.tolist() for w in self.model.weights_logvar],
                    'biases_mu': [b.tolist() for b in self.model.biases_mu],
                    'biases_logvar': [b.tolist() for b in self.model.biases_logvar],
                }
            joblib.dump(data, self.model_path)
            logger.info("Uncertainty-Aware model saved")

    def load_training_data(self):
        """Load historical execution data for training"""
        printer.status("UAH", "Loading training data", "info")

        tasks, method_stats, world_states = self.load_planning_db()
        X = []
        y = []
        for task, state in zip(tasks, world_states):
            method_id = task.get("selected_method")
            if method_id is None:
                continue
            # Temporarily set method for feature extraction
            task["selected_method"] = method_id
            features = self.extract_features(task, state, method_stats)
            X.append(features)
            outcome = task.get("outcome", "failure")
            y.append(1 if outcome == "success" else 0)
        return np.array(X), np.array(y)

    def load_planning_db(self, path: str = None):
        """Load task, world state, and method statistics from the planning database."""
        if path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base_dir, self.planning_db_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tasks = data["tasks"]
        method_stats = {tuple(k.split(":")): v for k, v in data["method_stats"].items()}
        world_states = data["world_states"]
        return tasks, method_stats, world_states

    def load_model(self):
        """Load model and scaler from disk"""
        printer.status("UAH", "Loading model", "info")

        try:
            model_data = joblib.load(self.model_path)
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.trained = model_data['trained']
            self.model_type = model_data.get('model_type', 'GP')
            self.uncertainty_threshold = model_data.get('uncertainty_threshold', 0.15)
            
            # Load specific model type
            if self.model_type == 'GP':
                self.model = model_data['gp_model']
            elif self.model_type == 'BNN':
                # Reconstruct BNN from parameters
                bnn_params = model_data['bnn_params']
                self.bnn = BayesianNeuralNetwork(
                    bnn_params['layer_sizes'],
                    bnn_params['learning_rate']
                )
                self.bnn.weights_mu = [np.array(w) for w in bnn_params['weights_mu']]
                self.bnn.weights_logvar = [np.array(w) for w in bnn_params['weights_logvar']]
                self.bnn.biases_mu = [np.array(b) for b in bnn_params['biases_mu']]
                self.bnn.biases_logvar = [np.array(b) for b in bnn_params['biases_logvar']]
                self.model = self.bnn
            
            logger.info(f"Loaded pre-trained {self.model_type} model")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.trained = False

    def update_with_outcome(self, task, world_state, method_stats, method_id: str, success: bool):
        """Update model with new execution outcome"""
        printer.status("UAH", "Updating model", "info")
    
        if not self.trained or self.model is None:
            return
    
        # Backup original method
        original_method = task.get("selected_method")
        task["selected_method"] = method_id  # Inject method for feature context
    
        # Extract features for this instance
        features = self.extract_features(task, world_state, method_stats)
        features = features.reshape(1, -1)
    
        # Restore original method
        if original_method is not None:
            task["selected_method"] = original_method
        else:
            del task["selected_method"]
    
        # Scale features
        scaled_features = self.scaler.transform(features)
    
        # Convert outcome to numeric label
        label = 1 if success else 0
    
        # Update model based on type
        if self.model_type == 'GP':
            logger.info("New data available - scheduling model retraining")
        elif self.model_type == 'BNN':
            try:
                # Use self.model, not self.bnn
                self.model.train_step(
                    scaled_features,
                    np.array([[label]]),
                    num_samples=5
                )
                logger.info("BNN updated with new execution data")
            except Exception as e:
                logger.error(f"BNN update failed: {str(e)}")


if __name__ == "__main__":
    print("\n=== Running Uncertainty-Aware Heuristic Test ===\n")
    printer.status("Init", "Uncertainty-Aware Heuristic initialized", "success")
    from datetime import timedelta, datetime

    uah = UncertaintyAwareHeuristic()
    print(uah )
    print("\n* * * * * Phase 2 - Prediction with Uncertainty * * * * *\n")

    task = {
        "name": "navigate_to_room",
        "priority": 0.8,
        "goal_state": {"location": "room_1"},
        "parent": {"name": "root_task", "parent": None},
        "creation_time": datetime.now().isoformat(),
        "deadline": (datetime.now() + timedelta(hours=2)).isoformat(),
        "selected_method": "A*"
    }
    state = {
        "location": "corridor",
        "cpu_available": 0.7,
        "memory_available": 0.6,
        "battery_level": 0.9
    }
    stats = {
        ("navigate_to_room", "A*"): {"success": 8, "total": 10},
        ("navigate_to_room", "RRT"): {"success": 6, "total": 10}
    }
    methods = ["A*", "RRT", "D*"]

    for method in methods:
        prob, uncertainty = uah.predict_with_uncertainty(task, state, stats, method)
        confidence = "✓" if uah.is_confidence_sufficient(uncertainty) else "✗"
        printer.pretty(
            f"{method}: prob={prob:.2f}, unc={uncertainty:.2f} {confidence}", 
            "", 
            "info"
        )

    print("\n* * * * * Phase 3 - Method Selection * * * * *\n")
    best_method, best_prob = uah.select_best_method(task, state, methods, stats)
    printer.pretty(f"Selected method: {best_method} (prob: {best_prob:.2f})", "", "success")

    print("\n* * * * * Phase 4 - Model Update * * * * *\n")
    uah.update_with_outcome(task, state, stats, "A*", success=True)
    printer.pretty("Model updated with execution outcome", "", "success")

    print("\n=== Successfully Ran Uncertainty-Aware Heuristic ===\n")
