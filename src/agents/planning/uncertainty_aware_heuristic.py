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
import joblib
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from src.agents.planning.utils.base_heuristic import BaseHeuristics
from src.tuning.utils.bayesian_neural_network import BayesianNeuralNetwork
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
        self.trained = self.heuristics_config.get('trained')
        self.random_state = self.heuristics_config.get('random_state')
        self.planning_db_path = self.heuristics_config.get('planning_db_path')
        self.heuristic_model_path = self.heuristics_config.get('heuristic_model_path')
        os.makedirs(self.heuristic_model_path, exist_ok=True)
        self.model_path = os.path.join(self.heuristic_model_path, 'uah_model.pkl')

        self.uah_config = get_config_section('uncertainty_aware_heuristic')
        self.uncertainty_threshold = self.uah_config.get('uncertainty_threshold')
        self.model_type = self.uah_config.get('model_type')
        self.feature_config = self.uah_config.get('feature_config', {})
        self.bnn_config = self.uah_config.get('bnn_config', {})

        # Initialize model components
        self.model = None
        self.bnn = None
        self.scaler = StandardScaler()
        self.feature_names = self._get_feature_names()

        # Load pre-trained model if available
        if os.path.exists(self.model_path):
            self.load_model()
        elif self.model_type == 'BNN':
            self._init_bnn()

        logger.info(f"Uncertainty-Aware Heuristic initialized (Model: {self.model_type})")

    def _get_feature_names(self) -> list:
        """Define feature set based on configuration"""
        base_features = [
            'task_depth',
            'goal_overlap',
            'method_failure_rate',
            'state_diversity'
        ]
        if self.feature_config.get("use_priority"):
            base_features.append('task_priority')
        if self.feature_config.get("use_resource_check"):
            base_features.extend(['cpu_available', 'memory_available'])
        if self.feature_config.get("use_temporal_features"):
            base_features.extend(['time_since_creation', 'deadline_proximity'])
        return base_features

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
        """Initialize probabilistic model based on configuration"""
        if self.model_type == 'GP':
            # Gaussian Process with RBF kernel + noise term
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            return GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-3,
                n_restarts_optimizer=5,
                random_state=self.random_state,
                normalize_y=True
            )
        elif self.model_type == 'BNN':
            # Use the Bayesian Neural Network
            self._init_bnn()
            return self.bnn
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def predict_with_uncertainty(self, task, world_state, method_stats, method_id: str) -> tuple:
        """
        Predict success probability and uncertainty for a specific method
        
        Returns:
            Tuple of (success_probability, uncertainty_measure)
        """
        printer.status("UAH", "predicting uncerainty", "info")

        # Set target method for feature extraction
        original_method = task.get("selected_method")
        task["selected_method"] = method_id
        
        # Extract features
        features = self.extract_features(task, world_state, method_stats)
        
        # Restore original method
        if original_method:
            task["selected_method"] = original_method
        else:
            del task["selected_method"]
            
        # Return default if model not trained
        if not self.trained or self.model is None:
            default_prob = 0.5
            return default_prob, 0.5  # High uncertainty
            
        # Scale features
        scaled_features = self.scaler.transform(features.reshape(1, -1))
        
        # Handle different model types
        if self.model_type == 'GP':
            # Gaussian Process prediction
            pred, std = self.model.predict(scaled_features, return_std=True)
            prob = np.clip(pred[0], 0, 1)  # Ensure valid probability
            return prob, std[0]
        elif self.model_type == 'BNN':
            # BNN prediction with Monte Carlo sampling
            mean, std = self.bnn.predict(scaled_features, num_samples=100)
            prob = np.clip(mean[0, 0], 0, 1)  # Single output, clip to probability range
            return prob, std[0, 0]
        else:
            # Fallback for unknown model types
            return 0.5, 0.5

    def predict_success_prob(self, task, world_state, method_stats, method_id: str) -> float:
        """Predict success probability (without uncertainty)"""
        printer.status("UAH", "predicting success prob", "info")

        prob, _ = self.predict_with_uncertainty(task, world_state, method_stats, method_id)
        return prob

    def is_confidence_sufficient(self, uncertainty: float) -> bool:
        """Determine if prediction confidence meets required threshold"""
        printer.status("UAH", "determine prediction confidence", "info")

        return uncertainty <= self.uncertainty_threshold

    def select_best_method(self, task, world_state, candidate_methods, method_stats) -> tuple:
        """
        Select best method considering both success probability and uncertainty
        
        Returns:
            Tuple of (selected_method, success_probability)
        """
        printer.status("UAH", "Selecting best method", "info")

        best_method = None
        best_prob = -1
        best_uncertainty = float('inf')
        confident_candidates = []
        
        # Evaluate all candidate methods
        for method_id in candidate_methods:
            prob, uncertainty = self.predict_with_uncertainty(
                task, world_state, method_stats, method_id
            )
            
            # Track best candidate regardless of confidence
            if prob > best_prob:
                best_prob = prob
                best_method = method_id
                best_uncertainty = uncertainty
            
            # Collect confident candidates
            if self.is_confidence_sufficient(uncertainty):
                confident_candidates.append((prob, uncertainty, method_id))
        
        # Select from confident candidates if available
        if confident_candidates:
            # Sort by highest probability, then lowest uncertainty
            confident_candidates.sort(key=lambda x: (-x[0], x[1]))
            best_prob, _, best_method = confident_candidates[0]
            logger.info(f"Selected confident method: {best_method} (prob: {best_prob:.2f})")
        elif best_method:
            logger.warning(
                f"Using uncertain method {best_method} (prob: {best_prob:.2f}, "
                f"uncertainty: {best_uncertainty:.2f} > threshold {self.uncertainty_threshold})"
            )
        
        return best_method, best_prob

    def extract_features(self, task, world_state, method_stats) -> np.ndarray:
        """Extract feature vector for current state and method"""
        printer.status("UAH", "Extracting feature...", "info")

        try:
            features = np.zeros(len(self.feature_names))
            feature_idx = 0
            
            # Base features
            features[feature_idx] = self._calculate_task_depth(task)
            feature_idx += 1
            
            features[feature_idx] = self._calculate_goal_overlap(task, world_state)
            feature_idx += 1
            
            features[feature_idx] = self._calculate_method_failure_rate(task, method_stats)
            feature_idx += 1
            
            features[feature_idx] = self._calculate_state_diversity(world_state)
            feature_idx += 1

            # Optional features
            if self.feature_config.get("use_priority"):
                features[feature_idx] = task.get('priority', 0.5)
                feature_idx += 1
                
            if self.feature_config.get("use_resource_check"):
                features[feature_idx] = world_state.get('cpu_available', 0)
                feature_idx += 1
                features[feature_idx] = world_state.get('memory_available', 0)
                feature_idx += 1
                
            if self.feature_config.get("use_temporal_features"):
                features[feature_idx] = self._time_since_creation(task)
                feature_idx += 1
                features[feature_idx] = self._deadline_proximity(task)
                feature_idx += 1
                
            return features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return np.zeros(len(self.feature_names))

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the probabilistic model"""
        printer.status("UAH", "Training prob model", "info")

        if len(np.unique(y)) < 2:
            logger.error("Training requires both success and failure examples")
            return

        # Initialize model if needed
        if self.model is None:
            self.model = self._init_model()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Handle different model types
        if self.model_type == 'GP':
            # Train Gaussian Process
            self.model.fit(X_scaled, y)
            logger.info("Gaussian Process model trained")
        elif self.model_type == 'BNN':
            # Train Bayesian Neural Network
            y = y.reshape(-1, 1)  # Reshape for BNN output

            # Get training parameters from config
            epochs = self.bnn_config.get('epochs', 100)
            batch_size = self.bnn_config.get('batch_size', 32)
            mc_samples = self.bnn_config.get('mc_samples', 5)

            # Training loop
            n_samples = X_scaled.shape[0]
            n_batches = int(np.ceil(n_samples / batch_size))

            for epoch in range(epochs):
                # Shuffle data
                indices = np.random.permutation(n_samples)
                total_elbo = 0

                for batch_idx in range(n_batches):
                    start = batch_idx * batch_size
                    end = min((batch_idx + 1) * batch_size, n_samples)
                    batch_indices = indices[start:end]

                    x_batch = X_scaled[batch_indices]
                    y_batch = y[batch_indices]

                    # Perform training step
                    elbo, _ = self.bnn.train_step(x_batch, y_batch, num_samples=mc_samples)
                    total_elbo += elbo

                avg_elbo = total_elbo / n_batches
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{epochs} - ELBO: {avg_elbo:.4f}")

            logger.info("BNN training completed")

        self.trained = True
        self.save_model()

    def load_training_data(self):
        """Load historical execution data for training"""
        printer.status("UAH", "Loading training data", "info")

        tasks, method_stats, world_states = self.load_planning_db()
        X = []
        y = []
        
        for task, state in zip(tasks, world_states):
            # Skip tasks without outcome
            if "outcome" not in task:
                continue
                
            # Extract features for each method used
            for method_id in set(m["method"] for m in task.get("method_history", [])):
                task["selected_method"] = method_id
                features = self.extract_features(task, state, method_stats)
                X.append(features)
                
                # Determine label (success=1, failure=0)
                outcome = task.get("outcome")
                y.append(1 if outcome == "success" else 0)
        
        return np.array(X), np.array(y)

    def load_planning_db(self, path: str = None):
        """Load task, world state, and method statistics from planning database"""
        printer.status("UAH", "Loading planning db", "info")

        if path is None:
            path = self.planning_db_path
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        
            tasks = data["tasks"]
            method_stats = {
                tuple(k.split(":")): v for k, v in data["method_stats"].items()
            }
            world_states = data["world_states"]
            
            return tasks, method_stats, world_states
            
        except Exception as e:
            logger.error(f"Failed to load planning DB: {str(e)}")
            return [], {}, []

    def save_model(self):
        """Save model and scaler to disk"""
        printer.status("UAH", "Saving model", "info")

        if not self.trained or self.model is None:
            return
            
        model_data = {
            'model_type': self.model_type,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'trained': self.trained,
            'uncertainty_threshold': self.uncertainty_threshold
        }
        
        # Handle different model types
        if self.model_type == 'GP':
            model_data['gp_model'] = self.model
        elif self.model_type == 'BNN':
            # Save BNN parameters
            model_data['bnn_params'] = {
                'layer_sizes': self.bnn.layer_sizes,
                'learning_rate': self.bnn.learning_rate,
                'weights_mu': [w.tolist() for w in self.bnn.weights_mu],
                'weights_logvar': [w.tolist() for w in self.bnn.weights_logvar],
                'biases_mu': [b.tolist() for b in self.bnn.biases_mu],
                'biases_logvar': [b.tolist() for b in self.bnn.biases_logvar],
            }
        
        joblib.dump(model_data, self.model_path)
        logger.info(f"Model saved to {self.model_path}")

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
                self.bnn.train_step(
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
    from datetime import timedelta

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
