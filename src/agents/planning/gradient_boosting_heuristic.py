"""
Gradient Boosting Heuristic for Task Planning

This module provides a machine learning–based heuristic using a Gradient Boosting Classifier to guide task planning decisions.
It analyzes features like task depth, goal overlap, failure history, resource availability, and priority to predict the success
probability of a task-method pair.

Real-World Use Case:
In robotics or autonomous systems, this heuristic helps choose the most promising method to achieve a goal (e.g., navigating or manipulating objects)
by learning from historical execution data. For instance, if a robot must navigate to a location, the heuristic can predict which
algorithm (e.g., A* or RRT) has the highest chance of success under current conditions like battery level or system load.
"""

import numpy as np
import yaml, json
import traceback
import joblib
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import List, Any, Dict, Tuple

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from src.agents.planning.utils.base_heuristic import BaseHeuristics
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Gradient Boosting Heuristic")
printer = PrettyPrinter

class DotDict(dict):
    def __getattr__(self, item):
        return self.get(item)

class GradientBoostingHeuristic(BaseHeuristics):
    def __init__(self):
        self.config = load_global_config()

        self.heuristics_config = get_config_section('global_heuristic')
        self.max_depth = self.heuristics_config.get('max_depth')
        self.min_samples_split = self.heuristics_config.get('min_samples_split')
        self.class_weight = self.heuristics_config.get('class_weight')
        self.trained = self.heuristics_config.get('trained')
        self.random_state = self.heuristics_config.get('random_state')
        self.planning_db_path = self.heuristics_config.get('planning_db_path')
        self.heuristic_model_path = self.heuristics_config.get('heuristic_model_path')
        os.makedirs(self.heuristic_model_path, exist_ok=True)

        self.gbh_config = get_config_section('gradient_boosting_heuristic')
        self.n_estimators = self.gbh_config.get('n_estimators')
        self.learning_rate = self.gbh_config.get('learning_rate')
        self.subsample = self.gbh_config.get('subsample')
        self.early_stopping_rounds = self.gbh_config.get('early_stopping_rounds')
        self.validation_fraction = self.gbh_config.get('validation_fraction')
        self.feature_config = self.gbh_config.get('feature_config', {
            'use_priority', 'use_resource_check', 'recent_success_window'
        })

        self.agent = {}
        self.model = self._init_model()
        self.scaler = StandardScaler(with_mean=False, with_std=False)
        self.feature_names = self._get_feature_names()
        self.feature_importances_ = None

        logger.info(f"Gradient Boosting Heuristic succesfully initialized")

    def _init_model(self):
        params = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "min_samples_split": self.min_samples_split,
            "n_iter_no_change": self.early_stopping_rounds,
            "validation_fraction": self.validation_fraction,
            "random_state": self.random_state,
        }
        if self.early_stopping_rounds and self.validation_fraction > 0:
            params["n_iter_no_change"] = self.early_stopping_rounds
            params["validation_fraction"] = self.validation_fraction
        return GradientBoostingClassifier(**params)

    def _get_feature_names(self) -> List[str]:
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
        return base_features

    def predict_success_prob(
        self,
        task: Dict[str, Any],
        world_state: Dict[str, Any],
        method_stats: Dict[Tuple[str, str], Dict[str, int]],
        method_id: str
    ) -> float:
        printer.status("INIT", "Success prob predictor succesfully initialized", "info")
        original_method = task.get("selected_method")
        task["selected_method"] = method_id
        
        try:
            if not self.trained:
                model_path = os.path.join(self.heuristic_model_path, 'gb_heuristic_model.pkl')
                if os.path.exists(model_path):
                    self.model, self.scaler = joblib.load(model_path)
                    self.trained = True
                else:
                    logger.warning("Using untrained model with default predictions")
                    return 0.5
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return 0.5  # Fallback to neutral probability

        # Call existing implementation
        prob = self._predict_success_prob(task, world_state, method_stats)

        # Restore original method
        if original_method is not None:
            task["selected_method"] = original_method
        else:
            del task["selected_method"]

        return prob

    def _predict_success_prob(self, task, world_state, method_stats) -> float:
        features = self.extract_features(task, world_state, method_stats)
        scaled_features = self.scaler.transform(features.reshape(1, -1))
        return self.model.predict_proba(scaled_features)[0][1]

    def extract_features(self, task, world_state, method_stats) -> np.ndarray:
        printer.status("INIT", "Freature extractor succesfully initialized", "info")

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
                features[feature_idx] = task.priority if hasattr(task, 'priority') else 0.5
                feature_idx += 1
                
            if self.feature_config.get("use_resource_check"):
                features[feature_idx] = world_state.get('cpu_available', 0)
                feature_idx += 1
                features[feature_idx] = world_state.get('memory_available', 0)
                feature_idx += 1
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}\n{traceback.format_exc()}")
            return np.zeros(len(self.feature_names))

        return features.astype(np.float32)

    def load_training_data(self):
        """Load and preprocess historical execution data"""
        printer.status("INIT", "Training data loaded", "info")

        tasks, _, world_states = self.load_planning_db()
        X = []
        y = []
        
        for task, state in zip(tasks, world_states):
            # Extract features
            features = self.extract_features(task, state, {})
            X.append(features)
            
            # Extract success label (1=success, 0=failure)
            outcome = task.get("outcome", 0)
            y.append(1 if outcome == "success" else 0)
        
        return np.array(X), np.array(y)

    def load_planning_db(self, path: str = None):
        """Load task, world state, and method statistics from the planning database."""
        printer.status("INIT", "Planning db succesfully initialized", "info")

        if path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Use instance attribute instead of undefined variable
            path = os.path.join(current_dir, self.planning_db_path)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    
        tasks = data["tasks"]
        method_stats = {
            tuple(k.split(":")): v for k, v in data["method_stats"].items()
        }
        world_states = data["world_states"]
    
        return tasks, method_stats, world_states

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))

        # Check if we have at least 2 classes
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logger.error("Cannot train with only one class in dataset")
            return 0.0

        if any(c < 2 for c in counts):
            logger.warning("Insufficient class distribution for stratified split. Proceeding without stratification.")
            stratify = None
        else:
            stratify = y

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, stratify=stratify
        )

        # Check training set has at least 2 classes
        if len(np.unique(y_train)) < 2:
            logger.warning("Training set has only one class - skipping training")
            return 0.0

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # If too few training samples per class, disable early stopping and remove validation_fraction
        if any(c < 2 for c in np.unique(y_train, return_counts=True)[1]):
            logger.warning("Disabling early stopping due to insufficient samples for internal validation.")
            self.model.set_params(n_iter_no_change=None)
            if 'validation_fraction' in self.model.get_params():
                self.model.set_params(validation_fraction=0.1)  # default safe value

        self.model.fit(X_train_scaled, y_train)

        # Track feature importances
        self.feature_importances_ = self.model.feature_importances_

        # Evaluate validation performance
        val_pred = self.model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_pred)
        logger.info(f"Validation accuracy: {val_acc:.2f}")

        self.trained = True
        model_path = os.path.join(self.heuristic_model_path, 'gb_heuristic_model.pkl')
        joblib.dump((self.model, self.scaler), model_path)
        return val_acc

    def select_best_method(self, task, world_state, candidate_methods, method_stats=None):
        """Select method with highest predicted success probability"""
        printer.status("INIT", "Method selecter succesfully initialized", "info")
    
        best_method = None
        best_prob = -1
        
        for method_id in candidate_methods:
            prob = self.predict_success_prob(task, world_state, {}, method_id)
            if prob > best_prob:
                best_prob = prob
                best_method = method_id
                
        return best_method, best_prob

    def update_model(self, task, world_state, outcome):
        """Update model with new execution result"""
        printer.status("INIT", "Model update succesfully initialized", "info")

        if not self.trained:
            return
            
        # Convert outcome to label
        label = 1 if outcome == "success" else 0
        
        # Extract features
        features = self.extract_features(task, world_state, {})
        
        # Partial model update
        # self.model.partial_fit(features.reshape(1, -1), [label])

        # Append to training database
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.planning_db_path)
        with open(db_path, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data["tasks"].append({**task, "outcome": outcome})
            data["world_states"].append(world_state)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

        # Retrain the model using the updated data
        X, y = self.load_training_data()
        self.train(X, y)

        # Update feature importances
        self.feature_importances_ = self.model.feature_importances_
        logger.info("Model updated with new execution data")

    def report_feature_importance(self):
        """Return feature importance analysis"""
        return sorted(
            zip(self.feature_names, self.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )
    
if __name__ == "__main__":
    print("\n=== Running Gradient Boosting Heuristic Test ===\n")
    printer.status("Init", "Gradient Boosting Heuristic initialized", "success")

    planner02 = GradientBoostingHeuristic()
    print(planner02)
    print("\n* * * * * Phase 2 - Extracted Features * * * * *\n")
    task = {
        "name": "navigate_to_room",
        "selected_method": 0,
        "priority": 0.8,
        "goal_state": {"location": "room_1"},
        "parent": {
            "name": "root_task",
            "parent": None  # Ensures depth = 2
        }
    }
    state = {
        "location": "room_1",
        "cpu_available": 0.7,
        "memory_available": 0.6,
        "battery_level": 0.9
    }
    stats = {
        ("navigate_to_room", 0): {"success": 8, "total": 10}
    }
    id = 784545

    features = planner02.extract_features(task=task, world_state=state, method_stats=stats)
    printer.pretty("Extracted Features:", features, "Success")
    print("\n* * * * * Phase 3 - Train * * * * *\n")
    X, y = planner02.load_training_data()
    unique, counts = np.unique(y, return_counts=True)
    printer.pretty(f"Class distribution: {dict(zip(unique, counts))}", "", "Info")
    
    if len(unique) < 2:
        printer.pretty("Skipping training - need at least 2 classes", "", "Warning")
        val_acc = 0.0
    else:
        val_acc = planner02.train(X, y)
        printer.pretty(f"Validation Accuracy: {val_acc:.2f}", "", "Success")

    print("\n* * * * * Phase 4 - Method Selection * * * * *\n")
    task = {
        "name": "navigation",
        "priority": 0.8,
        "goal_state": {"position": "target"},
        "parent": None
    }
    state = {"position": "start", "cpu_available": 70, "memory_available": 2048}
    candidate_methods = ["A*", "RRT"]
    
    best_method, confidence = planner02.select_best_method(task, state, candidate_methods)
    printer.pretty(f"Recommended method: {best_method} (confidence: {confidence:.2f})", "", "Success")
    
    print("\n* * * * * Phase 5 - Continuous Learning * * * * *\n")
    # Simulate successful execution
    planner02.update_model(task, state, "success")
    printer.pretty("Model updated with new execution data", "", "Success")
    print("\n=== Successfully Ran Gradient Boosting Heuristic ===\n")
