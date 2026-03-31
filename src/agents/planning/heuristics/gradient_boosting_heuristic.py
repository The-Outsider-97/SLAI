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

import os
import json
import threading
import numpy as np
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import List, Any, Dict, Tuple, Optional

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.base_heuristic import BaseHeuristics
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
        self.gb_config = get_config_section('gradient_boosting_heuristic')

        # Parameters
        self.n_estimators = self.gb_config.get('n_estimators', 200)
        self.learning_rate = self.gb_config.get('learning_rate', 0.05)
        self.subsample = self.gb_config.get('subsample', 0.8)
        self.early_stopping_rounds = self.gb_config.get('early_stopping_rounds', 20)
        self.validation_fraction = self.gb_config.get('validation_fraction', 0.2)
        self.max_depth = self.heuristics_config.get('max_depth', 8)
        self.min_samples_split = self.heuristics_config.get('min_samples_split', 15)
        self.random_state = self.heuristics_config.get('random_state', 42)

        # Paths
        self.planning_db_path = self.heuristics_config.get('planning_db_path')
        self.heuristic_model_path = self.heuristics_config.get('heuristic_model_path')
        os.makedirs(self.heuristic_model_path, exist_ok=True)
        self.model_path = os.path.join(self.heuristic_model_path, 'gb_heuristic_model.pkl')

        # Features
        self.feature_config = self.gb_config.get('feature_config', {})
        self.feature_names = self._get_feature_names()

        # Model
        self.model = None
        self.scaler = StandardScaler()
        self.trained = False
        self._lock = threading.RLock()

        self._load_model()

        logger.info(f"Gradient Boosting Heuristic succesfully initialized")

    def _get_feature_names(self) -> List[str]:
        base = ['task_depth', 'goal_overlap', 'method_failure_rate', 'state_diversity']
        if self.feature_config.get("use_priority"):
            base.append('task_priority')
        if self.feature_config.get("use_resource_check"):
            base.extend(['cpu_available', 'memory_available'])
        return base

    def _load_model(self):
        with self._lock:
            if os.path.exists(self.model_path):
                try:
                    self.model, self.scaler = joblib.load(self.model_path)
                    self.trained = True
                    logger.info("Loaded pre-trained Gradient Boosting model")
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")

    def _ensure_model_ready(self) -> bool:
        if self.trained:
            return True
        with self._lock:
            if self.trained:
                return True
            try:
                X, y = self.load_training_data()
                if len(X) > 0 and len(np.unique(y)) >= 2:
                    self.train(X, y)
                    return self.trained
            except Exception as e:
                logger.error(f"Auto‑training failed: {e}")
        return False

    def predict_success_prob(
        self,
        task: Dict[str, Any],
        world_state: Dict[str, Any],
        method_stats: Dict[Tuple[str, str], Dict[str, int]],
        method_id: str
    ) -> float:
        if not self._ensure_model_ready():
            return self._fallback_heuristic(task, world_state, method_stats, method_id)

        original_method = task.get("selected_method")
        task["selected_method"] = method_id
        try:
            features = self.extract_features(task, world_state, method_stats)
            scaled = self.scaler.transform(features.reshape(1, -1))
            prob = self.model.predict_proba(scaled)[0][1]
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            prob = 0.5
        finally:
            if original_method is not None:
                task["selected_method"] = original_method
            else:
                task.pop("selected_method", None)
        return prob

    def extract_features(self, task: Dict[str, Any], world_state: Dict[str, Any],
                         method_stats: Dict) -> np.ndarray:
        features = np.zeros(len(self.feature_names))
        idx = 0

        features[idx] = self._calculate_task_depth(task)
        idx += 1
        features[idx] = self._calculate_goal_overlap(task, world_state)
        idx += 1
        features[idx] = self._calculate_method_failure_rate(task, method_stats,
                                                            task.get("selected_method", "unknown"))
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

        return features

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        with self._lock:
            unique, counts = np.unique(y, return_counts=True)
            if len(unique) < 2:
                logger.error("Need at least two classes for training")
                return 0.0
    
            # Limit samples if configured
            max_samples = self.gb_config.get('validation', {}).get('max_training_samples', 10000)
            if len(X) > max_samples:
                indices = np.random.choice(len(X), max_samples, replace=False)
                X, y = X[indices], y[indices]
    
            # Determine if stratification is safe for our own split
            min_class_count = counts.min()
            stratify = y if min_class_count >= 2 else None
            if stratify is None and min_class_count < 2:
                logger.warning("Minority class has less than 2 samples; stratification disabled.")
    
            # Split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, stratify=stratify, random_state=self.random_state
            )
    
            # Fit scaler and transform
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
    
            # Build model with parameters
            model_params = {
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth,
                'subsample': self.subsample,
                'min_samples_split': self.min_samples_split,
                'random_state': self.random_state
            }
    
            # Determine if early stopping can be used safely
            # Early stopping requires at least 2 samples in each class in the internal validation set.
            # To be safe, we only enable early stopping if the minority class has at least 2 samples
            # in the *training* set (since the internal split will further split training set).
            use_early_stopping = (self.early_stopping_rounds and min_class_count >= 2)
            if use_early_stopping:
                model_params['n_iter_no_change'] = self.early_stopping_rounds
                model_params['validation_fraction'] = self.validation_fraction
            else:
                # No early stopping
                model_params['n_iter_no_change'] = None
    
            model = GradientBoostingClassifier(**model_params)
            model.fit(X_train_scaled, y_train)
    
            # Evaluate if validation set is non‑empty
            if len(y_val) > 0:
                val_acc = accuracy_score(y_val, model.predict(X_val_scaled))
                logger.info(f"Validation accuracy: {val_acc:.2f}")
            else:
                logger.warning("Validation set empty; skipping validation metrics.")
                val_acc = 0.0
    
            self.model = model
            self.trained = True
            joblib.dump((self.model, self.scaler), self.model_path)
            return val_acc

    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        tasks, _, world_states = self.load_planning_db()
        X = []
        y = []
        for task, state in zip(tasks, world_states):
            method_id = task.get("selected_method")
            if method_id is None:
                continue
            features = self.extract_features(task, state, {})
            X.append(features)
            y.append(1 if task.get("outcome") == "success" else 0)
        return np.array(X), np.array(y)

    def load_planning_db(self, path: str = None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), '..', self.planning_db_path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tasks = data["tasks"]
        world_states = data["world_states"]
        method_stats = {tuple(k.split(":")): v for k, v in data["method_stats"].items()}
        return tasks, method_stats, world_states

    def _fallback_heuristic(self, task, world_state, method_stats, method_id) -> float:
        key = (task.get("name"), method_id)
        stats = method_stats.get(key, {'success': 1, 'total': 2})
        if stats['total'] == 0:
            return 0.5
        return stats['success'] / stats['total']

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
        printer.status("INIT", "Model update successfully initialized", "info")
    
        if not self.trained:
            return
            
        # Compute the correct database path (same as in load_planning_db)
        if self.planning_db_path is None:
            logger.error("planning_db_path not set in config")
            return
            
        # Go up one level from heuristics/ to planning/
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(base_dir, self.planning_db_path)
        
        # Append to training database
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
