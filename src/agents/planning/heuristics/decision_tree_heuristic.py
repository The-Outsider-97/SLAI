"""
Decision Tree Heuristic for Task Planning

This module is a machine learning–based module that predicts the success probability of task-method pairs in the AI Planning Agent.
It computes features like task depth, goal overlap, failure rate, state diversity, and optional temporal/priority factors.
It uses a DecisionTreeClassifier to learn from historical task execution data.
During planning, it predicts the likelihood of success for different methods and helps choose the most promising one.

Real-World Use Case:
1. Robotics Task Selection: Choosing the best method to “pick up an object” (e.g., suction vs. gripper) based on current sensor readings, battery, and task urgency.
2. Cognitive Assistants: Selecting how to fulfill “schedule a meeting” — based on priority, deadlines, and system load — e.g., prioritize email invites vs. calendar syncing.
3. Game AI: Picking the most effective tactic (e.g., “ambush” vs. “defend base”) based on goal overlap and temporal urgency.
5. Disaster Response Simulation: Choosing navigation or search patterns under time constraints and environmental uncertainty.
"""

import os
import json
import threading
import numpy as np
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from typing import List, Any, Dict, Tuple, Optional

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.base_heuristic import BaseHeuristics
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Decision Tree Heuristic")
printer = PrettyPrinter

class DotDict(dict):
    def __getattr__(self, item):
        return self.get(item)

class DecisionTreeHeuristic(BaseHeuristics):
    def __init__(self):
        self.config = load_global_config()
        self.heuristics_config = get_config_section('global_heuristic')
        self.dt_config = get_config_section('decision_tree_heuristic')

        # Model parameters
        self.max_depth = self.heuristics_config.get('max_depth', 8)
        self.min_samples_split = self.heuristics_config.get('min_samples_split', 15)
        self.class_weight = self.heuristics_config.get('class_weight', 'balanced')
        self.random_state = self.heuristics_config.get('random_state', 42)
        self.min_impurity_decrease = self.dt_config.get('min_impurity_decrease', 0.01)
        self.ccp_alpha = self.dt_config.get('ccp_alpha', 0.02)

        # Paths
        self.planning_db_path = self.heuristics_config.get('planning_db_path')
        self.heuristic_model_path = self.heuristics_config.get('heuristic_model_path')
        os.makedirs(self.heuristic_model_path, exist_ok=True)
        self.model_path = os.path.join(self.heuristic_model_path, 'dt_heuristic_model.pkl')

        # Feature config
        self.feature_config = self.dt_config.get('feature_config', {})
        self.feature_names = self._get_feature_names()

        # Model state
        self.model = None
        self.scaler = StandardScaler()
        self.trained = False
        self._lock = threading.RLock()

        # Try to load existing model
        self._load_model()

        logger.info(f"Decision Tree Heuristic succesfully initialized")

    def _get_feature_names(self) -> List[str]:
        base = ['task_depth', 'goal_overlap', 'method_failure_rate', 'state_diversity']
        if self.feature_config.get("use_temporal_features"):
            base.extend(['time_since_creation', 'deadline_proximity'])
        if self.feature_config.get("use_priority_weighting"):
            base.append('priority_weight')
        return base

    def _load_model(self):
        with self._lock:
            if os.path.exists(self.model_path):
                try:
                    self.model, self.scaler = joblib.load(self.model_path)
                    self.trained = True
                    logger.info("Loaded pre-trained Decision Tree model")
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")

    def _init_model(self):
        params = {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_impurity_decrease": self.min_impurity_decrease,
            "ccp_alpha": self.ccp_alpha,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
        }
        return DecisionTreeClassifier(**params)

    def _ensure_model_ready(self) -> bool:
        if self.trained:
            return True
        with self._lock:
            # Try to load again in case another thread loaded it
            if self.trained:
                return True
            # Attempt auto‑training
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
            logger.warning("Model not ready, using fallback")
            return self._fallback_heuristic(task, world_state, method_stats, method_id)

        # Temporarily set method in task for feature extraction
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

    def select_best_method(
        self,
        task: Dict[str, Any],
        world_state: Dict[str, Any],
        candidate_methods: List[str],
        method_stats: Dict[Tuple[str, str], Dict[str, int]]
    ) -> Tuple[Optional[str], float]:
        best = None
        best_prob = -1.0
        for method in candidate_methods:
            prob = self.predict_success_prob(task, world_state, method_stats, method)
            if prob > best_prob:
                best_prob = prob
                best = method
        return best, best_prob

    def extract_features(self, task: Dict[str, Any], world_state: Dict[str, Any],
                         method_stats: Dict) -> np.ndarray:
        features = np.zeros(len(self.feature_names))
        idx = 0

        # Base features
        features[idx] = self._calculate_task_depth(task)
        idx += 1
        features[idx] = self._calculate_goal_overlap(task, world_state)
        idx += 1
        features[idx] = self._calculate_method_failure_rate(task, method_stats,
                                                            task.get("selected_method", "unknown"))
        idx += 1
        features[idx] = self._calculate_state_diversity(world_state)
        idx += 1

        # Temporal features
        if self.feature_config.get("use_temporal_features"):
            features[idx] = self._time_since_creation(task)
            idx += 1
            features[idx] = self._deadline_proximity(task)
            idx += 1

        # Priority weighting
        if self.feature_config.get("use_priority_weighting"):
            priority = task.get("priority", 0.5)
            decay = 0.95 ** (self._calculate_task_depth(task) * 20)  # convert to depth count
            features[idx] = priority * decay
            idx += 1

        return features

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.25):
        with self._lock:
            unique, counts = np.unique(y, return_counts=True)
            if len(unique) < 2:
                logger.error("Need at least two classes for training")
                return 0.0
    
            # Optionally limit training samples
            max_samples = self.dt_config.get('validation', {}).get('max_training_samples', 10000)
            if len(X) > max_samples:
                indices = np.random.choice(len(X), max_samples, replace=False)
                X, y = X[indices], y[indices]
    
            # Determine if stratification is safe
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
    
            # Create and train model
            self.model = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease,
                ccp_alpha=self.ccp_alpha,
                class_weight=self.class_weight,
                random_state=self.random_state
            )
            self.model.fit(X_train_scaled, y_train)
    
            # Evaluate if validation set is non-empty
            if len(y_val) > 0:
                val_acc = accuracy_score(y_val, self.model.predict(X_val_scaled))
                logger.info(f"Validation accuracy: {val_acc:.2f}")
                if self.dt_config.get('validation', {}).get('max_training_samples', 0):
                    logger.info(f"Validation F1: {f1_score(y_val, self.model.predict(X_val_scaled)):.2f}")
            else:
                logger.warning("Validation set empty; skipping validation metrics.")
    
            self.trained = True
            joblib.dump((self.model, self.scaler), self.model_path)
            return val_acc if len(y_val) > 0 else 0.0

    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        tasks, method_stats, world_states = self.load_planning_db()
        X = []
        y = []
        for task, state in zip(tasks, world_states):
            # Need a method_id – for training we could treat each task as a single sample
            # with its chosen method; but the feature extractor expects a method.
            # For simplicity, we assume each task in the DB has a "selected_method" field.
            method_id = task.get("selected_method")
            if method_id is None:
                continue
            features = self.extract_features(task, state, method_stats)   # pass method_stats
            X.append(features)
            outcome = task.get("outcome", "failure")
            y.append(1 if outcome == "success" else 0)
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
        # Simple success rate fallback
        key = (task.get("name"), method_id)
        stats = method_stats.get(key, {'success': 1, 'total': 2})
        if stats['total'] == 0:
            return 0.5
        return stats['success'] / stats['total']

    def _priority_decay_factor(self, task):
        return 0.95 ** self._calculate_task_depth(task)  # Exponential decay

    def retrain_model(self):
        """Periodic retraining with new data"""
        X, y = self.load_training_data()
        if len(np.unique(y)) >= 2:
            self.train(X, y)
            logger.info("Model retrained with updated data")

    def cost_complexity_pruning(self, X_train, y_train):
        """
        Perform post-training cost-complexity pruning and refit model with optimal ccp_alpha.
        """
        path = self.model.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas = path.ccp_alphas[:-1]  # Drop the last value (trivial tree)
        impurities = path.impurities[:-1]
    
        if len(ccp_alphas) == 0:
            logger.warning("No pruning candidates found.")
            return self.model
    
        models = []
        for alpha in ccp_alphas:
            clf = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                class_weight=self.class_weight,
                random_state=self.random_state,
                ccp_alpha=alpha
            )
            clf.fit(X_train, y_train)
            models.append(clf)
    
        # Evaluate and choose the best pruned tree (based on validation accuracy)
        best_score = -1
        best_model = self.model
        for clf in models:
            score = clf.score(X_train, y_train)  # Could use validation set instead
            if score > best_score:
                best_score = score
                best_model = clf
    
        logger.info(f"Pruned tree selected with ccp_alpha={best_model.ccp_alpha}")
        return best_model

    def report_feature_importance(self):
        """Return sorted feature importance scores"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return []
        importances = self.model.feature_importances_
        return sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )

if __name__ == "__main__":
    print("\n=== Running Decision Tree Heuristic Test ===\n")
    printer.status("Init", "Decision Tree Heuristic initialized", "success")

    planner01 = DecisionTreeHeuristic()
    print(planner01)
    print("\n* * * * * Phase 2 - Extracted Features * * * * *\n")
    from datetime import datetime, timedelta
    now = datetime.now()
    task = {
        "name": "navigate_to_room",
        "selected_method": 0,
        "priority": 0.8,
        "goal_state": {"location": "room_1"},
        "parent": {
            "name": "root_task",
            "parent": None
        },
        "creation_time": now.isoformat(),
        "deadline": (now + timedelta(hours=2)).isoformat()
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
    id = 2457467

    prob = planner01.predict_success_prob(task=task, world_state=state, method_stats=stats, method_id=id)
    features = planner01.extract_features(task=task, world_state=state, method_stats=stats)
    printer.pretty("Success probability:", prob, "Success")
    printer.pretty("Extracted Features:", features, "Success")
    print("\n* * * * * Phase 3 - Training * * * * *\n")
    X, y = planner01.load_training_data()
    
    # Class distribution check
    unique, counts = np.unique(y, return_counts=True)
    printer.pretty(f"Class distribution: {dict(zip(unique, counts))}", "", "Info")
    
    if len(unique) >= 2:
        planner01.train(X, y)
        # Test feature importance reporting
        importance = planner01.report_feature_importance()
        printer.pretty("Feature Importance:", 
                        "\n".join(f"{f}: {imp:.4f}" for f, imp in importance), 
                        "Success")
        
        # Test method selection
        print("\n* * * * * Phase 4 - Method Selection * * * * *\n")
        task = {
            "name": "navigation",
            "priority": 0.8,
            "goal_state": {"position": "target"},
            "creation_time": datetime.now().isoformat(),
            "deadline": (datetime.now() + timedelta(hours=1)).isoformat()
        }
        state = {"position": "start", "cpu_available": 70, "memory_available": 2048}
        methods = ["A*", "RRT", "D*"]
        
        best_method, confidence = planner01.select_best_method(task, state, methods, stats)
        printer.pretty(f"Recommended method: {best_method} (confidence: {confidence:.2f})", "", "Success")
    print("\n=== Successfully Ran Decision Tree Heuristic ===\n")
