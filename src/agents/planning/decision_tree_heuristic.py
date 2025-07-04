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
import joblib
import yaml, json
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
from typing import List, Any, Dict, Tuple

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from src.agents.planning.utils.base_heuristic import BaseHeuristics
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
        self.max_depth = self.heuristics_config.get('max_depth')
        self.min_samples_split = self.heuristics_config.get('min_samples_split')
        self.class_weight = self.heuristics_config.get('class_weight')
        self.trained = self.heuristics_config.get('trained')
        self.random_state = self.heuristics_config.get('random_state')
        self.planning_db_path = self.heuristics_config.get('planning_db_path')
        self.heuristic_model_path = self.heuristics_config.get('heuristic_model_path')
        os.makedirs(self.heuristic_model_path, exist_ok=True)
        self.model_path = os.path.join(self.heuristic_model_path, 'dt_heuristic_model.pkl')

        self.dth_config = get_config_section('decision_tree_heuristic')
        self.min_impurity_decrease = self.dth_config.get('min_impurity_decrease')
        self.ccp_alpha = self.dth_config.get('ccp_alpha')
        self.feature_config = self.dth_config.get('feature_config', {
            'use_temporal_features', 'use_priority_weighting', 'use_method_complexity'
        })
        self.validation = self.dth_config.get('validation', {
            'test_size', 'max_training_samples'
        })

        self.agent = {}
        self.model = self._init_model()
        self.scaler = StandardScaler(with_mean=False, with_std=False)
        self.feature_names = self._get_feature_names()
        self.feature_importances_ = None

        logger.info(f"Decision Tree Heuristic succesfully initialized")

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

    def _get_feature_names(self) -> List[str]:
        base_features = [
            'task_depth',
            'goal_overlap',
            'method_failure_rate',
            'state_diversity'
        ]
        if self.feature_config.get("use_temporal_features"):
            base_features.extend(['time_since_creation', 'deadline_proximity'])
        if self.feature_config.get("use_priority_weighting"):
            base_features.append('priority_weight')
        return base_features

    def predict_success_prob(self, task, world_state, method_stats, **kwargs):
        """Predict with confidence and resource checks"""
        printer.status("INIT", "Success prob predictor succesfully initialized", "info")

        if not self.trained:
            if os.path.exists(self.model_path):  # Use configured path
                self.model, self.scaler = joblib.load(self.model_path)
                self.trained = True
                logger.info("Loaded pre-trained decision tree model")
            else:
                logger.warning("Using fallback heuristic predictions")
                return self._fallback_heuristic(task, world_state, method_stats)

        try:
            features = self.extract_features(task, world_state, method_stats)
            features = features.reshape(1, -1)  # ← Ensure 2D array shape
            return self.model.predict_proba(features)[0][1]

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return self._fallback_heuristic(task, world_state, method_stats)

    def extract_features(self, task, world_state, method_stats) -> np.ndarray:
        printer.status("INIT", "Freature extractor succesfully initialized", "info")

        features = np.zeros(len(self.feature_names))
        feature_idx = 0
        
        # Core features
        features[feature_idx] = self._calculate_task_depth(task)
        feature_idx += 1
        
        features[feature_idx] = self._calculate_goal_overlap(task, world_state)
        feature_idx += 1
        
        features[feature_idx] = self._calculate_method_failure_rate(task, method_stats)
        feature_idx += 1
        
        features[feature_idx] = self._calculate_state_diversity(world_state)
        feature_idx += 1

        # Temporal features
        if self.feature_config.get("use_temporal_features"):
            features[feature_idx] = self._time_since_creation(task)
            feature_idx += 1
            
            features[feature_idx] = self._deadline_proximity(task)
            feature_idx += 1

        # Priority weighting
        if self.feature_config.get("use_priority_weighting"):
            features[feature_idx] = task.get("priority", 0.0) * self._priority_decay_factor(task)
            feature_idx += 1

        return features.astype(np.float32)

    def _priority_decay_factor(self, task):
        return 0.95 ** self._calculate_task_depth(task)  # Exponential decay

    def retrain_model(self):
        """Periodic retraining with new data"""
        X, y = self.load_training_data()
        if len(np.unique(y)) >= 2:
            self.train(X, y)
            logger.info("Model retrained with updated data")

    def load_training_data(self):
        """Load historical execution data for training"""
        tasks, _, world_states = self.load_planning_db()
        X = []
        y = []
        
        for task, state in zip(tasks, world_states):
            features = self.extract_features(task, state, {})
            X.append(features)
            
            outcome = task.get("outcome", "failure")
            y.append(1 if outcome == "success" else 0)
        
        return np.array(X), np.array(y)

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train with validation and resource constraints"""
        # Check class distribution
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logger.error("Cannot train with only one class in dataset")
            return 0.0

        if any(c < 2 for c in counts):
            logger.warning("Insufficient class distribution for stratified split. Proceeding without stratification.")
            stratify = None
        else:
            stratify = y

        if len(X) > self.validation.get("max_training_samples"):
            X = X[:self.validation.get("max_training_samples")]
            y = y[:self.validation.get("max_training_samples")]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.validation.get("test_size"),
            stratify=stratify
        )

        # Check training set has at least 2 classes
        if len(np.unique(y_train)) < 2:
            logger.warning("Training set has only one class - skipping training")
            return 0.0

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        self.model.fit(X_train_scaled, y_train)
        self.feature_importances_ = self.model.feature_importances_

        # Prune tree if configured
        if self.ccp_alpha > 0:
            self.model = self.cost_complexity_pruning(X_train_scaled, y_train)

        # Evaluate
        val_pred = self.model.predict(X_val_scaled)
        logger.info(f"Validation Accuracy: {accuracy_score(y_val, val_pred):.2f}")
        logger.info(f"Validation F1 Score: {f1_score(y_val, val_pred):.2f}")

        self.trained = True
        joblib.dump((self.model, self.scaler), self.model_path)

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

    def _fallback_heuristic(self, task, world_state, method_stats):
        """Fallback logic using simple rules"""
        printer.status("INIT", "Heuristic fallback activated", "info")

        key = (task.get("name"), task.get("selected_method", None))
        stats = method_stats.get(key, {'success': 1, 'total': 2})
        
        if stats['total'] > 0:
            return stats['success'] / stats['total']
        
        # Fallback to priority-based heuristic
        priority = task.get("priority", 0.5)
        return 0.3 + (priority * 0.4)  # Between 0.3-0.7
    
    def select_best_method(self, task, world_state, candidate_methods, method_stats):
        """Select method with highest predicted success probability"""
        best_method = None
        best_prob = -1
        
        for method_id in candidate_methods:
            task["selected_method"] = method_id
            prob = self.predict_success_prob(task, world_state, method_stats)
            if prob > best_prob:
                best_prob = prob
                best_method = method_id
                
        return best_method, best_prob
    
    def report_feature_importance(self):
        """Return sorted feature importance scores"""
        if self.feature_importances_ is None:
            return []
            
        return sorted(
            zip(self.feature_names, self.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )

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

    prob = planner01.predict_success_prob(task=task, world_state=state, method_stats=stats)
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
