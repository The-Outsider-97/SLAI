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

import joblib
import numpy as np
import os
import yaml, json

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
from typing import Dict, Union, List
from types import SimpleNamespace

from logs.logger import get_logger

logger = get_logger("Gradient Boosting Heuristic")

CONFIG_PATH = "src/agents/planning/configs/planning_config.yaml"
TEMPLATES_PATH = "src/agents/planning/templates/planning_db.json"

def dict_to_namespace(d):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    return d

def get_config_section(section: Union[str, Dict], config_file_path: str):
    if isinstance(section, dict):
        return dict_to_namespace(section)
    
    with open(config_file_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if section not in config:
        raise KeyError(f"Section '{section}' not found in config file: {config_file_path}")
    return dict_to_namespace(config[section])

class GradientBoostingHeuristic:
    def __init__(self, agent=None,
                 config_section_name: str = "gradient_boosting_heuristic",
                 config_file_path: str = CONFIG_PATH,):
        self.config = get_config_section(config_section_name, config_file_path)
        self.agent = agent
        self.model = self._init_model()
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_names = self._get_feature_names()
        self.feature_importances_ = None

    def _init_model(self):
        return GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            subsample=self.config.subsample,
            min_samples_split=self.config.min_samples_split,
            n_iter_no_change=self.config.early_stopping_rounds,
            validation_fraction=self.config.validation_fraction,
            random_state=42
        )

    def _get_feature_names(self) -> List[str]:
        base_features = [
            'task_depth',
            'goal_overlap',
            'method_failure_rate',
            'state_diversity'
        ]
        if self.config.feature_config.use_priority:
            base_features.append('task_priority')
        if self.config.feature_config.use_resource_check:
            base_features.extend(['cpu_available', 'memory_available'])
        return base_features

    def extract_features(self, task, world_state, method_stats) -> np.ndarray:
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
        if self.config.feature_config.use_priority:
            features[feature_idx] = task.priority if hasattr(task, 'priority') else 0.5
            feature_idx += 1
            
        if self.config.feature_config.use_resource_check:
            features[feature_idx] = world_state.get('cpu_available', 0)
            feature_idx += 1
            features[feature_idx] = world_state.get('memory_available', 0)
            feature_idx += 1

        return self.scaler.transform(features.reshape(1, -1))

    def _calculate_task_depth(self, task):
        depth = 0
        current = task
        while current.parent:
            depth += 1
            current = current.parent
        return depth / 10  # Normalized

    def _calculate_goal_overlap(self, task, world_state):
        return len(set(task.goal_state.keys()) & set(world_state.keys())) / len(task.goal_state)

    def _calculate_method_failure_rate(self, task, method_stats):
        key = (task.name, task.selected_method)
        stats = method_stats.get(key, {'success': 1, 'total': 2})
        return 1 - (stats['success'] / stats['total'])

    def _calculate_state_diversity(self, world_state):
        state_vals = [float(v) for v in world_state.values() 
                     if isinstance(v, (int, float))]
        return np.std(state_vals) if state_vals else 0

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        self.model.fit(X_train_scaled, y_train)
        
        # Track feature importances
        self.feature_importances_ = self.model.feature_importances_
        
        # Evaluate validation performance
        val_pred = self.model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_pred)
        logger.info(f"Validation accuracy: {val_acc:.2f}")
        
        self.trained = True
        joblib.dump((self.model, self.scaler), 'gb_heuristic_model.pkl')
        return val_acc

    def predict_success_prob(self, task, world_state, method_stats):
        if not self.trained:
            if os.path.exists('gb_heuristic_model.pkl'):
                self.model, self.scaler = joblib.load('gb_heuristic_model.pkl')
                self.trained = True
            else:
                logger.warning("Using untrained model with default predictions")
                return 0.5
                
        features = self.extract_features(task, world_state, method_stats)
        return self.model.predict_proba(features)[0][1]
    
    @staticmethod
    def load_planning_db(path: str = None):
        """Load task, world state, and method statistics from the planning database."""
        if path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, TEMPLATES_PATH)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    
        tasks = data["tasks"]
        method_stats = {
            tuple(k.split(":")): v for k, v in data["method_stats"].items()
        }
        world_states = data["world_states"]
    
        return tasks, method_stats, world_states

if __name__ == "__main__":
    print("")
    print("\n=== Running Task Scheduler ===")
    print("")
    from unittest.mock import Mock
    mock_agent = Mock()
    mock_agent.shared_memory = {}
    planner02 = GradientBoostingHeuristic(agent=mock_agent)
    print("")
    print("\n=== Successfully Ran Task Scheduler ===\n")
