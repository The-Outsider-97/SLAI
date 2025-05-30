
import joblib
import numpy as np
import os
import yaml, json

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
from typing import Dict, Union, List
from types import SimpleNamespace

from logs.logger import get_logger

logger = get_logger("Decision Tree Heuristic")

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

class DecisionTreeHeuristic:
    def __init__(self, agent=None,
                 config_section_name: str = "decision_tree_heuristic",
                 config_file_path: str = CONFIG_PATH):
        self.config = get_config_section(config_section_name, config_file_path)
        self.model = self._init_model()
        self.agent = agent
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_names = self._get_feature_names()
        self.feature_importances_ = None
        
    def _init_model(self):
        return DecisionTreeClassifier(
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_impurity_decrease=self.config.min_impurity_decrease,
            ccp_alpha=self.config.ccp_alpha,
            class_weight=self.config.class_weight,
            random_state=42
        )

    def _get_feature_names(self) -> List[str]:
        base_features = [
            'task_depth',
            'goal_overlap',
            'method_failure_rate',
            'state_diversity'
        ]
        if self.config.feature_config.use_temporal_features:
            base_features.extend(['time_since_creation', 'deadline_proximity'])
        if self.config.feature_config.use_priority_weighting:
            base_features.append('priority_weight')
        return base_features

    def extract_features(self, task, world_state, method_stats) -> np.ndarray:
        features = np.zeros(len(self.feature_names))
        feature_idx = 0
        
        # Core features
        features[feature_idx] = self._normalized_task_depth(task)
        feature_idx += 1
        
        features[feature_idx] = self._calculate_goal_overlap(task, world_state)
        feature_idx += 1
        
        features[feature_idx] = self._calculate_method_failure_rate(task, method_stats)
        feature_idx += 1
        
        features[feature_idx] = self._calculate_state_diversity(world_state)
        feature_idx += 1

        # Temporal features
        if self.config.feature_config.use_temporal_features:
            features[feature_idx] = self._time_since_creation(task)
            feature_idx += 1
            
            features[feature_idx] = self._deadline_proximity(task)
            feature_idx += 1

        # Priority weighting
        if self.config.feature_config.use_priority_weighting:
            features[feature_idx] = task.priority * self._priority_decay_factor(task)
            feature_idx += 1

        return self.scaler.transform(features.reshape(1, -1))

    def _normalized_task_depth(self, task):
        depth = 0
        current = task
        while current.parent:
            depth += 1
            current = current.parent
        return depth / 20  # Normalized assuming max depth 20

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

    def _time_since_creation(self, task):
        return (datetime.now() - task.creation_time).total_seconds() / 3600  # Hours

    def _deadline_proximity(self, task):
        if not hasattr(task, 'deadline'):
            return 0.0
        total_time = (task.deadline - task.creation_time).total_seconds()
        elapsed = (datetime.now() - task.creation_time).total_seconds()
        return elapsed / total_time if total_time > 0 else 0.0

    def _priority_decay_factor(self, task):
        return 0.95 ** self._normalized_task_depth(task)  # Exponential decay

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train with validation and resource constraints"""
        if len(X) > self.config.validation.max_training_samples:
            X = X[:self.config.validation.max_training_samples]
            y = y[:self.config.validation.max_training_samples]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.config.validation.test_size,
            stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        self.model.fit(X_train_scaled, y_train)
        self.feature_importances_ = self.model.feature_importances_

        # Prune tree if configured
        if self.config.ccp_alpha > 0:
            self.model = self.model.cost_complexity_pruning(X_train_scaled, y_train)

        # Evaluate
        val_pred = self.model.predict(X_val_scaled)
        logger.info(f"Validation Accuracy: {accuracy_score(y_val, val_pred):.2f}")
        logger.info(f"Validation F1 Score: {f1_score(y_val, val_pred):.2f}")

        self.trained = True
        joblib.dump((self.model, self.scaler), 'dt_heuristic_model.pkl')

    def predict_success_prob(self, task, world_state, method_stats):
        """Predict with confidence and resource checks"""
        if not self.trained:
            if os.path.exists('dt_heuristic_model.pkl'):
                self.model, self.scaler = joblib.load('dt_heuristic_model.pkl')
                self.trained = True
                logger.info("Loaded pre-trained decision tree model")
            else:
                logger.warning("Using fallback heuristic predictions")
                return self._fallback_heuristic(task, world_state)

        try:
            features = self.extract_features(task, world_state, method_stats)
            return self.model.predict_proba(features)[0][1]
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return self._fallback_heuristic(task, world_state)

    def _fallback_heuristic(self, task, world_state):
        """Fallback logic using simple rules"""
        base_prob = 0.5
        if hasattr(task, 'priority'):
            base_prob += task.priority * 0.2
        return min(max(base_prob, 0.0), 1.0)
    
    @staticmethod
    def load_planning_db(path: str = None):
        """Load task, world state, method statistics, and decision tree templates from the planning database."""
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
        dt_templates = data.get("decision_tree_templates", {})  # Optional fallback
    
        return tasks, method_stats, world_states, dt_templates

if __name__ == "__main__":
    print("")
    print("\n=== Running Decision Tree Heuristic ===")
    print("")
    from unittest.mock import Mock
    mock_agent = Mock()
    mock_agent.shared_memory = {}
    planner01 = DecisionTreeHeuristic(agent=mock_agent)
    print("")
    print("\n=== Successfully Ran Decision Tree Heuristic ===\n")
