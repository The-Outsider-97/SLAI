from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import os

class DecisionTreeHeuristic:
    def __init__(self, max_depth=5):
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_names = [
            'task_depth', 
            'goal_overlap',
            'method_failure_rate',
            'state_diversity'
        ]
        
    def extract_features(self, task, world_state, method_stats):
        """Convert planner state into feature vector"""
        features = np.zeros(len(self.feature_names))
        
        # Task depth calculation
        depth = 0
        current = task
        while current.parent:
            depth += 1
            current = current.parent
        features[0] = depth
        
        # Goal overlap (simplified)
        features[1] = len(set(task.goal_state.keys()) & 
                      set(world_state.keys())) / len(task.goal_state)
        
        # Method failure rate
        key = (task.name, task.selected_method)
        stats = method_stats.get(key, {'success': 1, 'total': 2})
        features[2] = 1 - (stats['success'] / stats['total'])
        
        # State diversity (std dev of key state vars)
        state_vals = [float(v) for v in world_state.values() 
                     if isinstance(v, (int, float))]
        features[3] = np.std(state_vals) if state_vals else 0
        
        return self.scaler.transform(features.reshape(1, -1))

    def train(self, X, y):
        """Train on historical planning outcomes"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.trained = True
        joblib.dump(self.model, 'dt_heuristic_model.pkl')

    def predict_success_prob(self, task, world_state, method_stats):
        """Predict plan success probability (0-1)"""
        if not self.trained and os.path.exists('dt_heuristic_model.pkl'):
            self.model = joblib.load('dt_heuristic_model.pkl')
            self.trained = True
            
        features = self.extract_features(task, world_state, method_stats)
        return self.model.predict_proba(features)[0][1] if self.trained else 0.5
