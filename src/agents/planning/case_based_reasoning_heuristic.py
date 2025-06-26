"""
Case-Based Reasoning Heuristic for Experience-Guided Planning

This module implements a case-based reasoning approach that retrieves similar past planning cases 
and adapts their successful solutions to the current problem. It's particularly effective when 
historical execution data is available, allowing the system to leverage proven solutions for 
recurring patterns in task planning.

Real-World Use Case:
1. Manufacturing Troubleshooting: When equipment fails, retrieve similar past failure scenarios 
   and their successful repair procedures.
2. Customer Support: For recurring customer issues, retrieve and adapt previously successful resolution paths.
3. Medical Diagnosis: Find similar patient cases and their effective treatment plans when diagnosing new cases.
4. Emergency Response: Access successful response strategies from similar past emergencies.
"""

import os
import json
import numpy as np

from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from src.agents.planning.utils.base_heuristic import BaseHeuristics
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Case-Based Reasoning Heuristic")
printer = PrettyPrinter

class DotDict(dict):
    def __getattr__(self, item):
        return self.get(item)

class CaseBasedReasoningHeuristic(BaseHeuristics):
    def __init__(self):
        self.config = load_global_config()

        self.heuristics_config = get_config_section('global_heuristic')
        self.trained = self.heuristics_config.get('trained')
        self.random_state = self.heuristics_config.get('random_state')
        self.planning_db_path = self.heuristics_config.get('planning_db_path')
        self.heuristic_model_path = self.heuristics_config.get('heuristic_model_path')
        os.makedirs(self.heuristic_model_path, exist_ok=True)
        self.model_path = os.path.join(self.heuristic_model_path, 'cbr_model.pkl')
        self.case_base_path = os.path.join(self.heuristic_model_path, 'cbr_case_base.json')

        self.cbr_config = get_config_section('case_based_reasoning_heuristic')
        self.similarity_threshold = self.cbr_config.get('similarity_threshold', 0.7)
        self.max_cases = self.cbr_config.get('max_cases', 1000)
        self.min_similar_cases = self.cbr_config.get('min_similar_cases', 3)
        self.feature_weights = self.cbr_config.get('feature_weights', {})
        self.adaptation_rules = self.cbr_config.get('adaptation_rules', {})
        
        self.feature_config = self.cbr_config.get('feature_config', {})
        self.feature_names = self._get_feature_names()
        
        self.case_base = []
        self.scaler = StandardScaler()
        self.nn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self.case_index = {}
        
        logger.info("Case-Based Reasoning Heuristic initialized")
        self.load_model()

    def _get_feature_names(self) -> List[str]:
        """Define feature set for case representation"""
        base_features = [
            'task_depth',
            'goal_overlap',
            'state_diversity',
            'resource_utilization'
        ]
        if self.feature_config.get("use_priority"):
            base_features.append('task_priority')
        if self.feature_config.get("use_temporal"):
            base_features.extend(['time_since_creation', 'deadline_proximity'])
        if self.feature_config.get("use_contextual"):
            base_features.extend(['environment_complexity', 'agent_experience'])
        return base_features

    def extract_features(self, task, world_state, method_stats) -> np.ndarray:
        """Extract normalized feature vector for current state"""
        features = np.zeros(len(self.feature_names))
        feature_idx = 0
        
        # Base features
        features[feature_idx] = self._calculate_task_depth(task)
        feature_idx += 1
        
        features[feature_idx] = self._calculate_goal_overlap(task, world_state)
        feature_idx += 1
        
        features[feature_idx] = self._calculate_state_diversity(world_state)
        feature_idx += 1
        
        features[feature_idx] = self._calculate_resource_utilization(world_state)
        feature_idx += 1

        # Optional features
        if self.feature_config.get("use_priority"):
            features[feature_idx] = task.get('priority', 0.5)
            feature_idx += 1
            
        if self.feature_config.get("use_temporal"):
            features[feature_idx] = self._time_since_creation(task)
            feature_idx += 1
            features[feature_idx] = self._deadline_proximity(task)
            feature_idx += 1
            
        if self.feature_config.get("use_contextual"):
            features[feature_idx] = self._environment_complexity(world_state)
            feature_idx += 1
            features[feature_idx] = self._agent_experience(task)
            feature_idx += 1
            
        return features.astype(np.float32)

    def _calculate_resource_utilization(self, world_state) -> float:
        """Calculate overall resource utilization percentage"""
        cpu = world_state.get('cpu_available', 1.0)
        memory = world_state.get('memory_available', 1.0)
        return 1.0 - ((cpu + memory) / 2.0)

    def _environment_complexity(self, world_state) -> float:
        """Quantify environmental complexity"""
        num_objects = len(world_state.get('objects', []))
        num_agents = len(world_state.get('agents', []))
        return min(1.0, (num_objects + num_agents) / 20.0)

    def _agent_experience(self, task) -> float:
        """Estimate agent experience with task type"""
        task_type = task.get('type', 'generic')
        return self.heuristics_config.get(f'experience_{task_type}', 0.5)

    def select_best_method(self, task, world_state, candidate_methods, method_stats) -> Tuple[str, float]:
        """
        Retrieve and adapt similar cases to select the best method
        
        Returns:
            Tuple of (selected_method, confidence_score)
        """
        current_features = self.extract_features(task, world_state, method_stats)
        scaled_features = self.scaler.transform([current_features])
        
        # Find similar cases
        similar_cases = self.retrieve_similar_cases(scaled_features)
        
        if not similar_cases:
            # Fallback to simple frequency-based selection
            return self._fallback_selection(candidate_methods)
        
        # Analyze successful methods in similar cases
        method_scores = self.analyze_similar_cases(similar_cases, candidate_methods)
        
        if not method_scores:
            return self._fallback_selection(candidate_methods)
        
        max_score = max(method_scores.values())
        if max_score <= 0:
            return self._fallback_selection(candidate_methods)
            
        # Apply adaptation rules for current context
        self.apply_adaptation_rules(method_scores, task, world_state)
        
        # Select best method with highest score
        best_method = max(method_scores, key=method_scores.get)
        confidence = method_scores[best_method] / max_score
        
        logger.info(f"Selected method {best_method} with confidence {confidence:.2f} "
                    f"based on {len(similar_cases)} similar cases")
        return best_method, confidence

    def retrieve_similar_cases(self, query_features) -> List[dict]:
        """Find most similar cases using nearest neighbors search"""
        if len(self.case_base) < self.min_similar_cases:
            return []
            
        distances, indices = self.nn_model.kneighbors(query_features)
        similar_cases = []
        
        for i, distance in zip(indices[0], distances[0]):
            if 1 - distance > self.similarity_threshold:
                similar_cases.append(self.case_base[i])
                
        return similar_cases

    def analyze_similar_cases(self, similar_cases, candidate_methods) -> Dict[str, float]:
        """Score methods based on success in similar cases"""
        method_stats = {method: {'success': 0, 'total': 0} for method in candidate_methods}
        
        for case in similar_cases:
            method = case['method_used']
            if method not in candidate_methods:
                continue
                
            method_stats[method]['total'] += 1
            if case['outcome'] == 'success':
                method_stats[method]['success'] += 1
                
        # Calculate weighted scores
        method_scores = {}
        for method, stats in method_stats.items():
            if stats['total'] > 0:
                base_score = stats['success'] / stats['total']
                age_seconds = datetime.now().timestamp() - case['timestamp']
                age_years = age_seconds / 31536000  # Seconds in a year
                recency_weight = min(1.0, max(0.0, 0.5 - 0.5 * age_years))
                method_scores[method] = base_score * recency_weight
                
        return method_scores

    def apply_adaptation_rules(self, method_scores, task, world_state):
        """Adjust scores based on contextual adaptation rules"""
        # Apply resource-based adaptations
        if 'resource_adaptation' in self.adaptation_rules:
            cpu = world_state.get('cpu_available', 1.0)
            memory = world_state.get('memory_available', 1.0)
            
            if cpu < 0.3 or memory < 0.3:
                for method in list(method_scores.keys()):
                    if 'heavy' in method:
                        method_scores[method] *= 0.7
                    elif 'light' in method:
                        method_scores[method] *= 1.3
        
        # Apply priority-based adaptations
        if 'priority_adaptation' in self.adaptation_rules:
            priority = task.get('priority', 0.5)
            if priority > 0.8:
                for method in list(method_scores.keys()):
                    if 'fast' in method:
                        method_scores[method] *= 1.2
                    elif 'slow' in method:
                        method_scores[method] *= 0.8

    def predict_success_prob(self, task, world_state, method_stats, method_id: str) -> float:
        """Predict success probability using similar historical cases"""
        current_features = self.extract_features(task, world_state, method_stats)
        scaled_features = self.scaler.transform([current_features])
        similar_cases = self.retrieve_similar_cases(scaled_features)
        
        if not similar_cases:
            return 0.5  # Neutral probability if no similar cases
            
        # Filter to cases using the specific method
        method_cases = [c for c in similar_cases if c['method_used'] == method_id]
        
        if not method_cases:
            return 0.5
            
        # Calculate success rate
        successes = sum(1 for c in method_cases if c['outcome'] == 'success')
        return successes / len(method_cases)

    def record_outcome(self, task, world_state, method_used: str, outcome: str):
        """Record new case after execution"""
        if outcome not in ['success', 'failure']:
            logger.warning(f"Invalid outcome '{outcome}' - must be 'success' or 'failure'")
            return
            
        # Create new case
        new_case = {
            'features': self.extract_features(task, world_state, {}).tolist(),
            'method_used': method_used,
            'outcome': outcome,
            'timestamp': datetime.now().timestamp(),
            'task_type': task.get('name', 'unknown'),
            'context': {
                'resources': {
                    'cpu': world_state.get('cpu_available', 0),
                    'memory': world_state.get('memory_available', 0)
                },
                'priority': task.get('priority', 0.5)
            }
        }
        
        # Add to case base
        self.case_base.append(new_case)
        
        # Maintain case base size
        if len(self.case_base) > self.max_cases:
            self.case_base = sorted(
                self.case_base, 
                key=lambda x: x['timestamp'], 
                reverse=True
            )[:self.max_cases]
            
        # Update similarity model
        self._update_similarity_model()
        self.save_model()
        
        logger.info(f"Recorded new case for {task.get('name', 'unknown')} "
                   f"with method {method_used} ({outcome})")

    def _update_similarity_model(self):
        """Retrain the similarity model with updated case base"""
        if not self.case_base:
            return
            
        # Prepare feature matrix
        features = np.array([c['features'] for c in self.case_base])
        
        # Fit scaler and model
        self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)
        self.nn_model.fit(scaled_features)
        self.trained = True

    def _fallback_selection(self, candidate_methods) -> Tuple[str, float]:
        """Fallback selection when no similar cases exist"""
        # Simple frequency-based fallback
        method = np.random.choice(candidate_methods)
        return method, 0.5

    def save_model(self):
        """Save case base and model configuration"""
        model_data = {
            'case_base': self.case_base,
            'feature_names': self.feature_names,
            'trained': self.trained
        }
        with open(self.case_base_path, 'w') as f:
            json.dump(model_data, f)
        logger.info(f"Saved {len(self.case_base)} cases to {self.case_base_path}")

    def load_model(self):
        """Load case base and similarity model"""
        if not os.path.exists(self.case_base_path):
            logger.info("No case base found - starting with empty case base")
            self.case_base = []
            self.trained = False
            return
            
        try:
            with open(self.case_base_path, 'r') as f:
                model_data = json.load(f)
            self.case_base = model_data.get('case_base', [])
            self.feature_names = model_data.get('feature_names', self._get_feature_names())
            self.trained = model_data.get('trained', False)
            
            if self.case_base:
                self._update_similarity_model()
                
            logger.info(f"Loaded {len(self.case_base)} cases from {self.case_base_path}")
        except Exception as e:
            logger.error(f"Failed to load case base: {str(e)}")
            self.case_base = []
            self.trained = False

    def get_case_stats(self) -> dict:
        """Return statistics about the case base"""
        stats = {
            'total_cases': len(self.case_base),
            'success_rate': 0,
            'recent_success_rate': 0,
            'cases_by_type': defaultdict(int),
            'cases_by_method': defaultdict(int)
        }
        
        if not self.case_base:
            return stats
            
        # Calculate overall success rate
        successes = sum(1 for c in self.case_base if c['outcome'] == 'success')
        stats['success_rate'] = successes / len(self.case_base)
        
        # Calculate recent success rate (last 30 days)
        recent_cases = [c for c in self.case_base 
                       if c['timestamp'] > (datetime.now().timestamp() - 2592000)]
        if recent_cases:
            recent_successes = sum(1 for c in recent_cases if c['outcome'] == 'success')
            stats['recent_success_rate'] = recent_successes / len(recent_cases)
            
        # Count cases by type and method
        for case in self.case_base:
            stats['cases_by_type'][case['task_type']] += 1
            stats['cases_by_method'][case['method_used']] += 1
            
        return stats

if __name__ == "__main__":
    print("\n=== Running Case-Based Reasoning Heuristic Test ===\n")
    printer.status("Init", "Case-Based Reasoning Heuristic initialized", "success")

    cbr_heuristic = CaseBasedReasoningHeuristic()
    
    # Create test cases
    for i in range(5):
        task = {
            "name": f"task_{i}",
            "priority": 0.7,
            "goal_state": {"target": i},
            "creation_time": datetime.now().isoformat(),
            "deadline": (datetime.now() + timedelta(hours=i+1)).isoformat()
        }
        state = {
            "cpu_available": 0.8 - (i*0.1),
            "memory_available": 0.9 - (i*0.1)
        }
        method = f"method_{i % 3}"
        outcome = "success" if i % 2 == 0 else "failure"
        cbr_heuristic.record_outcome(task, state, method, outcome)
    
    # Test case retrieval
    test_task = {
        "name": "test_task",
        "priority": 0.75,
        "goal_state": {"target": 3},
        "creation_time": datetime.now().isoformat(),
        "deadline": (datetime.now() + timedelta(hours=2)).isoformat()
    }
    test_state = {"cpu_available": 0.7, "memory_available": 0.8}
    
    print("\n* * * * * Phase 1 - Method Selection * * * * *\n")
    methods = ["method_0", "method_1", "method_2"]
    best_method, confidence = cbr_heuristic.select_best_method(
        test_task, test_state, methods, {}
    )
    printer.pretty(f"Selected method: {best_method} (confidence: {confidence:.2f})", "", "Success")
    
    print("\n* * * * * Phase 2 - Success Prediction * * * * *\n")
    for method in methods:
        prob = cbr_heuristic.predict_success_prob(test_task, test_state, {}, method)
        printer.pretty(f"Success probability for {method}: {prob:.2f}", "", "Info")
    
    print("\n* * * * * Phase 3 - Case Statistics * * * * *\n")
    stats = cbr_heuristic.get_case_stats()
    printer.pretty("Case Base Statistics:", stats, "Info")
    
    print("\n=== Successfully Ran Case-Based Reasoning Heuristic ===\n")