
import json
import sys
import numpy as np

from functools import lru_cache
from types import SimpleNamespace
from collections import defaultdict
from typing import Any, Dict, List
from scipy.stats import shapiro, t as student_t, f as f_dist, norm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, log_loss

from src.agents.evaluators.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Evaluators Calculations")
printer = PrettyPrinter

class EvaluatorsCalculations:
    def __init__(self):
        self.config = load_global_config()
        self.avg_token_baseline = self.config.get('avg_token_baseline')

        self.base_config = get_config_section('baselines')
        self.time_per_output = self.base_config.get('time_per_output')
        self.time_baseline = self.base_config.get('time_baseline')
        self.memory_baseline = self.base_config.get('memory_baseline')
        self.flops = self.base_config.get('flops')

        self.perform_config = get_config_section('performance_evaluator')
        self.average = self.perform_config.get('average')
        self.classes = self.perform_config.get('classes')
        self.metric_weights = self.perform_config.get('weights', {})
        self.zero_division = self.perform_config.get('zero_division', 
                                self.base_config.get('zero_division', 'warn'))

        self.rue_config = get_config_section('resource_utilization_evaluator')
        self.thresholds = self.rue_config.get('thresholds', {})

        self.statistic_config = get_config_section('statistical_evaluator')
        self.significance_threshold = self.statistic_config.get('alpha')
        self.min_sample_size = self.statistic_config.get('min_sample_size')

        self.eval_config = get_config_section('safety_evaluator')
        self.risk_categories = self.eval_config.get('risk_categories', [
            'collision', 'pinch_point', 'crush_hazard',
            'electrical', 'environmental', 'control_failure'
        ])

        self.constraints = {}
        self.violation_history = []
        self._lagrangian_multipliers = {'safety': 0.1, 'ethics': 0.2}
        self.hazard_data = []
        self.tokenizer = None
        self.resource = None
        self.nlp_engine = None
        current_flops=None
        self.current_flops = current_flops

    def _calculate_linguistic_complexity(self, outputs: List[Any]) -> Dict[str, float]:
        """Analyze text complexity using NLP Engine"""
        complexity = {
            'avg_sentence_length': 0.0,
            'pos_diversity': 0.0,
            'dependency_complexity': 0.0,
            'entity_density': 0.0
        }
        
        if not self.nlp_engine:
            return complexity

        total_sentences = 0
        pos_counts = defaultdict(int)
        total_dependencies = 0
        total_entities = 0
        total_tokens = 0

        for output in outputs:
            if not isinstance(output, str):
                continue
                
            try:
                # Process text through full NLP pipeline
                tokens = self.nlp_engine.process_text(output)
                sentences = [tokens]  # Simple sentence split
                deps = self.nlp_engine.apply_dependency_rules(tokens)
                entities = self.nlp_engine.resolve_coreferences(sentences)

                # Update metrics
                total_sentences += len(sentences)
                total_tokens += len(tokens)
                total_dependencies += len(deps)
                total_entities += len(entities)
                
                for token in tokens:
                    pos_counts[token.pos] += 1

            except Exception as e:
                logger.warning(f"Complexity analysis failed for output: {e}")

        # Calculate final metrics
        if total_sentences > 0:
            complexity['avg_sentence_length'] = total_tokens / total_sentences
            
        if pos_counts:
            complexity['pos_diversity'] = len(pos_counts) / total_tokens
            
        if total_tokens > 0:
            complexity['dependency_complexity'] = total_dependencies / total_tokens
            complexity['entity_density'] = total_entities / total_tokens

        return complexity

    def _calculate_token_efficiency(self, outputs: List[Any]) -> float:
        """Token efficiency with fallback to word count"""
        total_tokens = 0
        
        for o in outputs:
            text = str(o)
            total_tokens += len(text.split())  # Simple word count fallback
        
        avg_tokens = total_tokens / len(outputs) if outputs else 0
        baseline = self.avg_token_baseline
        return baseline / (avg_tokens + sys.float_info.epsilon)

    def _calculate_temporal(self, outputs: List[Any]) -> float:
        """Response latency efficiency using generation time metadata."""
        # Fallback if no temporal data exists
        total_time = 0.0
        valid_outputs = 0
        
        for output in outputs:
            if isinstance(output, dict) and 'generation_time' in output:
                total_time += output['generation_time']
                valid_outputs += 1
        
        if valid_outputs == 0:
            logger.warning("Temporal efficiency: No generation_time metadata found")
            return 0.0

        # Get baseline from config
        time_per_output = self.time_per_output
        time_baseline = self.time_baseline
        baseline = time_baseline if time_baseline else time_per_output * valid_outputs
            
        # Calculate efficiency ratio with smoothing
        return baseline / (total_time + sys.float_info.epsilon)

    def _calculate_spatial(self, outputs: List[Any]) -> float:
        """Memory footprint efficiency with serialization."""
        total_size = 0
        content_types = defaultdict(int)
        
        for output in outputs:
            # Prefer metadata if available
            if isinstance(output, dict) and 'serialized_size' in output:
                total_size += output['serialized_size']
                content_types[output.get('content_type', 'unknown')] += 1
                continue
                
            try:
                # Serialize and measure payload size
                if isinstance(output, (str, bytes)):
                    serialized = output
                else:
                    serialized = json.dumps(output).encode('utf-8')
                total_size += len(serialized)
                content_types['structured'] += 1
            except (TypeError, OverflowError):
                # Fallback for non-serializable objects
                total_size += sys.getsizeof(output)
                content_types['binary'] += 1
        
        # Get baseline
        baseline = self.memory_baseline  # Default 1KB
        
        # Calculate efficiency ratio
        return baseline / (total_size + sys.float_info.epsilon)

    def _calculate_computational(self):
        """FLOPs relative to baseline"""
        baseline = float(self.flops)
        current = float(self.current_flops or baseline)
        return baseline / (current + sys.float_info.epsilon)

    def _calculate_composite_score(self, results) -> float:
        """Context-aware scoring with linguistic weights"""

        # Get metric_weights from config if not available
        if not hasattr(self, 'metric_weights'):
            perf_config = get_config_section('performance_evaluator')
            self.metric_weights = perf_config.get('weights', {})
        
        return sum(
            self.metric_weights.get(metric, 0) * value 
            for metric, value in results.items()
            if metric in self.metric_weights
        )

    def _calculate_accuracy(self, outputs, truths):
        # Handle probabilistic outputs
        if isinstance(outputs[0], (list, np.ndarray)):
            outputs = [np.argmax(o) for o in outputs]
        correct = sum(o == g for o, g in zip(outputs, truths))
        return correct / len(truths) if truths else 0.0

    def _calculate_precision(self, outputs, truths):
        return precision_score(truths, outputs, 
                             average=self.average, 
                             zero_division=self.zero_division)

    def _calculate_recall(self, outputs, truths):
        return recall_score(truths, outputs,
                          average=self.average,
                          zero_division=0)

    def _calculate_f1(self, outputs, truths):
        return f1_score(truths, outputs,
                       average=self.average,
                       zero_division=0)

    #@lru_cache(maxsize=128)
    def _calculate_confusion_matrix(self, outputs, truths):
        if self.classes is not None:
            return confusion_matrix(truths, outputs, labels=self.classes).tolist()
        return confusion_matrix(truths, outputs).tolist()

    def _calculate_composite_score(self, results):
        return sum(
            self.metric_weights.get(metric, 0) * value 
            for metric, value in results.items()
            if metric in self.metric_weights
        )

    def _calculate_roc_auc(self, outputs, truths):
        """Calculate ROC AUC score with multi-class support"""
        try:
            if len(np.unique(truths)) > 2:
                return roc_auc_score(truths, outputs, multi_class='ovo')
            return roc_auc_score(truths, outputs)
        except ValueError as e:
            logger.warning(f"ROC AUC calculation failed: {e}")
            return 0.0
    
    def _calculate_log_loss(self, outputs, truths):
        """Calculate logarithmic loss"""
        try:
            return log_loss(truths, outputs)
        except (ValueError, TypeError) as e:
            logger.warning(f"Log loss calculation failed: {e}")
            return float('inf')
    
    def _calculate_adaptive_weights(self, ground_truths):
        class_counts = np.bincount(ground_truths)
        weights = 1. / class_counts
        return weights / weights.sum()
    
    def _calculate_scores(self, metrics: Dict) -> Dict[str, float]:
        """Calculate normalized resource efficiency scores"""
        return {
            'cpu': max(0, 1 - metrics['cpu']/self.thresholds['cpu']),
            'memory': max(0, 1 - metrics['memory']/self.thresholds['memory']),
            'disk': max(0, 1 - metrics['disk']/self.thresholds['disk']),
            'network': max(0, 1 - metrics['network']/self.thresholds['network'])
        }

    def _calculate_descriptive_stats(self, datasets: Dict[str, List[float]]) -> Dict[str, Dict]:
        """Calculate basic descriptive statistics for all datasets"""
        stats = {}
        for name, data in datasets.items():
            if len(data) < self.min_sample_size:
                logger.warning(f"Insufficient sample size for {name}: {len(data)}")
                continue
                
            stats[name] = {
                'mean': np.mean(data),
                'median': np.median(data),
                'std_dev': np.std(data, ddof=1),
                'min': np.min(data),
                'max': np.max(data),
                'sample_size': len(data)
            }
        return stats

    def _calculate_all_effect_sizes(self, datasets: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate effect sizes between all dataset pairs"""
        effects = {}
        names = list(datasets.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                key = f"{names[i]}_vs_{names[j]}"
                effects[key] = self.effect_size(
                    datasets[names[i]], datasets[names[j]])
                
        return effects

    def _calculate_power_analysis(self, datasets: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate statistical power for all dataset comparisons"""
        power = {}
        for name, data in datasets.items():
            effect = self.effect_size(data, [0])  # Compare against zero
            power[name] = self._calculate_power(len(data), effect)
        return power

    def _calculate_power(self, n: int, effect_size: float) -> float:
        """Calculate statistical power using normal approximation"""
        alpha = self.significance_threshold
        z_alpha = norm.ppf(1 - alpha/2)
        z_power = effect_size * np.sqrt(n) - z_alpha
        return norm.cdf(z_power)

    def effect_size(self, sample_a, sample_b):
        mean_a, mean_b = np.mean(sample_a), np.mean(sample_b)
        pooled_std = np.sqrt((np.std(sample_a, ddof=1)**2 + np.std(sample_b, ddof=1)**2) / 2)
        return 0.0 if pooled_std == 0 else (mean_a - mean_b) / pooled_std

    def _calculate_risk_distribution(self) -> Dict[str, float]:
        """Calculate risk distribution across categories"""
        distribution = {category: 0.0 for category in self.risk_categories}
        total_weight = 0
        
        for hazard in self.hazard_data:
            for category in self.risk_categories:
                if category in hazard:
                    distribution[category] += hazard[category]
                    total_weight += hazard[category]
        
        if total_weight > 0:
            for category in distribution:
                distribution[category] /= total_weight
                
        return distribution

    # Debt calculator
    # Weights from Baev et al.'s empirical study
    DEBT_WEIGHTS = {
        "nested_control": 0.3,
        "nested_loop": 0.3,
        "duplicate_code": 0.4,
        "violation_of_law_of_demeter": 0.2,
        "security_risk": 0.5
    }  

    def calculate_debt(self, issues: List[Dict]) -> float:  
        """Compute technical debt score using weighted sum"""  
        return sum(  
            issue["severity"] * self.DEBT_WEIGHTS.get(issue["type"], 0.1)  
            for issue in issues  
        )  

    def prioritize_remediation(self, issues: List[Dict]) -> List[Dict]:  
        """Order fixes by cost/benefit ratio (Baev Eq. 4.2)"""  
        return sorted(  
            issues,  
            key=lambda x: (  
                x["severity"] * self.DEBT_WEIGHTS[x["type"]] /  
                max(1, x.get("estimated_fix_time", 1))  
            ),  
            reverse=True  
        )

    # Reward calculator
    # Lagrangian optimization from Chow et al. (2017)
    def calculate_reward(self, state: Dict, action: Dict, outcome: Dict) -> float:
        """Multi-objective reward calculation with dynamic penalties"""
        base_reward = outcome.get('performance', 0.0)
        
        # Constraint calculations
        safety_penalty = self._calculate_safety_violation(outcome)
        ethics_penalty = self._calculate_ethical_violation(action)
        
        # Lagrangian formulation
        constrained_reward = base_reward \
            - self._lagrangian_multipliers['safety'] * safety_penalty \
            - self._lagrangian_multipliers['ethics'] * ethics_penalty
            
        self._update_multipliers(safety_penalty, ethics_penalty)
        return constrained_reward

    def _calculate_safety_violation(self, outcome: Dict) -> float:
        """Physical safety metrics using Hamilton-Jacobi reachability"""
        return max(0, outcome.get('hazard_prob', 0) - self.constraints['safety_tolerance'])

    def _calculate_ethical_violation(self, action: Dict) -> float:
        """Ethical penalty using Veale et al. (2018) bias metrics"""
        return sum(
            1 for pattern in self.constraints['ethical_patterns']
            if pattern in action['decision_path']
        )

    def _update_multipliers(self, safety_viol: float, ethics_viol: float):
        """Dual gradient descent update rule"""
        self._lagrangian_multipliers['safety'] *= (1 + safety_viol)
        self._lagrangian_multipliers['ethics'] *= (1 + ethics_viol)

    # ===== Evaluation Agent ===== #

    def _calculate_composite_score(self, results: Dict) -> float:
        """Weighted scoring with safety constraints"""
        weights = self.metric_weights
        return (
            weights.get("accuracy", 0.6) * results["accuracy"] +
            weights.get("safety", 0.3) * results["safety_score"] +
            weights.get("efficiency", 0.1) * (1 - results["resource_usage"])
        )
    
    def calculate_diagnostic_coverage(self) -> Dict[str, Any]:
        """
        Computes diagnostic coverage metrics based on logged hazard detection data.
    
        Returns:
            Dict[str, Any]: A dictionary with category-level and overall diagnostic coverage scores.
        """
        if not self.hazard_data:
            logger.warning("No hazard data available for diagnostic coverage calculation.")
            return {
                "coverage": 0.0,
                "details": {},
                "status": "No data"
            }
    
        coverage_by_category = {category: {"detected": 0, "total": 0} for category in self.risk_categories}
    
        for hazard in self.hazard_data:
            for category in self.risk_categories:
                if category in hazard:
                    coverage_by_category[category]["total"] += 1
                    if hazard[category] >= 1.0:  # Convention: 1.0 indicates detected/diagnosed
                        coverage_by_category[category]["detected"] += 1
    
        results = {}
        total_detected = 0
        total_possible = 0
    
        for category, stats in coverage_by_category.items():
            total = stats["total"]
            detected = stats["detected"]
            coverage = detected / total if total > 0 else 0.0
            results[category] = round(coverage, 3)
            total_detected += detected
            total_possible += total
    
        overall_coverage = total_detected / total_possible if total_possible > 0 else 0.0
    
        return {
            "coverage": round(overall_coverage, 3),
            "details": results,
            "status": "OK" if overall_coverage >= 0.85 else "Insufficient"
        }