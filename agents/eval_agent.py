import json
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from evaluators.performance_evaluator import PerformanceEvaluator
from evaluators.efficiency_evaluator import EfficiencyEvaluator
from evaluators.resource_utilization_evaluator import ResourceUtilizationEvaluator

class Evaluator(ABC):
    """Abstract base class for all evaluators."""
    @abstractmethod
    def evaluate(self, agent_outputs: Any, ground_truths: Any) -> float:
        pass

@dataclass
class EvaluationResult:
    """Standardized evaluation result with metadata."""
    score: float
    metric_name: str
    weight: float = 1.0
    explanation: Optional[str] = None
    timestamp: datetime = datetime.now()

class EvaluationAgent:
    def __init__(self, evaluators: Optional[Dict[str, Evaluator]] = None):
        """Initialize with optional custom evaluators."""
        self.evaluators = evaluators or {
            'performance': PerformanceEvaluator(),
            'efficiency': EfficiencyEvaluator(),
            'resource_utilization': ResourceUtilizationEvaluator()
        }
        self.scores: Dict[str, Dict[str, List[EvaluationResult]]] = {}  # Track history
        self.benchmarks: Dict[str, Dict[str, float]] = {}

    # ... (previous methods remain the same until _pareto_ranking) ...

    def _pareto_ranking(self) -> List[Tuple[str, Dict[str, float]]]:
        """
        Implement Pareto optimal ranking using non-dominated sorting.
        
        Returns:
            List of (agent_id, {metric: score}) sorted by Pareto frontier
            
        References:
            - Deb et al. (2002). "A Fast and Elitist Multiobjective Genetic Algorithm"
        """
        if len(self.scores) == 0:
            return []

        # Get current scores (latest evaluation for each agent)
        current_scores = {
            agent_id: {metric: res[-1].score for metric, res in metrics.items()}
            for agent_id, metrics in self.scores.items()
        }

        # Convert to minimization problem (Pareto assumes minimization)
        normalized_scores = {}
        for agent_id, metrics in current_scores.items():
            normalized_scores[agent_id] = {
                metric: -score if self.evaluators[metric].higher_is_better else score
                for metric, score in metrics.items()
            }

        # Non-dominated sorting
        frontiers = []
        remaining = list(normalized_scores.items())
        
        while remaining:
            frontier = []
            new_remaining = []
            
            for i, (agent_id, scores_i) in enumerate(remaining):
                dominated = False
                for j, (_, scores_j) in enumerate(remaining):
                    if i == j:
                        continue
                    if all(scores_j[k] <= scores_i[k] for k in scores_i.keys()):
                        dominated = True
                        break
                
                if not dominated:
                    frontier.append((agent_id, current_scores[agent_id]))
                else:
                    new_remaining.append((agent_id, scores_i))
            
            frontiers.append(frontier)
            remaining = new_remaining

        # Flatten frontiers into single ranked list
        ranked = []
        for frontier in frontiers:
            # Within frontier, sort by hypervolume contribution
            frontier_sorted = sorted(
                frontier,
                key=lambda x: self._hypervolume_contribution(
                    [scores for _, scores in frontier],
                    x[1]
                ),
                reverse=True
            )
            ranked.extend(frontier_sorted)
            
        return ranked

    def _hypervolume_contribution(self, solutions: List[Dict[str, float]], 
                                target: Dict[str, float]]) -> float:
        """
        Calculate hypervolume contribution for a solution.
        
        Args:
            solutions: List of all solutions in the frontier
            target: Target solution to measure contribution for
            
        Returns:
            Hypervolume contribution score
        """
        if not solutions:
            return 0.0
            
        # Convert to numpy array
        metrics = list(target.keys())
        points = np.array([[s[m] for m in metrics] for s in solutions])
        
        # Reference point (nadir point)
        ref = np.max(points, axis=0) + 1
        
        # Calculate hypervolume for full set
        from pymoo.indicators.hv import HV
        hv_full = HV(ref_point=ref)(points)
        
        # Calculate hypervolume without target
        points_without = np.array([
            [s[m] for m in metrics] 
            for s in solutions 
            if not all(np.isclose([s[m] for m in metrics], 
                                [target[m] for m in metrics]))
        ])
        hv_without = HV(ref_point=ref)(points_without) if len(points_without) > 0 else 0
        
        return hv_full - hv_without

    def statistical_significance_test(
        self,
        agent_id_1: str,
        agent_id_2: str,
        metric: str,
        test_type: str = 't-test',
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform statistical significance testing between two agents.
        
        Args:
            agent_id_1: First agent ID
            agent_id_2: Second agent ID
            metric: Metric to compare
            test_type: Type of test ('t-test', 'mann-whitney')
            alpha: Significance level
            
        Returns:
            Dictionary with test results
            
        References:
            - Demšar (2006). "Statistical Comparisons of Classifiers"
        """
        if metric not in self.evaluators:
            raise ValueError(f"Unknown metric: {metric}")
            
        # Get historical scores for the metric
        scores_1 = [r.score for r in self.scores.get(agent_id_1, {}).get(metric, [])]
        scores_2 = [r.score for r in self.scores.get(agent_id_2, {}).get(metric, [])]
        
        if len(scores_1) < 3 or len(scores_2) < 3:
            raise ValueError("Insufficient data for statistical testing")
            
        results = {
            'agent_1': agent_id_1,
            'agent_2': agent_id_2,
            'metric': metric,
            'test': test_type,
            'alpha': alpha,
            'mean_1': np.mean(scores_1),
            'mean_2': np.mean(scores_2),
            'n_1': len(scores_1),
            'n_2': len(scores_2)
        }
        
        if test_type == 't-test':
            # Check normality
            _, p1 = stats.shapiro(scores_1)
            _, p2 = stats.shapiro(scores_2)
            results['normality_p_1'] = p1
            results['normality_p_2'] = p2
            
            # Perform test
            if p1 > alpha and p2 > alpha:
                # Parametric t-test
                _, p_value = stats.ttest_ind(scores_1, scores_2)
                results['p_value'] = p_value
                results['test_used'] = 'independent_t_test'
            else:
                # Non-parametric alternative
                _, p_value = stats.mannwhitneyu(scores_1, scores_2)
                results['p_value'] = p_value
                results['test_used'] = 'mann_whitney_u'
        elif test_type == 'mann-whitney':
            _, p_value = stats.mannwhitneyu(scores_1, scores_2)
            results['p_value'] = p_value
            results['test_used'] = 'mann_whitney_u'
        else:
            raise ValueError(f"Unknown test type: {test_type}")
            
        results['significant'] = results['p_value'] < alpha
        return results

    def visualize_metrics(
        self,
        agent_ids: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        plot_type: str = 'bar',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Generate visualization of evaluation metrics.
        
        Args:
            agent_ids: Agents to include (None for all)
            metrics: Metrics to include (None for all)
            plot_type: Type of plot ('bar', 'radar', 'trend')
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        if not self.scores:
            raise ValueError("No evaluation data available")
            
        agent_ids = agent_ids or list(self.scores.keys())
        metrics = metrics or list(self.evaluators.keys())
        
        # Get latest scores for each agent
        data = {
            agent_id: {
                metric: res[-1].score 
                for metric, res in metrics_dict.items() 
                if metric in metrics
            }
            for agent_id, metrics_dict in self.scores.items()
            if agent_id in agent_ids
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == 'bar':
            # Convert to DataFrame for easier plotting
            df = pd.DataFrame(data).T
            df.plot(kind='bar', ax=ax)
            ax.set_ylabel('Score')
            ax.set_title('Agent Performance Comparison')
            
        elif plot_type == 'radar':
            # Radar chart implementation
            from math import pi
            categories = metrics
            N = len(categories)
            
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]
            
            ax = plt.subplot(111, polar=True)
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            
            plt.xticks(angles[:-1], categories)
            ax.set_rlabel_position(0)
            
            for agent_id, scores in data.items():
                values = list(scores.values())
                values += values[:1]
                ax.plot(angles, values, linewidth=1, linestyle='solid', label=agent_id)
                ax.fill(angles, values, alpha=0.1)
                
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.title('Radar Chart of Agent Performance', y=1.1)
            
        elif plot_type == 'trend':
            # Trend over time for multiple evaluations
            for agent_id in agent_ids:
                for metric in metrics:
                    if metric in self.scores[agent_id]:
                        timestamps = [r.timestamp for r in self.scores[agent_id][metric]]
                        scores = [r.score for r in self.scores[agent_id][metric]]
                        ax.plot(timestamps, scores, 'o-', 
                               label=f"{agent_id} - {metric}")
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_ylabel('Score')
            ax.set_title('Performance Trends Over Time')
            plt.xticks(rotation=45)
            fig.tight_layout()
            
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
            
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            
        return fig

    def track_evaluation_over_time(
        self,
        agent_id: str,
        new_outputs: Any,
        ground_truths: Any,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, EvaluationResult]:
        """
        Record a new evaluation while maintaining historical data.
        
        Args:
            agent_id: Agent to evaluate
            new_outputs: New outputs to evaluate
            ground_truths: Ground truth for comparison
            weights: Optional metric weights
            
        Returns:
            Latest evaluation results
        """
        # Perform new evaluation
        new_results = self.evaluate_agent(agent_id, new_outputs, ground_truths, weights)
        
        # Convert to historical format
        if agent_id not in self.scores:
            self.scores[agent_id] = {k: [] for k in self.evaluators.keys()}
            
        for metric, result in new_results.items():
            self.scores[agent_id][metric].append(result)
            
        return new_results

    def get_performance_history(
        self,
        agent_id: str,
        metric: str
    ) -> List[Tuple[datetime, float]]:
        """
        Retrieve historical performance data for an agent.
        
        Args:
            agent_id: Agent to query
            metric: Metric to retrieve
            
        Returns:
            List of (timestamp, score) tuples
        """
        if agent_id not in self.scores or metric not in self.scores[agent_id]:
            return []
            
        return [(r.timestamp, r.score) for r in self.scores[agent_id][metric]]
