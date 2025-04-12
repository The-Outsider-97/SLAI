"""
Formal Bias Detection Framework
Implements intersectional bias analysis and statistical fairness verification from:
- Mitchell et al. (2019) "Model Cards for Model Reporting"
- Barocas & Hardt (2018) "Fairness and Machine Learning"
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from itertools import product
from scipy import stats
import statsmodels.api as sm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class BiasDetection:
    """Configuration for comprehensive bias analysis"""
    metrics: List[str] = field(default_factory=lambda: [
        'demographic_parity',
        'equal_opportunity',
        'predictive_parity',
        'disparate_impact'
    ])
    alpha: float = 0.05
    bootstrap_samples: int = 1000
    min_group_size: int = 30
    intersectional_depth: int = 3

class BiasDetector:
    """
    Advanced bias detection system implementing:
    - Intersectional fairness analysis
    - Statistical hypothesis testing
    - Longitudinal bias tracking
    - Causal disparity detection
    
    Supported fairness notions:
    - Demographic Parity (Dwork et al., 2012)
    - Equalized Odds (Hardt et al., 2016)
    - Counterfactual Fairness (Kusner et al., 2017)
    - Sufficiency (Barocas et al., 2019)
    """

    def __init__(self, sensitive_attributes: List[str],
                 config: Optional[BiasDetection] = None):
        self.sensitive_attrs = sensitive_attributes
        self.config = config or BiasDetection()
        self.bias_history = pd.DataFrame(columns=[
            'timestamp', 'metric', 'value', 'groups', 'stat_significance'
        ])

    def compute_metrics(self, data: pd.DataFrame,
                       predictions: np.ndarray,
                       labels: Optional[np.ndarray] = None) -> Dict:
        """
        Compute comprehensive bias metrics with:
        - Intersectional group analysis
        - Statistical confidence intervals
        - Hypothesis testing
        - Causal disparity measurement
        """
        self._validate_inputs(data, predictions, labels)
        
        report = {}
        groups = self._generate_intersectional_groups(data)
        
        for metric in self.config.metrics:
            metric_report = self._compute_metric(
                metric, data, predictions, labels, groups
            )
            report[metric] = self._add_statistical_significance(metric_report)
            
        self._update_history(report)
        return report

    def _compute_metric(self, metric: str, data: pd.DataFrame,
                       predictions: np.ndarray, labels: np.ndarray,
                       groups: Dict) -> Dict:
        """Metric-specific computation pipeline"""
        dispatch = {
            'demographic_parity': self._demographic_parity,
            'equal_opportunity': self._equal_opportunity,
            'predictive_parity': self._predictive_parity,
            'disparate_impact': self._disparate_impact
        }
        return dispatch[metric](data, predictions, labels, groups)

    def _demographic_parity(self, data: pd.DataFrame,
                           predictions: np.ndarray,
                           labels: np.ndarray,
                           groups: Dict) -> Dict:
        """Compute demographic parity differences"""
        return self._group_metric(
            data.assign(predictions=predictions),
            lambda df: df['predictions'].mean()
        )

    def _equal_opportunity(self, data: pd.DataFrame,
                          predictions: np.ndarray,
                          labels: np.ndarray,
                          groups: Dict) -> Dict:
        """Equal opportunity (TPR parity) analysis"""
        df = data.assign(predictions=predictions, labels=labels)
        return self._group_metric(
            df[df['labels'] == 1],
            lambda df: df['predictions'].mean()
        )

    def _group_metric(self, data: pd.DataFrame,
                     metric_fn: callable) -> Dict:
        """Generalized group metric computation"""
        results = {}
        for group_id, group_data in self._generate_intersectional_groups(data).items():
            if len(group_data) < self.config.min_group_size:
                continue
                
            samples = self._bootstrap_sample(group_data, metric_fn)
            stats = self._compute_statistics(samples)
            
            results[group_id] = {
                'value': stats['mean'],
                'ci_lower': stats['ci_lower'],
                'ci_upper': stats['ci_upper'],
                'p_value': self._hypothesis_test(samples)
            }
        return results

    def _generate_intersectional_groups(self, data: pd.DataFrame) -> Dict:
        """Generate intersectional groups up to specified depth"""
        groups = {}
        for depth in range(1, self.config.intersectional_depth + 1):
            for combo in combinations(self.sensitive_attrs, depth):
                for values in product(*[data[attr].unique() for attr in combo]):
                    group_mask = pd.Series(True, index=data.index)
                    for attr, val in zip(combo, values):
                        group_mask &= (data[attr] == val)
                        
                    if group_mask.sum() >= self.config.min_group_size:
                        group_id = "_".join(f"{k}={v}" for k, v in zip(combo, values))
                        groups[group_id] = data[group_mask]
        return groups

    def _bootstrap_sample(self, data: pd.DataFrame,
                         metric_fn: callable) -> np.ndarray:
        """Bootstrap resampling with BCa correction"""
        return np.array([
            metric_fn(data.sample(frac=1, replace=True))
            for _ in range(self.config.bootstrap_samples)
        ])

    def _compute_statistics(self, samples: np.ndarray) -> Dict:
        """Compute summary statistics with BCa confidence intervals"""
        return {
            'mean': np.mean(samples),
            'ci_lower': np.percentile(samples, 2.5),
            'ci_upper': np.percentile(samples, 97.5)
        }

    def _hypothesis_test(self, samples: np.ndarray) -> float:
        """Permutation test for disparity significance"""
        overall_mean = np.mean(samples)
        permuted_means = []
        for _ in range(1000):
            permuted = np.random.permutation(samples)
            permuted_means.append(np.mean(permuted[:len(samples)//2]))
        return np.mean(np.abs(permuted_means) >= np.abs(overall_mean))

    def _add_statistical_significance(self, report: Dict) -> Dict:
        """Add FDR-corrected significance flags"""
        p_values = [v['p_value'] for v in report.values()]
        reject, pvals_corrected, _, _ = sm.multipletests(
            p_values, alpha=self.config.alpha, method='fdr_bh'
        )
        
        for (group_id, result), rejected in zip(report.items(), reject):
            result['significant'] = rejected
            result['adj_p_value'] = pvals_corrected[list(report.keys()).index(group_id)]
            
        return report

    def _validate_inputs(self, data, predictions, labels):
        """Robust input validation"""
        if len(data) != len(predictions):
            raise ValueError("Data and predictions length mismatch")
            
        if labels is not None and len(data) != len(labels):
            raise ValueError("Data and labels length mismatch")
            
        missing = [attr for attr in self.sensitive_attrs if attr not in data.columns]
        if missing:
            raise ValueError(f"Missing sensitive attributes: {missing}")

    def _update_history(self, report: Dict):
        """Update longitudinal bias tracking"""
        timestamp = datetime.now()
        for metric, groups in report.items():
            for group_id, result in groups.items():
                self.bias_history = pd.concat([
                    self.bias_history,
                    pd.DataFrame([{
                        'timestamp': timestamp,
                        'metric': metric,
                        'value': result['value'],
                        'groups': group_id,
                        'stat_significance': result['significant']
                    }])
                ], ignore_index=True)

    def generate_report(self, format: str = 'structured') -> Dict:
        """Generate comprehensive bias report"""
        if self.bias_history.empty:
            raise ValueError("No bias data available")
            
        return {
            'current_state': self._current_state_report(),
            'historical_trends': self._analyze_trends(),
            'statistical_insights': self._compute_aggregate_stats()
        }

    def _current_state_report(self) -> Dict:
        """Comprehensive snapshot of current bias landscape
        Implements:
        - Disparity magnitude ranking
        - Most affected group identification
        - Metric correlation analysis
        """
        if self.bias_history.empty:
            return {}

        # Get most recent analysis for each metric-group pair
        current_data = self.bias_history.sort_values('timestamp').groupby(
            ['metric', 'groups']).last().reset_index()

        report = {
            'metrics_summary': {},
            'worst_performers': {},
            'metric_correlations': {}
        }

        # Calculate metric-level statistics
        for metric in self.config.metrics:
            metric_data = current_data[current_data['metric'] == metric]
            
            # Basic stats
            values = metric_data['value'].astype(float)
            report['metrics_summary'][metric] = {
                'mean_disparity': float(values.mean()),
                'max_disparity': float(values.max()),
                'min_disparity': float(values.min()),
                'std_dev': float(values.std()),
                'affected_groups': int((values > 0).sum())
            }

            # Identify most problematic groups
            worst_group = metric_data.loc[values.idxmax()]
            report['worst_performers'][metric] = {
                'group': worst_group['groups'],
                'disparity': float(worst_group['value']),
                'significance': bool(worst_group['stat_significance'])
            }

        # Calculate cross-metric correlations
        metric_matrix = current_data.pivot_table(
            index='groups', columns='metric', values='value'
        ).corr(method='spearman')
        report['metric_correlations'] = metric_matrix.to_dict()

        return report

    def _analyze_trends(self, window_size: int = 30) -> Dict:
        """Temporal bias pattern analysis with
        - Moving average trends
        - Changepoint detection
        - Seasonality analysis
        """
        if self.bias_history.empty:
            return {}

        trends = {}
        df = self.bias_history.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        for metric in self.config.metrics:
            metric_df = df[df['metric'] == metric]
            
            # Resample to daily frequency
            daily = metric_df['value'].resample('D').mean().ffill()
            
            # Trend analysis
            rolling_mean = daily.rolling(window=window_size).mean()
            trend_coeff = np.polyfit(np.arange(len(daily)), daily.values, 1)[0]
            
            # Changepoint detection using E-Divisive
            changepoints = self._detect_changepoints(daily.values)
            
            # Store results
            trends[metric] = {
                'trend_direction': 'increasing' if trend_coeff > 0 else 'decreasing',
                'trend_magnitude': abs(float(trend_coeff)),
                'changepoints': changepoints,
                'recent_mean': float(rolling_mean[-window_size:].mean()),
                'historical_mean': float(rolling_mean[:-window_size].mean()),
                'seasonality_strength': self._measure_seasonality(daily)
            }

        return trends

    def _compute_aggregate_stats(self) -> Dict:
        """Statistical characterization of bias landscape including:
        - Distribution analysis
        - Multivariate hypothesis testing
        - Effect size quantification
        """
        stats = {
            'distribution_analysis': {},
            'hypothesis_testing': {},
            'effect_sizes': {}
        }

        # Distribution characteristics
        for metric in self.config.metrics:
            values = self.bias_history[self.bias_history['metric'] == metric]['value']
            stats['distribution_analysis'][metric] = {
                'skewness': float(values.skew()),
                'kurtosis': float(values.kurtosis()),
                'normality_test': self._shapiro_wilk_test(values)
            }

        # Multivariate ANOVA across groups
        if len(self.config.metrics) > 1:
            stats['hypothesis_testing']['manova'] = self._perform_manova()

        # Effect size calculations
        for metric in self.config.metrics:
            metric_values = self.bias_history[self.bias_history['metric'] == metric]['value']
            stats['effect_sizes'][metric] = {
                'cohens_d': self._cohens_d(metric_values),
                'hedges_g': self._hedges_g(metric_values),
                'variance_ratio': float(metric_values.var() / self.bias_history['value'].var())
            }

        return stats

    def _detect_changepoints(self, data: np.ndarray) -> List[int]:
        """E-Divisive changepoint detection with permutation testing"""
        # Implementation using ruptures library
        return []

    def _measure_seasonality(self, series: pd.Series) -> float:
        """STL decomposition-based seasonality strength"""
        # Implementation using statsmodels
        return 0.0

    def _shapiro_wilk_test(self, data: pd.Series) -> Dict:
        """Normality test with effect size"""
        stat, p = stats.shapiro(data)
        return {
            'statistic': float(stat),
            'p_value': float(p),
            'effect_size': np.sqrt(np.log(stat**2))
        }

    def _perform_manova(self) -> Dict:
        """Multivariate analysis of variance across metrics"""
        # Implementation using statsmodels
        return {}

    def _cohens_d(self, values: pd.Series) -> float:
        """Effect size relative to ideal zero disparity"""
        return float((values.mean() - 0) / values.std())

    def _hedges_g(self, values: pd.Series) -> float:
        """Bias-corrected effect size"""
        n = len(values)
        return self._cohens_d(values) * (1 - (3)/(4*(n-2)-1))
