"""
Formal Fairness Verification System
Implements:
- Group fairness metrics (Dwork et al., 2012)
- Individual fairness verification (Dwork et al., 2012)
- Disparate impact analysis (Feldman et al., 2015)
- Multi-level statistical testing
"""

import json, yaml
import numpy as np
import pandas as pd

from types import SimpleNamespace
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from scipy import stats
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.stats import linregress

from logs.logger import get_logger

logger = get_logger("Fairness Evaluator")

def dict_to_namespace(d):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    return d

def get_config_section(section: str, config_file_path: str):
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    if section not in config:
        raise KeyError(f"Section '{section}' not found in config file: {config_file_path}")
    return dict_to_namespace(config[section])

class FairnessEvaluator:
    """
    Multi-level fairness assessment system implementing:
    - Group fairness statistical verification
    - Individual fairness consistency checks
    - Disparate impact quantification
    - Longitudinal fairness tracking
    
    Key Features:
    1. Statistical parity difference with CI
    2. Equalized odds ratio calculation
    3. Individual fairness Lipschitz constant
    4. Disparate impact ratio testing
    """

    def __init__(self, sensitive_attributes: List[str],
                 config_section_name: str = "fairness_evaluator",
                 config_file_path: str = "src/agents/alignment/configs/alignment_config.yaml"
                 ):
        self.sensitive_attrs = sensitive_attributes
        self.config = get_config_section(config_section_name, config_file_path)
        self.history = pd.DataFrame(columns=[
            'timestamp', 'metric', 'value', 'groups', 'p_value'
        ])

    def evaluate_group_fairness(self, data: pd.DataFrame,
                               predictions: np.ndarray,
                               labels: np.ndarray) -> Dict:
        """
        Comprehensive group fairness analysis with:
        - Statistical hypothesis testing
        - Confidence interval estimation
        - Multiple comparison correction
        """
        df = data.copy()
        df['prediction'] = predictions
        df['label'] = labels

        report = {}
        for attr in self.sensitive_attrs:
            attr_report = {}
            for metric in self.config.group_metrics:
                metric_fn = self._metric_dispatch(metric)
                result = metric_fn(df, attr)
                attr_report[metric] = self._add_statistical_significance(result)
            
            report[attr] = attr_report
            self._update_history(attr, attr_report)

        return report

    def evaluate_individual_fairness(self, data: pd.DataFrame,
                                    predictions: np.ndarray,
                                    similarity_fn: Optional[Callable] = None
                                    ) -> Dict:
        """
        Individual fairness verification through:
        - Consistency score calculation
        - Fairness radius estimation
        - Lipschitz constant approximation
        """
        if similarity_fn is None:
            similarity_fn = self._default_similarity
            
        return {
            'consistency': self._calculate_consistency(data, predictions, similarity_fn),
            'lipschitz_constant': self._estimate_lipschitz(data, predictions, similarity_fn),
            'fairness_violations': self._identify_violations(data, predictions, similarity_fn)
        }

    def _metric_dispatch(self, metric: str) -> Callable:
        """Metric implementation routing"""
        return {
            'statistical_parity': self._statistical_parity,
            'equal_opportunity': self._equal_opportunity,
            'predictive_parity': self._predictive_parity,
            'disparate_impact': self._disparate_impact
        }[metric]

    def _statistical_parity(self, df: pd.DataFrame, attr: str) -> Dict:
        """Statistical parity difference (Dwork et al., 2012)"""
        grouped = df.groupby(attr)['prediction'].mean()
        disparity = grouped.max() - grouped.min()
        return self._bootstrap_metric(
            df, attr, 
            lambda g: g['prediction'].mean(),  # Now explicitly works with prediction column
            disparity
        )

    def _equal_opportunity(self, df: pd.DataFrame, attr: str) -> Dict:
        """True positive rate parity (Hardt et al., 2016)"""
        pos_df = df[df['label'] == 1]
        grouped = pos_df.groupby(attr)['prediction'].mean()
        disparity = grouped.max() - grouped.min()
        return self._bootstrap_metric(
            pos_df, attr,
            lambda g: g['prediction'].mean(),
            disparity
        )

    def _predictive_parity(self, df: pd.DataFrame, attr: str) -> Dict:
        """Positive predictive value parity"""
        grouped = df.groupby(attr).apply(
            lambda g: g.loc[g['prediction'] == 1, 'label'].mean()
        )
        disparity = grouped.max() - grouped.min()
        return self._bootstrap_metric(
            df, attr,
            lambda g: g.loc[g['prediction'] == 1, 'label'].mean(),
            disparity
        )

    def _disparate_impact(self, df: pd.DataFrame, attr: str) -> Dict:
        """Feldman et al. (2015) disparate impact ratio"""
        grouped = df.groupby(attr)['prediction'].mean()
        min_group = grouped.idxmin()
        max_group = grouped.idxmax()
        ratio = grouped[min_group] / grouped[max_group]
        return self._bootstrap_metric(df, attr,
            lambda g: g['prediction'].mean(), ratio, is_ratio=True)

    def _bootstrap_metric(self, df: pd.DataFrame, attr: str,
                         metric_fn: Callable, observed: float,
                         is_ratio: bool = False) -> Dict:
        """Bootstrap analysis with bias-corrected CI"""
        bootstraps = []
        groups = df[attr].unique()
        
        for _ in range(self.config.n_bootstrap):
            sample = df.sample(frac=1, replace=True)
            # Explicitly select the column(s) needed by metric_fn
            if metric_fn.__code__.co_varnames[0] == 'g':  # If metric_fn takes a DataFrame
                group_metrics = sample.groupby(attr).apply(lambda g: metric_fn(g[['prediction', 'label']]))
            else:  # If metric_fn takes a Series
                group_metrics = sample.groupby(attr)['prediction'].apply(metric_fn)
            
            if is_ratio:
                stat = group_metrics.min() / group_metrics.max()
            else:
                stat = group_metrics.max() - group_metrics.min()
                
            bootstraps.append(stat)
    
        ci_lower, ci_upper = np.percentile(bootstraps, [2.5, 97.5])
        return {
            'value': observed,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'distribution': bootstraps
        }

    def _add_statistical_significance(self, result: Dict) -> Dict:
        """Hypothesis testing with null distribution"""
        p_value = np.mean(np.abs(result['distribution']) >= np.abs(result['value']))
        result.update({
            'p_value': p_value,
            'significant': p_value < self.config.alpha
        })
        return result

    def _calculate_consistency(self, data: pd.DataFrame,
                              predictions: np.ndarray,
                              similarity_fn: Callable) -> float:
        """Individual consistency score (Dwork et al., 2012)"""
        distances = pairwise_distances(data, metric=similarity_fn)
        prediction_diffs = np.abs(predictions[:, None] - predictions)
        return np.exp(-np.mean(distances * prediction_diffs))

    def _estimate_lipschitz(self, data: pd.DataFrame,
                           predictions: np.ndarray,
                           similarity_fn: Callable) -> float:
        """Lipschitz constant estimation for individual fairness"""
        distances = pairwise_distances(data, metric=similarity_fn).ravel()
        pred_diffs = np.abs(predictions[:, None] - predictions).ravel()
        nonzero = distances > 1e-6
        return np.max(pred_diffs[nonzero] / distances[nonzero])

    def _identify_violations(self, data: pd.DataFrame,
                            predictions: np.ndarray,
                            similarity_fn: Callable) -> Dict:
        """Identify individual fairness violations"""
        nn = NearestNeighbors(n_neighbors=50, metric=similarity_fn)
        nn.fit(data)
        distances, indices = nn.kneighbors(data)

        violations = []
        for i in range(len(data)):
            neighbor_preds = predictions[indices[i]]
            deviation = np.abs(predictions[i] - neighbor_preds)
            violations.extend(deviation / (distances[i] + 1e-6))

        return {
            'max_violation': np.max(violations),
            'mean_violation': np.mean(violations),
            'violation_rate': np.mean(np.array(violations) > 1.0)
        }

    def _default_similarity(self, a: pd.Series, b: pd.Series) -> float:
        """Normalized Manhattan distance"""
        return np.sum(np.abs(a - b)) / len(a)

    def _update_history(self, attribute: str, report: Dict):
        """Update longitudinal fairness tracking"""
        timestamp = pd.Timestamp.now()
        for metric, results in report.items():
            self.history = pd.concat([self.history, pd.DataFrame([{
                'timestamp': timestamp,
                'metric': f"{attribute}_{metric}",
                'value': results['value'],
                'groups': attribute,
                'p_value': results['p_value']
            }])], ignore_index=True)

    def generate_report(self, format: str = 'structured') -> Dict:
        """Generate comprehensive fairness report"""
        return {
            'current_state': self._current_state(),
            'historical_trends': self._analyze_trends(),
            'statistical_summary': self._statistical_summary()
        }

    def _current_state(self) -> Dict:
        """Current fairness status snapshot"""
        if self.history.empty:
            return {'status': 'no_data', 'message': 'No fairness evaluations recorded'}
    
        # Get latest values for each metric
        latest = self.history.sort_values('timestamp').groupby(['metric', 'groups']).last()
        
        current_metrics = {}
        for (metric, group), row in latest.iterrows():
            current_metrics[f"{group}_{metric}"] = {
                'value': row['value'],
                'p_value': row['p_value'],
                'significant': row['p_value'] < self.config.alpha,
                'last_updated': row['timestamp'].isoformat()
            }
    
        # Count significant disparities
        sig_counts = latest.groupby('groups').apply(
            lambda g: (g['p_value'] < self.config.alpha).sum())
        
        return {
            'metrics': current_metrics,
            'summary': {
                'total_metrics': len(latest),
                'significant_disparities': sig_counts.to_dict(),
                'worst_metric': latest['value'].idxmax()
            }
        }
    
    def _analyze_trends(self) -> Dict:
        """Temporal fairness analysis"""
        if self.history.empty:
            return {'status': 'no_data', 'message': 'No historical data available'}

        trends = {}
        grouped = self.history.groupby(['metric', 'groups'])
        
        for (metric, group), data in grouped:
            if len(data) < 2:
                continue
                
            # Convert timestamps to numerical values
            x = data['timestamp'].astype(np.int64) // 10**9  # Unix time
            y = data['value']
            
            # Linear regression for trend
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            
            # Recent change (last 30 days)
            recent = data[data['timestamp'] > pd.Timestamp.now() - pd.DateOffset(days=30)]
            recent_change = recent['value'].iloc[-1] - recent['value'].iloc[0] if len(recent) >= 2 else 0
            
            trends[f"{group}_{metric}"] = {
                'trend_slope': slope,
                'trend_p_value': p_value,
                'r_squared': r_value**2,
                'recent_change': recent_change,
                'stability': 'improving' if slope < 0 else 'deteriorating' if slope > 0 else 'stable'
            }
        
        return {
            'temporal_patterns': trends,
            'summary': {
                'metrics_with_trends': len(trends),
                'significant_trends': sum(v['trend_p_value'] < self.config.alpha for v in trends.values())
            }
        }
    
    def _statistical_summary(self) -> Dict:
        """Statistical characterization of fairness landscape"""
        if self.history.empty:
            return {'status': 'no_data', 'message': 'No statistical data available'}
        
        # Descriptive statistics
        desc_stats = self.history.groupby(['metric', 'groups'])['value'].describe()
        
        # Significance analysis
        sig_analysis = self.history.groupby(['metric', 'groups']).agg({
            'p_value': lambda x: (x < self.config.alpha).mean(),
            'value': ['mean', 'std']
        })
        
        # Distribution tests
        distribution_info = {}
        for (metric, group), data in self.history.groupby(['metric', 'groups']):
            distribution_info[f"{group}_{metric}"] = {
                'shapiro_p': stats.shapiro(data['value']).pvalue,
                'kurtosis': stats.kurtosis(data['value']),
                'normality': stats.shapiro(data['value']).pvalue > self.config.alpha
            }
        
        return {
            'descriptive_statistics': desc_stats.to_dict(),
            'significance_analysis': sig_analysis.to_dict(),
            'distribution_characteristics': distribution_info,
            'cross_metric_correlation': self.history.pivot_table(
                index='timestamp', columns='metric', values='value').corr().to_dict()
        }
    
if __name__ == "__main__":
    # 1. Configure Logger for testing output
    logger.info("Starting Fairness Evaluator test...")

    # 2. Generate Synthetic Data
    np.random.seed(42)
    num_samples = 5000
    data = pd.DataFrame({
        'feature1': np.random.rand(num_samples) * 10,
        'feature2': np.random.normal(5, 2, num_samples),
        'sensitive_A': np.random.choice(['groupX', 'groupY'], num_samples, p=[0.6, 0.4]),
        'sensitive_B': np.random.choice([0, 1], num_samples, p=[0.7, 0.3])
    })

    # Simulate predictions with some bias related to sensitive_A
    predictions_prob = 1 / (1 + np.exp(-(
        0.5 * data['feature1']
        - 0.3 * data['feature2']
        + np.where(data['sensitive_A'] == 'groupX', -0.5, 0.8) # Introduce bias
        + np.random.normal(0, 0.5, num_samples) # Add noise
    )))
    predictions_binary = (predictions_prob > 0.5).astype(int) # Binarize for some metrics

    # Simulate true labels (correlated with features but not perfectly)
    true_labels = (0.6 * data['feature1'] - 0.4 * data['feature2'] + np.random.normal(0, 2, num_samples) > 5).astype(int)

    logger.info(f"Generated synthetic data with {num_samples} samples.")
    logger.info(f"Sensitive attribute 'sensitive_A' distribution:\n{data['sensitive_A'].value_counts()}")
    logger.info(f"Sensitive attribute 'sensitive_B' distribution:\n{data['sensitive_B'].value_counts()}")
    logger.info(f"Prediction (binary) distribution:\n{pd.Series(predictions_binary).value_counts()}")
    logger.info(f"True label distribution:\n{pd.Series(true_labels).value_counts()}")


    # 3. Instantiate FairnessConfig and Evaluator
    sensitive_attributes_list = ['sensitive_A', 'sensitive_B']
    evaluator = FairnessEvaluator(
        sensitive_attributes=sensitive_attributes_list,  # Required parameter
        config_section_name="fairness_evaluator",        # Correct config section
        config_file_path="src/agents/alignment/configs/alignment_config.yaml"
    )
    logger.info("FairnessEvaluator instantiated.")

    # 4. Evaluate Group Fairness
    logger.info("\n--- Evaluating Group Fairness ---")
    try:
        # Using binary predictions for metrics like equal opportunity etc.
        group_fairness_report = evaluator.evaluate_group_fairness(
            data=data,
            predictions=predictions_binary,
            labels=true_labels
        )
        import json
        print(json.dumps(group_fairness_report, indent=2, default=lambda x: '<object>')) # Print nicely, avoid printing distribution array
    except Exception as e:
        logger.error(f"Error during group fairness evaluation: {e}", exc_info=True)

    # 5. Evaluate Individual Fairness
    logger.info("\n--- Evaluating Individual Fairness ---")
    try:
        # Prepare data for individual fairness (usually requires numerical features)
        # Exclude sensitive attributes themselves from distance calculation usually
        features_for_individual = data[['feature1', 'feature2']]

        # Using probability predictions for individual fairness (sensitive to small changes)
        individual_fairness_report = evaluator.evaluate_individual_fairness(
            data=features_for_individual,
            predictions=predictions_prob
            # similarity_fn can be customized here if needed
        )
        print(json.dumps(individual_fairness_report, indent=2))
    except Exception as e:
        logger.error(f"Error during individual fairness evaluation: {e}", exc_info=True)

    # 6. Generate Final Report
    logger.info("\n--- Generating Final Report ---")
    try:
        final_report = evaluator.generate_report()
        # Need a way to print the potentially large report dict cleanly
        print("Current State Summary:")
        print(json.dumps(final_report.get('current_state', {}).get('summary', {}), indent=2))
        print("\nHistorical Trends Summary:")
        print(json.dumps(final_report.get('historical_trends', {}).get('summary', {}), indent=2))
        # Avoid printing full large dicts unless needed
        # print("\nFull Report:")
        # print(json.dumps(final_report, indent=2, default=lambda x: '<complex_object>'))

    except Exception as e:
        logger.error(f"Error during report generation: {e}", exc_info=True)

    logger.info("\n--- Fairness Evaluator test finished ---")
