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

from typing import Dict, List, Optional, Callable
from scipy import stats
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.stats import linregress

from src.agents.alignment.utils.config_loader import load_global_config, get_config_section
from src.agents.alignment.alignment_memory import AlignmentMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Fairness Evaluator")
printer = PrettyPrinter

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

    def __init__(self):
        self.config = load_global_config()
        self.sensitive_attributes = self.config.get('sensitive_attributes')

        self.fe_config = get_config_section('fairness_evaluator')
        self.group_metrics = self.fe_config.get('group_metrics', [])
        self.individual_metrics = self.fe_config.get('individual_metrics', [])
        self.alpha = self.fe_config.get('alpha')
        self.n_bootstrap = self.fe_config.get('n_bootstrap')
        self.batch_size = self.fe_config.get('batch_size')
        self.similarity_metric = self.fe_config.get('similarity_metric')
        self.sensitive_attrs = self.fe_config.get(
            'sensitive_attributes_override', self.sensitive_attributes)

        self.history = pd.DataFrame(columns=[
            "metric", "value", "groups", "p_value", "timestamp", "sensitive_attr", "metric_name", "metric_value", "violation_flag"
        ])

        self.sensitive_attrs = self.sensitive_attributes
        self.alignment_memory = AlignmentMemory()

        logger.info(f"Fairness Evaluator succesfully initialized")

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
            for metric in self.group_metrics:
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
        if not isinstance(self.n_bootstrap, int) or self.n_bootstrap <= 0:
            logger.error("Invalid bootstrap count. Using default 1000")
            n_bootstrap = 1000
        else:
            n_bootstrap = self.n_bootstrap

        bootstraps = []
        
        for _ in range(n_bootstrap):
            sample = df.sample(frac=1, replace=True)
            grouped = sample.groupby(attr)

            group_metrics_vals = {}
            for group_name, group_df in grouped:
                if metric_fn.__code__.co_varnames[0] == 'g':  # Takes DataFrame
                    group_metrics_vals[group_name] = metric_fn(group_df[['prediction', 'label']])
                else:  # Takes Series
                    group_metrics_vals[group_name] = metric_fn(group_df['prediction'])
                    
            group_metrics = pd.Series(group_metrics_vals)

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
            'significant': p_value < self.alpha
        })
        return result

    def _calculate_consistency(self, data: pd.DataFrame,
                              predictions: np.ndarray,
                              similarity_fn: Callable) -> float:
        """Convert predictions to numpy array first"""
        predictions = np.asarray(predictions)
        distances = pairwise_distances(data, metric=similarity_fn)
        prediction_diffs = np.abs(predictions[:, None] - predictions)
        return np.exp(-np.mean(distances * prediction_diffs))

    def _estimate_lipschitz(self, data: pd.DataFrame,
                           predictions: np.ndarray,
                           similarity_fn: Callable) -> float:
        """Lipschitz constant estimation for individual fairness"""
        predictions = np.asarray(predictions)
        distances = pairwise_distances(data, metric=similarity_fn).ravel()
        pred_diffs = np.abs(predictions[:, None] - predictions).ravel()
        nonzero = distances > 1e-6
        return np.max(pred_diffs[nonzero] / distances[nonzero])

    def _identify_violations(self, data: pd.DataFrame,
                            predictions: np.ndarray,
                            similarity_fn: Callable) -> Dict:
        predictions = np.asarray(predictions)
        if data.empty or len(data) < 2:
            logger.warning("Not enough samples to identify individual fairness violations.")
            return {
                'max_violation': 0.0,
                'mean_violation': 0.0,
                'violation_rate': 0.0,
                'status': 'skipped_insufficient_data'
            }

        n_neighbors_config = getattr(self.config, 'n_neighbors', 50)
        if isinstance(self.config, dict):
            n_neighbors_config = self.config.get('n_neighbors', 50)

        num_samples = len(data)
        n_neighbors_to_use = min(n_neighbors_config, num_samples)
        if n_neighbors_to_use == 0 and num_samples > 0:
            n_neighbors_to_use = 1

        if n_neighbors_to_use == 0 :
             logger.warning("Not enough samples for NearestNeighbors after adjusting n_neighbors.")
             return { 'max_violation': 0.0, 'mean_violation': 0.0, 'violation_rate': 0.0, 'status': 'skipped_no_neighbors_possible'}


        nn = NearestNeighbors(n_neighbors=n_neighbors_to_use, metric=similarity_fn)
        nn.fit(data)
        
        if num_samples < n_neighbors_to_use :
            logger.warning(f"Number of samples ({num_samples}) is less than n_neighbors ({n_neighbors_to_use}). Skipping Kneighbors.")
            return { 'max_violation': np.nan, 'mean_violation': np.nan, 'violation_rate': np.nan, 'status': 'skipped_nn_error'}

        distances, indices = nn.kneighbors(data)

        violations = []
        for i in range(len(data)):
            if distances[i].size == 0:
                continue
            neighbor_preds = predictions[indices[i]]
            safe_distances = distances[i] + 1e-9
            deviation = np.abs(predictions[i] - neighbor_preds) / safe_distances
            violations.extend(deviation)

        if not violations:
            return {
                'max_violation': 0.0,
                'mean_violation': 0.0,
                'violation_rate': 0.0,
                'status': 'no_violations_calculated'
            }

        return {
            'max_violation': np.max(violations) if violations else 0.0,
            'mean_violation': np.mean(violations) if violations else 0.0,
            'violation_rate': np.mean(np.array(violations) > 1.0) if violations else 0.0
            # Example threshold for violation is deviation > 1.0 (Lipschitz constant > 1)
        }

    def _default_similarity(self, a: pd.Series, b: pd.Series) -> float:
        """Normalized Manhattan distance"""
        # Convert to numpy arrays for safe subtraction
        a_values = a.values if isinstance(a, pd.Series) else a
        b_values = b.values if isinstance(b, pd.Series) else b
        return np.sum(np.abs(a_values - b_values)) / len(a_values)

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
                'significant': row['p_value'] < self.alpha,
                'last_updated': row['timestamp'].isoformat()
            }
    
        # Count significant disparities
        sig_counts = latest.groupby('groups').apply(
            lambda g: (g['p_value'] < self.alpha).sum())
        
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
                'significant_trends': sum(v['trend_p_value'] < self.alpha for v in trends.values())
            }
        }
    
    def _statistical_summary(self) -> Dict:
        """Statistical characterization of fairness landscape"""
        if self.history.empty:
            return {'status': 'no_data', 'message': 'No statistical data available'}
        
        # Descriptive statistics
        desc_stats = self.history.groupby(['metric', 'groups'])['value'].describe()
        # Convert multi-index to string keys
        descriptive_statistics = {}
        for idx, row in desc_stats.iterrows():
            key = f"{idx[0]}_{idx[1]}"
            descriptive_statistics[key] = row.to_dict()
        
        # Significance analysis
        sig_analysis = self.history.groupby(['metric', 'groups']).agg({
            'p_value': lambda x: (x < self.alpha).mean(),
            'value': ['mean', 'std']
        })
        # Convert multi-index columns to string keys
        significance_analysis = {}
        for idx, row in sig_analysis.iterrows():
            key = f"{idx[0]}_{idx[1]}"
            row_dict = {}
            for col in row.index:
                col_name = '_'.join(col) if isinstance(col, tuple) else col
                row_dict[col_name] = row[col]
            significance_analysis[key] = row_dict
        
        # Distribution tests
        distribution_info = {}
        for (metric, group), data in self.history.groupby(['metric', 'groups']):
            key = f"{group}_{metric}"
            distribution_info[key] = {
                'shapiro_p': stats.shapiro(data['value']).pvalue,
                'kurtosis': stats.kurtosis(data['value']),
                'normality': stats.shapiro(data['value']).pvalue > self.alpha
            }
        
        # Cross metric correlation
        correlation_df = self.history.pivot_table(
            index='timestamp', columns='metric', values='value').corr()
        cross_metric_correlation = {
            col: correlation_df[col].to_dict() 
            for col in correlation_df.columns
        }
        
        return {
            'descriptive_statistics': descriptive_statistics,
            'significance_analysis': significance_analysis,
            'distribution_characteristics': distribution_info,
            'cross_metric_correlation': cross_metric_correlation
        }

    def fairness_records(self) -> pd.DataFrame:
        """Return comprehensive fairness records with contextual metadata"""
        # Add contextual metadata to history records
        enriched_records = self.history.copy()
        
        # Add summary statistics
        enriched_records['7d_rolling_avg'] = (
            enriched_records.groupby('metric')['value']
            .transform(lambda x: x.rolling(7).mean())
        )
        
        # Add violation flags based on metric-specific thresholds
        thresholds = {
            'statistical_parity': 0.1,
            'equal_opportunity': 0.1,
            'predictive_parity': 0.15,
            'disparate_impact': 0.8
        }
        
        enriched_records['threshold'] = (
            enriched_records['metric']
            .map(lambda x: thresholds.get(x.split('_')[-1], 0.1))
        )
        
        enriched_records['violation'] = (
            enriched_records['value'] > enriched_records['threshold']
        )
        
        # Add temporal features
        enriched_records['day_of_week'] = enriched_records['timestamp'].dt.dayofweek
        enriched_records['hour_of_day'] = enriched_records['timestamp'].dt.hour
        
        # Add drift indicators
        enriched_records['drift_score'] = (
            enriched_records.groupby('metric')['value']
            .transform(lambda x: x.diff().abs().rolling(30).mean())
        )
        
        return enriched_records

    def _safe_get_int(self, key, default):
        """Safely get integer configuration with fallback"""
        value = self.fe_config.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.warning(f"Invalid config for {key}: {value}. Using default {default}")
            return default
    
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
    evaluator = FairnessEvaluator()
    logger.info("FairnessEvaluator instantiated.")
    evaluator.sensitive_attrs = ['sensitive_A', 'sensitive_B']

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
        print("\nFull Report:")
        print(json.dumps(final_report, indent=2, default=lambda x: '<complex_object>'))

    except Exception as e:
        logger.error(f"Error during report generation: {e}", exc_info=True)

    logger.info("\n--- Fairness Evaluator test finished ---")
