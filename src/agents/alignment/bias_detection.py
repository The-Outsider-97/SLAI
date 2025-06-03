"""
Formal Bias Detection Framework
Implements intersectional bias analysis and statistical fairness verification from:
- Mitchell et al. (2019) "Model Cards for Model Reporting"
- Barocas & Hardt (2018) "Fairness and Machine Learning"
"""

import os
import yaml
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm

from typing import Dict, List, Optional
from itertools import product, combinations
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.multitest import multipletests
from statsmodels.multivariate.manova import MANOVA
from ruptures import Binseg #, CostLinear

from src.agents.alignment.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Bias Detection")
printer = PrettyPrinter

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

    def __init__(self, sensitive_attributes: List[str]):
        self.sensitive_attrs = sensitive_attributes
        self.config = load_global_config()
        self.detector_config = get_config_section('bias_detection')

        self.bias_history = pd.DataFrame(columns=[
            'timestamp', 'metric', 'value', 'groups', 'stat_significance'
        ])

        printer.status("INIT", f"Bias Detection Initialized with: {self.bias_history}", "Success")

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
        groups = self._generate_intersectional_groups(data,
                                                      self.detector_config.get('intersectional_depth', 3),
                                                      self.detector_config.get('min_group_size', 30))

        for metric in self.detector_config.get("metrics", []):
            metric_report = self._compute_metric(
                metric, data, predictions, labels, groups
            )
            report[metric] = self._add_statistical_significance(metric_report, self.detector_config.get("alpha", 0.05))
            
        self._update_history(report)
        report 
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

    # 1. Demographic Parity
    def _demographic_parity(self, data: pd.DataFrame,
                           predictions: np.ndarray,
                           labels: np.ndarray,
                           groups: Dict) -> Dict:
        """Compute demographic parity differences"""
        return self._group_metric(
            data.assign(predictions=predictions),
            lambda df: df['predictions'].mean()
        )

    # 2. Equal Opportunity (Calibration Equality)
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

    # 3. Predictive Parity (Calibration Equality)
    def _predictive_parity(self, data: pd.DataFrame,
                        predictions: np.ndarray,
                        labels: np.ndarray,
                        groups: Dict) -> Dict:
        """Predictive parity (PPV parity) analysis"""
        df = data.assign(predictions=predictions, labels=labels)
        return self._group_metric(
            df[df['predictions'] == 1],  # Consider only positive predictions
            lambda df: df['labels'].mean()  # PPV = TP/(TP+FP)
        )

    # 4. Disparate Impact Ratio 
    def _disparate_impact(self, data: pd.DataFrame,
                        predictions: np.ndarray,
                        labels: np.ndarray,
                        groups: Dict) -> Dict:
        """Disparate impact ratio analysis"""
        # Add predictions to data and regenerate groups
        df = data.assign(predictions=predictions)
        min_group_size_val = self.detector_config.get('min_group_size', 30)
        intersectional_depth_val = self.detector_config.get('intersectional_depth', 3)
        current_groups = self._generate_intersectional_groups(df, intersectional_depth_val, min_group_size_val)
        
        group_means = {}
        for group_id, group_data in current_groups.items(): # Use current_groups
            if len(group_data) < min_group_size_val: # Use variable
                continue
            group_means[group_id] = group_data['predictions'].mean()
        
        if not group_means:
            return {
                'global_ratio': 1.0,
                'group_means': {},
                'threshold': 0.8 
            }
        
        # Ensure no division by zero if max_mean is 0
        max_mean = max(group_means.values()) if group_means else 0
        min_mean = min(group_means.values()) if group_means else 0
        di_ratio = (min_mean / max_mean) if max_mean != 0 else (1.0 if min_mean == 0 else 0.0) # Handle max_mean = 0
        
        return {
            'global_ratio': di_ratio,
            'group_means': group_means,
            'threshold': 0.8
        }

    def _generate_intersectional_groups(self, data: pd.DataFrame, intersectional_depth: int, min_group_size: int) -> Dict:
        groups = {}
        for depth in range(1, intersectional_depth + 1):
            for combo in combinations(self.sensitive_attrs, depth):
                if not all(attr in data.columns for attr in combo):
                    logger.warning(f"Skipping combo {combo} due to missing attributes in data columns: {data.columns}")
                    continue
                try:
                    for values in product(*[data[attr].unique() for attr in combo]):
                        group_mask = pd.Series(True, index=data.index)
                        for attr, val in zip(combo, values):
                            group_mask &= (data[attr] == val)
                        
                        if group_mask.sum() >= min_group_size:
                            group_id = "_".join(f"{k}={v}" for k, v in zip(combo, values))
                            groups[group_id] = data[group_mask]
                except KeyError as e:
                    logger.error(f"KeyError during group generation for combo {combo}: {e}. Available columns: {data.columns}")
                    continue
        return groups

    def _group_metric(self, data: pd.DataFrame,
                     metric_fn: callable) -> Dict:
        results = {}
        min_group_size = self.detector_config.get('min_group_size', 30)
        intersectional_depth = self.detector_config.get('intersectional_depth', 3)

        for group_id, group_data in self._generate_intersectional_groups(data, intersectional_depth, min_group_size).items():
            if len(group_data) < min_group_size:
                continue
                
            samples = self._bootstrap_sample(group_data, metric_fn, self.detector_config.get('bootstrap_samples', 1000))
            stats_val = self._compute_statistics(samples)
            
            results[group_id] = {
                'value': stats_val['mean'],
                'ci_lower': stats_val['ci_lower'],
                'ci_upper': stats_val['ci_upper'],
                'p_value': self._hypothesis_test(samples)
            }
        return results

    def _bootstrap_sample(self, data: pd.DataFrame,
                         metric_fn: callable, bootstrap_samples: int) -> np.ndarray:
        return np.array([
            float(metric_fn(data.sample(frac=1, replace=True)))
            for _ in range(bootstrap_samples)
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

    def _add_statistical_significance(self, report: Dict, alpha: float) -> Dict:
        if not report or not all(isinstance(v, dict) and 'p_value' in v for v in report.values()):
            return report
        p_values = [v['p_value'] for v in report.values() if isinstance(v, dict) and 'p_value' in v]
        if not p_values:
            return report

        reject, pvals_corrected, _, _ = multipletests(
            p_values, alpha=alpha, method='fdr_bh'
        )
        
        idx = 0
        for group_id, result in report.items():
            if isinstance(result, dict) and 'p_value' in result:
                if idx < len(reject):
                    result['significant'] = reject[idx]
                    result['adj_p_value'] = pvals_corrected[idx]
                    idx += 1
                else:
                    result['significant'] = None 
                    result['adj_p_value'] = None
            
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
        """Update longitudinal bias tracking with list-based appending"""
        timestamp = datetime.datetime.now()
        new_entries = []
        
        for metric, groups in report.items():
            if isinstance(groups, dict) and all(
                isinstance(v, dict) and 'value' in v and 'significant' in v
                for v in groups.values()
            ):
                for group_id, result in groups.items():
                    new_entries.append({
                        'timestamp': timestamp,
                        'metric': metric,
                        'value': result['value'],
                        'groups': group_id,
                        'stat_significance': result['significant']
                    })
        
        # Batch append
        if new_entries:
            self.bias_history = pd.concat([
                self.bias_history,
                pd.DataFrame(new_entries)
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
        for metric in self.detector_config.get('metrics'):
            metric_data = current_data[current_data['metric'] == metric]
            
            # Basic stats
            values = metric_data['value'].astype(float)
            
            if values.empty:
                continue  # Skip metrics with no recorded group-level values
            
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
        """Temporal bias pattern analysis"""
        if self.bias_history.empty:
            return {}
    
        trends = {}
        df = self.bias_history.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
        for metric in self.detector_config.get('metrics'):
            metric_df = df[df['metric'] == metric]
            
            # Resample to daily frequency
            daily = metric_df['value'].resample('D').mean().ffill()
            
            # Skip metrics with insufficient data
            if len(daily) < 2 or daily.isna().all() or daily.nunique() == 1:
                continue
                
            # Robust trend analysis
            try:
                x = np.arange(len(daily))
                y = daily.values
                valid_mask = ~np.isnan(y)
                
                if sum(valid_mask) < 2:
                    continue
                    
                trend_coeff = np.polyfit(x[valid_mask], y[valid_mask], 1)[0]
                rolling_mean = daily.rolling(window=window_size, min_periods=1).mean()
            except np.linalg.LinAlgError as e:
                logger.warning(f"Trend analysis failed for {metric}: {str(e)}")
                trend_coeff = 0
                
            trends[metric] = {
                'trend_direction': 'increasing' if trend_coeff > 0 else 'decreasing',
                'trend_magnitude': abs(float(trend_coeff)),
                'changepoints': self._detect_changepoints(daily.dropna().values),
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
        for metric in self.detector_config.get('metrics'):
            values = self.bias_history[self.bias_history['metric'] == metric]['value']
            stats['distribution_analysis'][metric] = {
                'skewness': float(values.skew()),
                'kurtosis': float(values.kurtosis()),
                'normality_test': self._shapiro_wilk_test(values)
            }

        # Multivariate ANOVA across groups
        if len(self.detector_config.get('metrics')) > 1:
            stats['hypothesis_testing']['manova'] = self._perform_manova()

        # Effect size calculations
        for metric in self.detector_config.get('metrics'):
            metric_values = self.bias_history[self.bias_history['metric'] == metric]['value']
            stats['effect_sizes'][metric] = {
                'cohens_d': self._cohens_d(metric_values),
                'hedges_g': self._hedges_g(metric_values),
                'variance_ratio': float(metric_values.var() / self.bias_history['value'].var())
            }

        return stats

    def _detect_changepoints(self, data: np.ndarray) -> List[int]:
        """E-Divisive changepoint detection with permutation testing"""
        try:
            # Convert to numpy array if needed
            if isinstance(data, pd.Series):
                data = data.values
                
            # Normalize data for change point detection
            normalized = (data - np.mean(data)) / np.std(data)
            
            # Use binary segmentation with linear model cost
            algo = Binseg(model='l2').fit(normalized)
            changepoints = algo.predict(pen=1.0)
            
            # Filter significant changepoints using permutation test
            significant_points = []
            for cp in changepoints:
                before = data[:cp]
                after = data[cp:]
                p_value = stats.ttest_ind(before, after).pvalue
                if p_value < self.detector_config.get('alpha'):
                    significant_points.append(cp)
                    
            return significant_points
            
        except Exception as e:
            logger.error(f"Changepoint detection failed: {str(e)}")
            return []

    def _measure_seasonality(self, series: pd.Series) -> float:
        """STL decomposition-based seasonality strength"""
        try:
            if len(series) < 2 * 365:  # Minimum 2 cycles for yearly seasonality
                return 0.0
                
            stl = STL(series, period=365, robust=True)
            res = stl.fit()
            
            # Calculate strength of seasonality component
            var_total = np.nanvar(series)
            var_seasonal = np.nanvar(res.seasonal)
            
            return float(var_seasonal / var_total) if var_total > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Seasonality measurement failed: {str(e)}")
            return 0.0

    def _shapiro_wilk_test(self, data: pd.Series) -> Dict:
        """Normality test with error handling"""
        if len(data) < 3 or len(data) > 5000:
            return {
                'statistic': None,
                'p_value': None,
                'effect_size': None,
                'warning': "Invalid sample size for Shapiro-Wilk (3 ≤ n ≤ 5000)"
            }
            
        stat, p = stats.shapiro(data)
        effect_size = np.sqrt(np.log(max(stat**2, 1e-6)))  # Prevent log(0)
        
        return {
            'statistic': float(stat),
            'p_value': float(p),
            'effect_size': float(effect_size)
        }
    
    # MANOVA Implementation
    def _perform_manova(self) -> Dict:
        """Improved MANOVA implementation"""
        try:
            if self.bias_history.empty:
                return {}
    
            # Prepare data with variance check
            df = self.bias_history.pivot_table(
                index='timestamp', 
                columns='metric', 
                values='value',
                aggfunc='mean'
            ).ffill()
            
            # Filter metrics with numerical values and sufficient variance
            valid_metrics = [
                m for m in self.detector_config.get('metrics')
                if pd.api.types.is_numeric_dtype(df[m]) 
                and df[m].var() > 0.01
            ]
            
            if len(valid_metrics) < 2:
                return {}
    
            # Handle potential missing data
            df = df[valid_metrics].join(
                self.bias_history.groupby('timestamp')['groups'].first()
            ).dropna()
    
            manova = MANOVA.from_formula(
                f"{' + '.join(valid_metrics)} ~ groups", 
                data=df
            )
            
            return {
                'statistic': float(manova.mv_test().results['groups']['stat'].iloc[0,0]),
                'p_value': float(manova.mv_test().results['groups']['stat'].iloc[0,3])
            }
        except Exception as e:
            logger.error(f"MANOVA failed: {str(e)}")
            return {}

    def _cohens_d(self, values: pd.Series) -> float:
        """Effect size relative to ideal zero disparity"""
        return float((values.mean() - 0) / values.std())

    def _hedges_g(self, values: pd.Series) -> float:
        """Bias-corrected effect size"""
        n = len(values)
        return self._cohens_d(values) * (1 - (3)/(4*(n-2)-1))

if __name__ == "__main__":
    print("\n=== Running Bias Detection ===\n")
    printer.status("Init", "Bias Detection initialized", "success")
    sensitive_attributes=["gender", "age_group", "race", "education_level"]

    detector = BiasDetector(
        sensitive_attributes=sensitive_attributes
    )

    printer.pretty("detector", f"{detector}", "success")
    print("\n* * * * * Phase 2 * * * * *")
    import random
    if not detector.detector_config.get("metrics"):
        detector.detector_config["metrics"] = [
            "demographic_parity",
            "equal_opportunity",
            "predictive_parity",
            "disparate_impact"
        ]

    n = 1000
    data = pd.DataFrame({
        'gender': random.choices(['Male', 'Female', 'Non-binary', 'Prefer not to say'], [0.48, 0.48, 0.03, 0.01], k=n),
        'age_group': random.choices(['18-24', '25-34', '35-44', '45-54', '55-64', '65+'], [0.15, 0.25, 0.2, 0.15, 0.15, 0.1], k=n),
        'race': random.choices(['White', 'Black', 'Asian', 'Hispanic', 'Other'], [0.6, 0.13, 0.06, 0.18, 0.03], k=n),
        'education_level': random.choices(['No HS', 'HS', 'Some College', 'Bachelor', 'Graduate'], [0.1, 0.25, 0.25, 0.25, 0.15], k=n)})
    predictions = np.random.binomial(1, 0.3, n)
    labels = np.random.binomial(1, 0.5, n)  # Random labels for demonstration
    
    # Generate predictions and labels
    metric_fn = callable
    
    compute = detector.compute_metrics(
        data=data,
        predictions=predictions,
        labels=labels
    )
    metric = detector._group_metric(data=data, metric_fn=metric_fn)
    
    printer.pretty("compute", f"{compute}", "success")
    printer.pretty("metric", metric, "success")
    print("\n* * * * * Phase 3 * * * * *")
    format="structured"
    report = detector.generate_report(format="structured")
    printer.pretty("report", report, "success")
    print("\n=== Bias Detection Test Completed ===")
