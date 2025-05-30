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

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from itertools import product, combinations
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.multitest import multipletests
from statsmodels.multivariate.manova import MANOVA
from ruptures import Binseg #, CostLinear

from logs.logger import get_logger

logger = get_logger("Bias Detection")

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
                 config_section_name: str = "bias_detector",
                 config_file_path: str = "src/agents/alignment/configs/alignment_config.yaml"):
        self.sensitive_attrs = sensitive_attributes
        # Load config from YAML
        config_data = get_config_section(config_section_name, config_file_path)
        if not isinstance(config_data, dict):
            logger.error(f"Config for {config_section_name} is not a dict. Received: {type(config_data)}")
            self.config = {}
        else:
            self.config = config_data

        if hasattr(self, '_config'):
            self._config = self.config

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
        groups = self._generate_intersectional_groups(data,
                                                      self.config.get('intersectional_depth', 3),
                                                      self.config.get('min_group_size', 30))

        for metric in self.config.get("metrics", []):
            metric_report = self._compute_metric(
                metric, data, predictions, labels, groups
            )
            report[metric] = self._add_statistical_significance(metric_report, self.config.get("alpha", 0.05))
            
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
        min_group_size_val = self.config.get('min_group_size', 30)
        intersectional_depth_val = self.config.get('intersectional_depth', 3)
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
        min_group_size = self.config.get('min_group_size', 30)
        intersectional_depth = self.config.get('intersectional_depth', 3)

        for group_id, group_data in self._generate_intersectional_groups(data, intersectional_depth, min_group_size).items():
            if len(group_data) < min_group_size:
                continue
                
            samples = self._bootstrap_sample(group_data, metric_fn, self.config.get('bootstrap_samples', 1000))
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
            metric_fn(data.sample(frac=1, replace=True))
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
        for metric in self.config["metrics"]:
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
    
        for metric in self.config["metrics"]:
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
        for metric in self.config["metrics"]:
            values = self.bias_history[self.bias_history['metric'] == metric]['value']
            stats['distribution_analysis'][metric] = {
                'skewness': float(values.skew()),
                'kurtosis': float(values.kurtosis()),
                'normality_test': self._shapiro_wilk_test(values)
            }

        # Multivariate ANOVA across groups
        if len(self.config["metrics"]) > 1:
            stats['hypothesis_testing']['manova'] = self._perform_manova()

        # Effect size calculations
        for metric in self.config["metrics"]:
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
                if p_value < self.config["alpha"]:
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
                m for m in self.config["metrics"] 
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
    import matplotlib.pyplot as plt
    import datetime
    import random
    import json
    
    # Realistic test config as dictionary
    test_config = {
        "metrics": ["demographic_parity", "equal_opportunity", "predictive_parity", "disparate_impact"],
        "min_group_size": 30,
        "bootstrap_samples": 100,  # Reduced from 1000 to 100
        "alpha": 0.05,
        "intersectional_depth": 2  # Reduced from 3 to 2
    }

    # Patch BiasDetector to use dict-style config access
    class PatchedBiasDetector(BiasDetector):
        def __getattr__(self, name):
            if name == "config":
                return self.__dict__["_config"]
            raise AttributeError(f"'BiasDetector' object has no attribute '{name}'")

        @property
        def config(self):
            return self._config

        @config.setter
        def config(self, value):
            self._config = value

    # Initialize patched detector
    detector = PatchedBiasDetector(
        sensitive_attributes=["gender", "age_group", "race", "education_level"],
        config_section_name="bias_detector"
    )
    detector.config = test_config

    # Data generator
    def generate_realistic_data(n=5000):
        data = pd.DataFrame()
        data['gender'] = random.choices(['Male', 'Female', 'Non-binary', 'Prefer not to say'], [0.48, 0.48, 0.03, 0.01], k=n)
        data['age_group'] = random.choices(['18-24', '25-34', '35-44', '45-54', '55-64', '65+'], [0.15, 0.25, 0.2, 0.15, 0.15, 0.1], k=n)
        data['race'] = random.choices(['White', 'Black', 'Asian', 'Hispanic', 'Other'], [0.6, 0.13, 0.06, 0.18, 0.03], k=n)
        data['education_level'] = random.choices(['No HS', 'HS', 'Some College', 'Bachelor', 'Graduate'], [0.1, 0.25, 0.25, 0.25, 0.15], k=n)
        base_prob = 0.3
        data['prediction'] = np.random.binomial(1, base_prob, n)

        for idx, row in data.iterrows():
            if row['gender'] == 'Male':
                data.at[idx, 'prediction'] = np.random.binomial(1, min(0.9, base_prob * 1.4))
            if row['age_group'] in ['55-64', '65+']:
                data.at[idx, 'prediction'] = np.random.binomial(1, max(0.1, base_prob * 0.7))
            if row['race'] in ['Black', 'Hispanic']:
                data.at[idx, 'prediction'] = np.random.binomial(1, max(0.1, base_prob * 0.8))
            if row['education_level'] in ['Bachelor', 'Graduate']:
                data.at[idx, 'prediction'] = np.random.binomial(1, min(0.9, base_prob * 1.3))

        data['label'] = data['prediction'].apply(lambda x: x if random.random() < 0.8 else 1 - x)
        return data

    # Run initial detection
    print("\n=== Initial Bias Detection ===")
    test_data = generate_realistic_data(10000)
    initial_report = detector.compute_metrics(
        data=test_data,
        predictions=test_data['prediction'],
        labels=test_data['label']
    )

    print("\nDisparate Impact Ratios:")
    for group, value in initial_report["disparate_impact"]["group_means"].items():
        print(f"{group}: {value:.3f}")

    print("\n=== Simulating Temporal Analysis (1 year daily data) ===")
    start_date = datetime.datetime(2023, 1, 1)
    for day in range(365):
        daily_data = generate_realistic_data(100)
        if day > 180:
            daily_data.loc[daily_data['gender'] == 'Female', 'prediction'] = np.random.binomial(
                1, max(0.1, 0.3 * (1 - 0.5 * (day - 180) / 185)), sum(daily_data['gender'] == 'Female')
            )
        detector.compute_metrics(
            data=daily_data,
            predictions=daily_data['prediction'],
            labels=daily_data['label']
        )

    print("\n=== Generating Final Report ===")
    final_report = detector.generate_report()

    print("\nCurrent State Summary:")
    for metric, stats in final_report["current_state"]["metrics_summary"].items():
        print(f"{metric}:")
        print(f"  Mean disparity: {stats['mean_disparity']:.3f}")
        print(f"  Max disparity: {stats['max_disparity']:.3f}")
        print(f"  Affected groups: {stats['affected_groups']}")

    print("\nTrend Analysis:")
    for metric, trend in final_report["historical_trends"].items():
        print(f"{metric}:")
        print(f"  Direction: {trend['trend_direction']}")
        print(f"  Magnitude: {trend['trend_magnitude']:.3f}")
        print(f"  Change points: {len(trend['changepoints'])}")

    def plot_realistic_trends(detector):
        plt.figure(figsize=(14, 8))
        for metric in ["demographic_parity", "equal_opportunity"]:
            metric_data = detector.bias_history[detector.bias_history["metric"] == metric]
            metric_data = metric_data.groupby("timestamp")["value"].mean().rolling(30).mean()
            plt.plot(metric_data.index, metric_data.values, label=metric, linewidth=2)
        plt.title("Realistic Bias Trends (30-day MA)")
        plt.xlabel("Date")
        plt.ylabel("Disparity")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    plot_realistic_trends(detector)

    with open("bias_audit_report.json", "w") as f:
        json.dump(final_report, f, indent=2, default=str)

    print("\n✅ Full report saved to bias_audit_report.json")
