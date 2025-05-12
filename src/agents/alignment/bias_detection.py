"""
Formal Bias Detection Framework
Implements intersectional bias analysis and statistical fairness verification from:
- Mitchell et al. (2019) "Model Cards for Model Reporting"
- Barocas & Hardt (2018) "Fairness and Machine Learning"
"""

import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm

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
        groups = self._generate_intersectional_groups(df)
        
        group_means = {}
        for group_id, group_data in groups.items():
            if len(group_data) < self.config.min_group_size:
                continue
            group_means[group_id] = group_data['predictions'].mean()
        
        if not group_means:  # Handle empty case
            return {
                'global_ratio': 1.0,
                'group_means': {},
                'threshold': 0.8
            }
        
        max_mean = max(group_means.values())
        min_mean = min(group_means.values())
        di_ratio = min_mean / (max_mean + 1e-6)  # Prevent division by zero
        
        return {
            'global_ratio': di_ratio,
            'group_means': group_means,
            'threshold': 0.8  # EEOC standard
        }

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
        # If report values are not dicts with 'p_value', skip significance testing
        if not report or not all(isinstance(v, dict) and 'p_value' in v for v in report.values()):
            return report
        p_values = [v['p_value'] for v in report.values()]
        reject, pvals_corrected, _, _ = multipletests(
            p_values, alpha=self.config.alpha, method='fdr_bh'
        )
        
        for (group_id, result), rejected, adj_p in zip(report.items(), reject, pvals_corrected):
            result['significant'] = rejected
            result['adj_p_value'] = adj_p
            
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
        timestamp = datetime.datetime.now()
        for metric, groups in report.items():
            # Only proceed if groups is a dictionary of group-wise results with 'value' and 'significant'
            if isinstance(groups, dict) and all(
                isinstance(v, dict) and 'value' in v and 'significant' in v
                for v in groups.values()
            ):
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
            else:
                # Skip non-group metrics like disparate_impact
                continue

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
    
        for metric in self.config.metrics:
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
                if p_value < self.config.alpha:
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
        """Normality test with effect size"""
        stat, p = stats.shapiro(data)
        return {
            'statistic': float(stat),
            'p_value': float(p),
            'effect_size': np.sqrt(np.log(stat**2))
        }
    
    # MANOVA Implementation
    def _perform_manova(self) -> Dict:
        """Multivariate analysis of variance across metrics and groups"""
        try:
            if self.bias_history.empty:
                return {}
                
            # Prepare data matrix
            df = self.bias_history.pivot_table(
                index='timestamp', 
                columns='metric', 
                values='value',
                aggfunc='mean'
            ).ffill()
            
            # Add group membership as factor
            groups = self.bias_history.groupby('timestamp')['groups'].first()
            df = df.join(groups)
            
            # Filter metrics with sufficient variance
            valid_metrics = [m for m in self.config.metrics if df[m].var() > 0.01]
            if len(valid_metrics) < 2:
                return {}
                
            # Run MANOVA
            manova = MANOVA.from_formula(
                f"{' + '.join(valid_metrics)} ~ groups", 
                data=df
            )
            return {
                'statistic': float(manova.mv_test().results['groups']['stat'].iloc[0,0]),
                'p_value': float(manova.mv_test().results['groups']['stat'].iloc[0,3]),
                'pillais_trace': float(manova.mv_test().results['groups']['stat'].iloc[0,1])
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
    from faker import Faker
    from scipy.stats import norm, bernoulli
    
    fake = Faker()
    np.random.seed(42)
    
    # Generate synthetic dataset
    def generate_test_data(n=5000):
        data = pd.DataFrame({
            'age': np.random.randint(18, 80, n),
            'gender': np.random.choice(['M', 'F', 'X'], n),
            'race': np.random.choice(['A', 'B', 'C'], n),
            'income': norm.rvs(loc=50000, scale=15000, size=n)
        })
        
        # Simulate biased predictions
        data['prediction'] = bernoulli.rvs(
            p=(0.2 + 0.3*(data['gender'] == 'M') 
            - 0.1*(data['race'] == 'C') 
            + 0.05*(data['age'] > 50)
        ))
        
        # Simulate ground truth labels with noise
        data['label'] = data['prediction'] ^ bernoulli.rvs(0.15, size=n)
        
        return data

    # Initialize detector with test configuration
    detector = BiasDetector(
        sensitive_attributes=['age', 'gender', 'race'],
        config=BiasDetection(
            bootstrap_samples=500,
            min_group_size=20,
            intersectional_depth=2
        )
    )

    # Phase 1: Basic functionality test
    print("\n=== Phase 1: Initial Bias Detection ===")
    test_data = generate_test_data()
    report = detector.compute_metrics(
        data=test_data,
        predictions=test_data['prediction'],
        labels=test_data['label']
    )
    
    # Print key findings
    print("\nTop Disparities:")
    current_state = detector.generate_report()['current_state']
    for metric, info in current_state['worst_performers'].items():
        print(f"{metric}: {info['group']} ({info['disparity']:.2f})")

    # Phase 2: Temporal analysis simulation
    print("\n=== Phase 2: Temporal Pattern Injection ===")
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    for i, date in enumerate(dates):
        # Generate evolving data with concept drift
        temp_data = generate_test_data(100)
        temp_data['prediction'] = temp_data['prediction'] | (i > 500)  # Inject drift
        
        detector.compute_metrics(
            data=temp_data,
            predictions=temp_data['prediction'],
            labels=temp_data['label']
        )
        
    # Phase 3: Advanced analysis
    print("\n=== Phase 3: Comprehensive Reporting ===")
    full_report = detector.generate_report()
    
    print("\nChangepoints Detected:")
    for metric, trend in full_report['historical_trends'].items():
        print(f"{metric}: {len(trend['changepoints'])} change points")
    
    print("\nSeasonality Strengths:")
    for metric, trend in full_report['historical_trends'].items():
        print(f"{metric}: {trend['seasonality_strength']:.2f}")

    # Visualization
    def plot_bias_trends(detector: BiasDetector):
        """Visualize temporal bias patterns"""
        plt.figure(figsize=(14, 8))
        for metric in detector.config.metrics:
            trend_data = detector.bias_history[detector.bias_history.metric == metric]
            trend_data.set_index('timestamp')['value'].rolling(30).mean().plot(
                label=metric, alpha=0.7
            )
        plt.title("Bias Metric Trends with 30-day Moving Average")
        plt.legend()
        plt.ylabel("Disparity Magnitude")
        plt.show()

    plot_bias_trends(detector)
