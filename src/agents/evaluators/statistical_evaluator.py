"""
    Implements non-dominated sorting from:
    Deb et al. (2002) "A Fast Elitist Multiobjective Genetic Algorith
"""

import numpy as np
import yaml, json
import math

from typing import List, Dict, Any
from scipy.stats import shapiro, t as student_t, f as f_dist
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QSize
from datetime import datetime

from src.agents.evaluators.utils.config_loader import load_global_config, get_config_section
from src.agents.evaluators.utils.evaluators_calculations import EvaluatorsCalculations
from src.agents.evaluators.utils.evaluation_errors import (EvaluationError, ReportGenerationError,
                                                           MetricCalculationError, MemoryAccessError,
                                                           VisualizationError, ValidationFailureError)
from src.agents.evaluators.utils.report import get_visualizer
from src.agents.evaluators.evaluators_memory import EvaluatorsMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Statistical Evaluator")
printer = PrettyPrinter

class StatisticalEvaluator:
    def __init__(self):
        self.config = load_global_config()
        self.statistic_config = get_config_section('statistical_evaluator')
        self.significance_threshold = self.statistic_config.get('alpha')
        self.confidence_level = self.statistic_config.get('confidence_level')
        self.min_sample_size = self.statistic_config.get('min_sample_size')
        self.required_metrics = [
            'descriptive_stats', 'hypothesis_tests', 
            'confidence_intervals', 'effect_sizes'
        ]
        self.memory = EvaluatorsMemory()
        self.calculations = EvaluatorsCalculations()
        self.objectives = []
        self.weights = {}
        self.history = []

        logger.info(f"Statistical Evaluator succesfully initialized")

    def evaluate(self, datasets: Dict[str, List[float]], report: bool = False) -> Dict[str, Any]:
        """
        Comprehensive statistical analysis of multiple datasets with optional reporting
        """
        try:
            # Validate input datasets
            self._validate_datasets(datasets)
            
            results = {}
            
            # Calculate required metrics with error handling
            for metric in self.required_metrics:
                try:
                    if metric == 'descriptive_stats':
                        results[metric] = self.calculations._calculate_descriptive_stats(datasets)
                    elif metric == 'hypothesis_tests':
                        results[metric] = self._run_hypothesis_tests(datasets)
                    elif metric == 'confidence_intervals':
                        results[metric] = {
                            name: self.compute_confidence_interval(data) 
                            for name, data in datasets.items()
                        }
                    elif metric == 'effect_sizes':
                        results[metric] = self.calculations._calculate_all_effect_sizes(datasets)
                except Exception as e:
                    raise MetricCalculationError(
                        metric_name=metric,
                        inputs={k: v[:5] for k, v in datasets.items()},
                        reason=str(e)
                    )

            # Advanced metrics
            if self.config.get('enable_advanced_metrics', True):
                try:
                    results['power_analysis'] = self.calculations._calculate_power_analysis(datasets)
                    results['normality_tests'] = self._check_normality(datasets)
                except Exception as e:
                    logger.error(f"Advanced metrics failed: {str(e)}")
                    results['advanced_metrics_error'] = str(e)

            # Report generation
            if report:
                try:
                    results['report'] = self.generate_report(results)
                except Exception as e:
                    logger.error(f"Report generation failed: {str(e)}")
                    results['report_error'] = str(e)

            # Store results
            if self.config.get('store_results', False):
                try:
                    self.memory.add(
                        entry=results,
                        tags=["statistical_analysis", f"eval_{datetime.now().date()}"],
                        priority="medium"
                    )
                    if self.memory.access_counter % self.memory.config.get("checkpoint_freq", 500) == 0:
                        self.memory.create_checkpoint()
                except Exception as e:
                    raise MemoryAccessError(
                        operation="add",
                        key="statistical_results",
                        error_details=str(e)
                    )

            return results
        except EvaluationError as e:
            logger.error(f"Statistical evaluation failed: {e.to_audit_dict()}")
            return {
                'error': str(e),
                'error_type': type(e).__name__,
                'error_id': getattr(e, 'error_id', 'N/A'),
                'forensic_hash': getattr(e, 'forensic_hash', 'N/A')
            }
        except Exception as e:
            logger.error(f"Unexpected error during statistical evaluation: {str(e)}")
            return {
                'error': str(e),
                'error_type': 'UnexpectedError'
            }

    def _validate_datasets(self, datasets: Dict[str, List[float]]):
        """Validate input datasets before processing"""
        if not datasets:
            raise ValidationFailureError(
                rule_name="dataset_validation",
                data=datasets,
                expected="Non-empty dictionary of datasets"
            )
            
        for name, data in datasets.items():
            if len(data) < self.min_sample_size:
                raise ValidationFailureError(
                    rule_name="sample_size_validation",
                    data={"dataset": name, "size": len(data)},
                    expected=f"Minimum sample size {self.min_sample_size}"
                )
            if not all(isinstance(x, (int, float)) for x in data):
                invalid_samples = [x for x in data if not isinstance(x, (int, float))]
                raise ValidationFailureError(
                    rule_name="data_type_validation",
                    data={"dataset": name, "invalid_samples": invalid_samples[:5]},
                    expected="All numeric values"
                )

    def _run_hypothesis_tests(self, datasets: Dict[str, List[float]]) -> Dict[str, Any]:
        """Automated hypothesis testing between dataset pairs"""
        tests = {}
        names = list(datasets.keys())
        
        # Pairwise comparisons
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                key = f"{names[i]}_vs_{names[j]}"
                try:
                    tests[key] = self.t_test(datasets[names[i]], datasets[names[j]])
                except Exception as e:
                    logger.error(f"T-test failed for {key}: {str(e)}")
                    tests[key] = {
                        'error': str(e),
                        'p_value': None,
                        'significant': None
                    }
                
        # ANOVA if more than two groups
        if len(names) > 2:
            try:
                tests['anova'] = self.anova(list(datasets.values()))
            except Exception as e:
                logger.error(f"ANOVA failed: {str(e)}")
                tests['anova'] = {
                    'error': str(e),
                    'f_stat': None,
                    'p_value': None,
                    'significant': None
                }
            
        return tests

    def _check_normality(self, datasets: Dict[str, List[float]]) -> Dict[str, bool]:
        """Shapiro-Wilk normality test for all datasets"""
        normality = {}
        for name, data in datasets.items():
            if len(data) < 3:
                normality[name] = False
                continue
            try:
                _, p_value = shapiro(data)
                normality[name] = p_value > self.significance_threshold
            except Exception as e:
                logger.error(f"Normality test failed for {name}: {str(e)}")
                normality[name] = None  # Indicate error
        return normality

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive statistical report"""
        try:
            report = [
                "\n## Statistical Analysis Report",
                f"**Generated**: {datetime.now().isoformat()}\n",
                "### Descriptive Statistics\n"
            ]
            visualizer = get_visualizer()
            
            # Descriptive stats
            if 'descriptive_stats' in results:
                try:
                    visualizer.add_metrics("descriptive_stats", results['descriptive_stats'])
                except Exception as e:
                    logger.error(f"Failed to add descriptive stats to visualizer: {str(e)}")
                
                for name, stats in results['descriptive_stats'].items():
                    try:
                        report.append(
                            f"- **{name}**: μ={stats['mean']:.2f}, σ={stats['std_dev']:.2f}, "
                            f"n={stats['sample_size']} (Min={stats['min']:.2f}, Max={stats['max']:.2f})"
                        )
                    except Exception as e:
                        logger.error(f"Failed to format stats for {name}: {str(e)}")
                        report.append(f"- **{name}**: Stats unavailable")

            # Hypothesis tests
            report.append("\n### Hypothesis Testing Results\n")
            if 'hypothesis_tests' in results:
                for test, result in results['hypothesis_tests'].items():
                    if 'error' in result:
                        report.append(f"- **{test}**: ❌ Error: {result['error']}")
                    else:
                        try:
                            report.append(
                                f"- **{test}**: p={result['p_value']:.3f} "
                                f"({'✅ Significant' if result['significant'] else '❌ Not significant'})"
                            )
                        except Exception as e:
                            report.append(f"- **{test}**: Result formatting failed")
            else:
                report.append("- Hypothesis tests unavailable")
                
            # Visualization
            try:
                chart = visualizer.render_temporal_chart(QSize(600, 400), 'statistical_results')
                report.append(f"![Statistical Chart](data:image/png;base64,{visualizer._chart_to_base64(chart)})")
            except Exception as e:
                raise VisualizationError(
                    chart_type="temporal",
                    data=results.get('hypothesis_tests', {}),
                    error_details=f"Chart rendering failed: {str(e)}"
                )

            # Effect sizes
            report.append("\n### Effect Sizes (Cohen's d)\n")
            if 'effect_sizes' in results:
                for comparison, d in results['effect_sizes'].items():
                    try:
                        report.append(f"- **{comparison}**: {d:.2f}")
                    except Exception as e:
                        report.append(f"- **{comparison}**: Formatting error")
            else:
                report.append("- Effect sizes unavailable")

            # Advanced metrics
            if 'power_analysis' in results:
                report.append("\n### Statistical Power Analysis\n")
                for name, power in results['power_analysis'].items():
                    try:
                        report.append(f"- **{name}**: {power:.2%} power")
                    except Exception as e:
                        report.append(f"- **{name}**: Power calculation error")
            
            if 'normality_tests' in results:
                report.append("\n### Normality Tests\n")
                for name, normal in results['normality_tests'].items():
                    status = "✅ Normal" if normal else "❌ Not normal" if normal is False else "⚠️ Test failed"
                    report.append(f"- **{name}**: {status}")

            report.append(f"\n---\n*Significance threshold: α={self.significance_threshold}*")
            report.append(f"*Report generated by {self.__class__.__name__}*")
            
            return "\n".join(report)
        except Exception as e:
            raise ReportGenerationError(
                report_type="Statistical Analysis",
                template="statistical_report",
                error_details=f"Error generating report: {str(e)}"
            )

    def compute_confidence_interval(self, samples, confidence=None):
        """Compute confidence interval with error handling"""
        try:
            n = len(samples)
            if n == 0:
                return (0.0, 0.0)
            mean = np.mean(samples)
            std_err = np.std(samples, ddof=1) / math.sqrt(n)
            z = 1.96 if confidence == 0.95 else 2.58
            margin = z * std_err
            return mean - margin, mean + margin
        except Exception as e:
            raise MetricCalculationError(
                metric_name="confidence_interval",
                inputs=samples[:5],
                reason=str(e)
            )

    def t_test(self, sample_a, sample_b):
        """T-test with error handling"""
        try:
            if len(sample_a) != len(sample_b) or len(sample_a) == 0:
                return {"p_value": 1.0, "significant": False}
                
            diffs = np.array(sample_a) - np.array(sample_b)
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs, ddof=1)
            t_stat = mean_diff / (std_diff / np.sqrt(len(diffs)))
            df = len(diffs) - 1
            p_value = 2 * (1 - student_t.cdf(abs(t_stat), df))
            return {"p_value": p_value, "significant": p_value < 0.05}
        except Exception as e:
            raise MetricCalculationError(
                metric_name="t_test",
                inputs={"sample_a": sample_a[:5], "sample_b": sample_b[:5]},
                reason=str(e)
            )

    def anova(self, groups):
        """ANOVA with error handling"""
        try:
            k = len(groups)
            n_total = sum(len(g) for g in groups)
            grand_mean = np.mean([x for g in groups for x in g])
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
            ss_within = sum(sum((x - np.mean(g))**2 for x in g) for g in groups)
            df_between, df_within = k - 1, n_total - k
            ms_between = ss_between / df_between if df_between else 0
            ms_within = ss_within / df_within if df_within else 1e-6
            f_stat = ms_between / ms_within
            p_value = 1 - f_dist.cdf(f_stat, df_between, df_within)
            return {"f_stat": f_stat, "p_value": p_value, "significant": p_value < 0.05}
        except Exception as e:
            raise MetricCalculationError(
                metric_name="anova",
                inputs=[g[:5] for g in groups],
                reason=str(e)
            )

    def pareto_frontier(self, solutions: List[Dict]) -> List[Dict]:
        """NSGA-II inspired non-dominated sorting"""
        frontiers = []
        remaining = solutions.copy()
        
        while remaining:
            current_front = []
            dominated = []
            
            for candidate in remaining:
                if not any(self._dominates(other, candidate) for other in remaining):
                    current_front.append(candidate)
                else:
                    dominated.append(candidate)
            
            frontiers.append(current_front)
            remaining = dominated
            
        return frontiers

    def _dominates(self, a: Dict, b: Dict) -> bool:
        """Pareto domination criteria with weighted objectives"""
        better = 0
        for obj in self.objectives:
            a_val = a[obj] * self.weights.get(obj, 1.0)
            b_val = b[obj] * self.weights.get(obj, 1.0)
            if a_val > b_val:
                better += 1
            elif a_val < b_val:
                better -= 1
        return better > 0

    def statistical_analysis(self, baseline: List[float], treatment: List[float]) -> Dict:
        """Implements Demšar (2006) statistical comparison protocol"""
        n = len(baseline)
        z_scores = [(b - t) / math.sqrt(n) for b, t in zip(baseline, treatment)]
        return {
            'mean_diff': sum(b - t for b, t in zip(baseline, treatment)) / n,
            'effect_size': self._hedges_g(baseline, treatment),
            'significance': any(abs(z) > 2.58 for z in z_scores)  # p<0.01
        }

    def _hedges_g(self, a: List[float], b: List[float]) -> float:
        """Bias-corrected effect size metric"""
        n1, n2 = len(a), len(b)
        var_pooled = (sum((x - sum(a)/n1)**2 for x in a) + 
                     sum((x - sum(b)/n2)**2 for x in b)) / (n1 + n2 - 2)
        return (sum(a)/n1 - sum(b)/n2) / math.sqrt(var_pooled)

    def disable_temporarily(self):
        """Temporarily disable statistical testing during degraded mode"""
        self.test_cases = []
        logger.warning("Statistical Evaluator temporarily disabled.")

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Statistical Evaluator ===\n")
    import sys
    app = QApplication(sys.argv)

    try:
        statistics = StatisticalEvaluator()
        datasets = {
            "Control":      [12.3, 14.2, 15.1, 13.8, 16.0, 14.4, 15.2, 13.5, 14.9, 15.3],
            "Treatment_A":  [18.4, 19.2, 17.8, 20.1, 19.5, 18.6, 19.7, 18.2, 19.1, 20.0],
            "Treatment_B":  [15.6, 16.9, 14.2, 17.3, 16.1, 15.8, 17.0, 15.3, 16.2, 16.5]
        }
        results = statistics.evaluate(datasets, report=True)
        logger.info(f"{statistics}")
        if 'report' in results:
            print(results['report'])
        elif 'error' in results:
            printer.pretty("Evaluation failed", results, "error")
            
    except Exception as e:
        printer.pretty("Fatal error during evaluation", str(e), "error")
    
    print("\n=== Statistical Evaluation Complete ===\n")
