import numpy as np
import yaml, json
import math

from typing import List, Dict, Any
from scipy.stats import shapiro, t as student_t, f as f_dist, norm
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QSize
from datetime import datetime

from src.agents.evaluators.evaluators_memory import EvaluatorsMemory
from logs.logger import get_logger

logger = get_logger("Statistical Evaluator")

CONFIG_PATH = "src/agents/evaluators/configs/evaluator_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

class StatisticalEvaluator:
    def __init__(self, config):
        config = load_config() or {}
        self.config = config.get('statistical_evaluator', {})
        memory = EvaluatorsMemory(config)
        self.memory = memory
        self.significance_threshold = self.config.get('alpha', 0.05)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        self.min_sample_size = self.config.get('min_sample_size', 10)

        logger.info(f"Statistical Evaluator succesfully initialized")

    def evaluate(self, datasets: Dict[str, List[float]], report: bool = False) -> Dict[str, Any]:
        """
        Comprehensive statistical analysis of multiple datasets with optional reporting
        """
        results = {
            'descriptive_stats': self._calculate_descriptive_stats(datasets),
            'hypothesis_tests': self._run_hypothesis_tests(datasets),
            'confidence_intervals': {name: self.compute_confidence_interval(data) 
                                    for name, data in datasets.items()},
            'effect_sizes': self._calculate_all_effect_sizes(datasets)
        }

        if self.config.get('enable_advanced_metrics', True):
            results.update({
                'power_analysis': self._calculate_power_analysis(datasets),
                'normality_tests': self._check_normality(datasets)
            })

        if report:
            results['report'] = self.generate_report(results)

        if self.config.get('store_results', False):
            self.memory.add(
                entry=results,
                tags=["statistical_analysis", f"eval_{datetime.now().date()}"],
                priority="medium"
            )
            if self.memory.access_counter % self.memory.config.get("checkpoint_freq", 500) == 0:
                self.memory.create_checkpoint()

        return results

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

    def _run_hypothesis_tests(self, datasets: Dict[str, List[float]]) -> Dict[str, Any]:
        """Automated hypothesis testing between dataset pairs"""
        tests = {}
        names = list(datasets.keys())
        
        # Pairwise comparisons
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                key = f"{names[i]}_vs_{names[j]}"
                tests[key] = self.t_test(datasets[names[i]], datasets[names[j]])
                
        # ANOVA if more than two groups
        if len(names) > 2:
            tests['anova'] = self.anova(list(datasets.values()))
            
        return tests

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

    def _check_normality(self, datasets: Dict[str, List[float]]) -> Dict[str, bool]:
        """Shapiro-Wilk normality test for all datasets"""
        normality = {}
        for name, data in datasets.items():
            if len(data) < 3:
                normality[name] = False
                continue
            _, p_value = shapiro(data)
            normality[name] = p_value > self.significance_threshold
        return normality

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive statistical report"""
        from src.agents.evaluators.utils.report import get_visualizer
        report = [
            "\n## Statistical Analysis Report",
            f"**Generated**: {datetime.now().isoformat()}\n",
            "### Descriptive Statistics\n"
        ]
        visualizer = get_visualizer()
        visualizer.add_metrics()

        # Descriptive stats table
        for name, stats in results['descriptive_stats'].items():
            report.append(
                f"- **{name}**: μ={stats['mean']:.2f}, σ={stats['std_dev']:.2f}, "
                f"n={stats['sample_size']} (Min={stats['min']:.2f}, Max={stats['max']:.2f})"
            )

        # Hypothesis tests
        report.append("\n### Hypothesis Testing Results\n")
        for test, result in results['hypothesis_tests'].items():
            report.append(
                f"- **{test}**: p={result['p_value']:.3f} "
                f"({'✅ Significant' if result['significant'] else '❌ Not significant'})"
            )
        chart = visualizer.render_temporal_chart(QSize(600, 400), 'success_rate')
        report.append(f"![Statistical Chart](data:image/png;base64,{visualizer._chart_to_base64(chart)})")

        # Effect sizes
        report.append("\n### Effect Sizes (Cohen's d)\n")
        for comparison, d in results['effect_sizes'].items():
            report.append(f"- **{comparison}**: {d:.2f}")

        # Advanced metrics
        if 'power_analysis' in results:
            report.append("\n### Statistical Power Analysis\n")
            for name, power in results['power_analysis'].items():
                report.append(f"- **{name}**: {power:.2%} power")

        report.append(f"\n---\n*Significance threshold: α={self.significance_threshold}*")
        report.append(f"*Report generated by {self.__class__.__name__}*")
        
        return "\n".join(report)

    def compute_confidence_interval(self, samples, confidence=None):
        n = len(samples)
        if n == 0:
            return (0.0, 0.0)
        mean = np.mean(samples)
        std_err = np.std(samples, ddof=1) / math.sqrt(n)
        z = 1.96 if confidence == 0.95 else 2.58
        margin = z * std_err
        return mean - margin, mean + margin

    def t_test(self, sample_a, sample_b):
        if len(sample_a) != len(sample_b) or len(sample_a) == 0:
            return {"p_value": 1.0, "significant": False}
        diffs = np.array(sample_a) - np.array(sample_b)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)
        t_stat = mean_diff / (std_diff / np.sqrt(len(diffs)))
        df = len(diffs) - 1
        p_value = 2 * (1 - student_t.cdf(abs(t_stat), df))
        return {"p_value": p_value, "significant": p_value < 0.05}

    def anova(self, groups):
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

    def effect_size(self, sample_a, sample_b):
        mean_a, mean_b = np.mean(sample_a), np.mean(sample_b)
        pooled_std = np.sqrt((np.std(sample_a, ddof=1)**2 + np.std(sample_b, ddof=1)**2) / 2)
        return 0.0 if pooled_std == 0 else (mean_a - mean_b) / pooled_std

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Statistical Evaluator ===\n")
    import sys
    app = QApplication(sys.argv)
    config = load_config()

    statistics = StatisticalEvaluator(config)
    datasets = {
        "Control": [12.3, 14.2, 15.1, 13.8, 16.0],
        "Treatment_A": [18.4, 19.2, 17.8, 20.1, 19.5],
        "Treatment_B": [15.6, 16.9, 14.2, 17.3, 16.1]
    }
    results = statistics.evaluate(datasets, report=True)
    logger.info(f"{statistics}")
    if 'report' in results:
        print(results['report'])
    print(f"\n* * * * * Phase 2 * * * * *\n")

    print(f"\n* * * * * Phase 3 * * * * *\n")

    print("\n=== Successfully Ran Statistical Evaluator ===\n")
