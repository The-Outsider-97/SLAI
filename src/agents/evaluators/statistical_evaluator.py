import numpy as np
import math
from scipy.stats import t as student_t, f as f_dist

class StatisticalEvaluator:
    def compute_confidence_interval(self, samples, confidence=0.95):
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
