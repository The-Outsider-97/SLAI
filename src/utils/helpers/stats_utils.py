
import math

from functools import lru_cache
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor

# ---------------------------
# Type Definitions
# ---------------------------

Vector = List[float]
HypothesisTestResult = Tuple[float, float] 

# ---------------------------
# Statistical Functions
# ---------------------------

class StatisticalAnalysis:
    """
    Expanded statistical test suite with parallel execution.
    Implements methods from:
    - Sheskin (2020) "Handbook of Parametric and Nonparametric Procedures"
    - Hollander et al. (2013) "Nonparametric Statistical Methods"
    """
    
    @staticmethod
    def independent_t_test(
        sample1: Vector,
        sample2: Vector,
        equal_var: bool = True
    ) -> HypothesisTestResult:
        """
        Student's t-test for independent samples.
        Implements Welch's correction for unequal variances.
        """
        n1, n2 = len(sample1), len(sample2)
        var1, var2 = StatisticalAnalysis._variance(sample1), StatisticalAnalysis._variance(sample2)
        
        if equal_var:
            pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2)
            std_err = math.sqrt(pooled_var * (1/n1 + 1/n2))
        else:
            std_err = math.sqrt(var1/n1 + var2/n2)
            df = (var1/n1 + var2/n2)**2 / ((var1**2)/(n1**2*(n1-1)) + (var2**2)/(n2**2*(n2-1)))
            
        t = (sum(sample1)/n1 - sum(sample2)/n2) / std_err
        df = n1 + n2 - 2 if equal_var else df
        p = 2 * (1 - StatisticalAnalysis._t_cdf(abs(t), df))
        return t, p

    @staticmethod
    def anova(groups: List[Vector]) -> HypothesisTestResult:
        """One-way ANOVA with parallel processing"""
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(
                lambda g: (sum(g), sum(x**2 for x in g), len(g)),
                groups
            ))
        
        grand_total = sum(t[0] for t in results)
        total_ss = sum(t[1] for t in results) - grand_total**2/(sum(t[2] for t in results))
        between_ss = sum(t[0]**2/t[2] for t in results) - grand_total**2/sum(t[2] for t in results)
        within_ss = total_ss - between_ss
        
        df_between = len(groups) - 1
        df_within = sum(t[2] for t in results) - len(groups)
        F = (between_ss/df_between) / (within_ss/df_within)
        p = 1 - StatisticalAnalysis._f_cdf(F, df_between, df_within)
        return F, p

    @staticmethod
    def kolmogorov_smirnov(sample1: Vector, sample2: Vector) -> HypothesisTestResult:
        """Two-sample Kolmogorov-Smirnov test"""
        combined = sorted(set(sample1 + sample2))
        ecdf1 = [sum(1 for x in sample1 if x <= t)/len(sample1) for t in combined]
        ecdf2 = [sum(1 for x in sample2 if x <= t)/len(sample2) for t in combined]
        D = max(abs(e1 - e2) for e1, e2 in zip(ecdf1, ecdf2))
        n = len(sample1)*len(sample2)/(len(sample1) + len(sample2))
        p = math.exp(-2 * n * D**2)
        return D, p

    @staticmethod
    @lru_cache(maxsize=256)
    def _t_cdf(t: float, df: int) -> float:
        """Cumulative distribution function for t-distribution"""
        x = (t + math.sqrt(t**2 + df)) / (2*math.sqrt(t**2 + df))
        return StatisticalAnalysis._beta_cdf(x, df/2, df/2)

    @staticmethod
    @lru_cache(maxsize=256)
    def _beta_cdf(x: float, a: float, b: float) -> float:
        """Incomplete beta function using continued fraction expansion"""
        # Implemented per Press et al. (2007) Numerical Recipes
        EPS = 1e-12
        if x < 0 or x > 1:
            return 0.0
        if x == 0:
            return 0.0
        if x == 1:
            return 1.0
            
        # Continued fraction expansion
        aa, bb = 1.0, 1.0
        fpmin = 1e-30
        qab = a + b
        qap = a + 1.0
        qam = a - 1.0
        c = 1.0
        d = 1.0 - qab*x/qap
        if abs(d) < fpmin:
            d = fpmin
        d = 1.0/d
        h = d
        
        for m in range(1, 10000):
            m2 = 2*m
            aa = m*(b-m)*x/((qam+m2)*(a+m2))
            d = 1.0 + aa*d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa/c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0/d
            h *= d*c
            aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
            d = 1.0 + aa*d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa/c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0/d
            delta = d*c
            h *= delta
            if abs(delta - 1.0) < EPS:
                break
                
        return h * math.exp(math.lgamma(a+b) - math.lgamma(a) - math.lgamma(b) + 
                           a*math.log(x) + b*math.log(1-x)) / a
