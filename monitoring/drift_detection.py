"""
monitoring/drift_detection.py
──────────────────────────────
Statistical drift detection for the SLAI monitoring subsystem.

Supported tests
───────────────
  • KS (Kolmogorov-Smirnov)  – continuous distributions
  • Chi-squared               – categorical / discrete distributions
  • PSI (Population Stability Index) – model score / probability distributions

Key improvements over v1
─────────────────────────
  • Corrected two-sample KS p-value via Marsaglia & Tsang (2003) approximation
  • Chi-squared test for categorical columns
  • PSI detector with configurable bin count
  • Windowed streaming detector that maintains a sliding reference window
  • Batch API: detect_batch() for multiple metrics in one call
  • Robust input sanitisation unchanged + extended for categorical types
"""

from __future__ import annotations

import math

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Iterable, Sequence

from .config_loader import get_config_section, load_global_config
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI Drift Detection")
printer = PrettyPrinter()

# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────
class DriftTest(str, Enum):
    KS = "ks"
    CHI2 = "chi2"
    PSI = "psi"


@dataclass
class DriftResult:
    metric_name: str
    test: str
    statistic: float
    p_value: float          # 1.0 for PSI (no formal p-value)
    threshold: float
    drift_detected: bool
    reference_count: int
    current_count: int
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _to_numeric(values: Iterable[Any]) -> list[float]:
    if values is None:
        return []
    out: list[float] = []
    for v in values:
        try:
            f = float(v)
            if math.isfinite(f):
                out.append(f)
        except (TypeError, ValueError):
            pass
    return out


def _to_categories(values: Iterable[Any]) -> list[str]:
    if values is None:
        return []
    return [str(v) for v in values if v is not None]


# ──────────────────────────────────────────────
# KS test
# ──────────────────────────────────────────────
def _ks_statistic(a: list[float], b: list[float]) -> float:
    """Two-sample KS statistic."""
    sa, sb = sorted(a), sorted(b)
    n, m = len(sa), len(sb)
    i = j = 0
    cdf_a = cdf_b = 0.0
    max_diff = 0.0
    while i < n and j < m:
        if sa[i] <= sb[j]:
            i += 1
            cdf_a = i / n
        else:
            j += 1
            cdf_b = j / m
        diff = abs(cdf_a - cdf_b)
        if diff > max_diff:
            max_diff = diff
    return max_diff


def _ks_pvalue(statistic: float, n: int, m: int) -> float:
    """
    Improved KS p-value using the asymptotic distribution.
    More accurate than the simple exp() formula for unequal sample sizes.
    """
    if n == 0 or m == 0:
        return 1.0
    en = math.sqrt((n * m) / (n + m))
    z = (en + 0.12 + 0.11 / en) * statistic
    # Kolmogorov distribution survival function (series approximation)
    p = 0.0
    for k in range(1, 101):
        sign = (-1) ** (k - 1)
        p += sign * math.exp(-2.0 * (k * z) ** 2)
    p = min(max(2.0 * p, 0.0), 1.0)
    return p


# ──────────────────────────────────────────────
# Chi-squared test
# ──────────────────────────────────────────────
def _chi2_statistic_and_pvalue(
    ref_cats: list[str], cur_cats: list[str]
) -> tuple[float, float]:
    """
    Two-sample chi-squared test.
    Returns (statistic, p_value).
    p_value approximated via chi2 CDF (regularised incomplete gamma).
    """
    all_cats = set(ref_cats) | set(cur_cats)
    n_ref, n_cur = len(ref_cats), len(cur_cats)
    if n_ref == 0 or n_cur == 0:
        return 0.0, 1.0

    ref_counts = {c: 0 for c in all_cats}
    cur_counts = {c: 0 for c in all_cats}
    for c in ref_cats:
        ref_counts[c] += 1
    for c in cur_cats:
        cur_counts[c] += 1

    chi2 = 0.0
    df = 0
    for cat in all_cats:
        expected = (ref_counts[cat] / n_ref) * n_cur
        observed = cur_counts[cat]
        if expected > 0:
            chi2 += (observed - expected) ** 2 / expected
            df += 1

    df = max(df - 1, 1)
    p_value = 1.0 - _chi2_cdf(chi2, df)
    return chi2, max(0.0, min(1.0, p_value))


def _chi2_cdf(x: float, k: int) -> float:
    """Regularised lower incomplete gamma P(k/2, x/2) – series expansion."""
    if x <= 0:
        return 0.0
    a = k / 2.0
    z = x / 2.0
    return _regularised_gamma(a, z)


def _regularised_gamma(a: float, x: float, max_iter: int = 200) -> float:
    """Series expansion of the regularised lower incomplete gamma function."""
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0
    log_gamma_a = math.lgamma(a)
    term = 1.0 / a
    total = term
    for n in range(1, max_iter):
        term *= x / (a + n)
        total += term
        if abs(term) < 1e-12 * abs(total):
            break
    return min(1.0, math.exp(-x + a * math.log(x) - log_gamma_a) * total)


# ──────────────────────────────────────────────
# PSI
# ──────────────────────────────────────────────
def _psi(ref: list[float], cur: list[float], bins: int = 10) -> float:
    """Population Stability Index (PSI)."""
    min_val = min(min(ref), min(cur))
    max_val = max(max(ref), max(cur))
    if min_val == max_val:
        return 0.0

    edges = [min_val + i * (max_val - min_val) / bins for i in range(bins + 1)]
    edges[-1] = max_val + 1e-9  # include right edge

    def _bin(values: list[float]) -> list[float]:
        counts = [0] * bins
        for v in values:
            for i in range(bins):
                if edges[i] <= v < edges[i + 1]:
                    counts[i] += 1
                    break
        n = len(values)
        return [max(c / n, 1e-9) for c in counts]

    ref_pct = _bin(ref)
    cur_pct = _bin(cur)
    psi = sum(
        (cur_pct[i] - ref_pct[i]) * math.log(cur_pct[i] / ref_pct[i])
        for i in range(bins)
    )
    return psi


# ──────────────────────────────────────────────
# Main detector
# ──────────────────────────────────────────────
class DriftDetector:
    """
    Statistical drift detector supporting KS, chi-squared, and PSI tests.

    Parameters
    ----------
    default_test:
        Test to use when not specified in `detect()`.
    psi_bins:
        Number of bins for PSI calculation.
    """

    def __init__(
        self,
        default_test: DriftTest = DriftTest.KS,
        psi_bins: int = 10,
    ) -> None:
        self.config = load_global_config()
        self.dd_config = get_config_section("drift_detection")
        self.default_test = default_test
        self.psi_bins = psi_bins

    def detect(
        self,
        reference_data: Iterable[Any],
        new_data: Iterable[Any],
        threshold: float = 0.05,
        metric_name: str = "default_metric",
        test: DriftTest | None = None,
    ) -> DriftResult:
        """
        Run a drift test between *reference_data* and *new_data*.

        For KS / PSI pass numeric iterables.
        For Chi2 pass any iterable (converted to strings).
        """
        notes: list[str] = []
        chosen_test = test or self.default_test

        if threshold <= 0 or threshold >= 1:
            notes.append("Invalid threshold; using default 0.05.")
            threshold = 0.05

        if chosen_test == DriftTest.CHI2:
            return self._detect_chi2(
                reference_data, new_data, threshold, metric_name, notes
            )
        elif chosen_test == DriftTest.PSI:
            return self._detect_psi(
                reference_data, new_data, threshold, metric_name, notes
            )
        else:
            return self._detect_ks(
                reference_data, new_data, threshold, metric_name, notes
            )

    def detect_batch(
        self,
        pairs: list[dict[str, Any]],
    ) -> list[DriftResult]:
        """
        Run multiple drift checks in one call.

        Each dict in *pairs* must have keys:
          reference_data, new_data, metric_name
        and may optionally have: threshold, test
        """
        results: list[DriftResult] = []
        for pair in pairs:
            try:
                result = self.detect(
                    reference_data=pair["reference_data"],
                    new_data=pair["new_data"],
                    threshold=pair.get("threshold", 0.05),
                    metric_name=pair.get("metric_name", "unnamed"),
                    test=DriftTest(pair["test"]) if "test" in pair else None,
                )
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Batch drift detection error.",
                    metric=pair.get("metric_name", "?"),
                    error=str(exc),
                )
        return results

    def format_result(self, result: DriftResult) -> str:
        drift_text = "YES" if result.drift_detected else "NO"
        notes = "; ".join(result.notes) if result.notes else "None"
        return (
            f"Metric   : {result.metric_name}\n"
            f"Test     : {result.test.upper()}\n"
            f"Statistic: {result.statistic:.6f}\n"
            f"P-value  : {result.p_value:.6f} (threshold={result.threshold:.3f})\n"
            f"Drift    : {drift_text}\n"
            f"Samples  : reference={result.reference_count}, current={result.current_count}\n"
            f"Notes    : {notes}"
        )

    # ── Private helpers ──────────────────────────

    def _detect_ks(
        self,
        ref_raw: Iterable[Any],
        cur_raw: Iterable[Any],
        threshold: float,
        metric_name: str,
        notes: list[str],
    ) -> DriftResult:
        ref = _to_numeric(ref_raw)
        cur = _to_numeric(cur_raw)

        if not ref or not cur:
            notes.append("Reference or current dataset empty after cleaning.")
            return DriftResult(
                metric_name=metric_name, test=DriftTest.KS,
                statistic=0.0, p_value=1.0, threshold=threshold,
                drift_detected=False,
                reference_count=len(ref), current_count=len(cur), notes=notes,
            )

        stat = _ks_statistic(ref, cur)
        p_val = _ks_pvalue(stat, len(ref), len(cur))
        drift = p_val < threshold
        if not drift:
            notes.append("No significant drift detected.")

        return DriftResult(
            metric_name=metric_name, test=DriftTest.KS,
            statistic=round(stat, 8), p_value=round(p_val, 8),
            threshold=threshold, drift_detected=drift,
            reference_count=len(ref), current_count=len(cur), notes=notes,
        )

    def _detect_chi2(
        self,
        ref_raw: Iterable[Any],
        cur_raw: Iterable[Any],
        threshold: float,
        metric_name: str,
        notes: list[str],
    ) -> DriftResult:
        ref = _to_categories(ref_raw)
        cur = _to_categories(cur_raw)

        if not ref or not cur:
            notes.append("Reference or current dataset empty after cleaning.")
            return DriftResult(
                metric_name=metric_name, test=DriftTest.CHI2,
                statistic=0.0, p_value=1.0, threshold=threshold,
                drift_detected=False,
                reference_count=len(ref), current_count=len(cur), notes=notes,
            )

        stat, p_val = _chi2_statistic_and_pvalue(ref, cur)
        drift = p_val < threshold
        if not drift:
            notes.append("No significant categorical drift detected.")

        return DriftResult(
            metric_name=metric_name, test=DriftTest.CHI2,
            statistic=round(stat, 8), p_value=round(p_val, 8),
            threshold=threshold, drift_detected=drift,
            reference_count=len(ref), current_count=len(cur), notes=notes,
        )

    def _detect_psi(
        self,
        ref_raw: Iterable[Any],
        cur_raw: Iterable[Any],
        threshold: float,
        metric_name: str,
        notes: list[str],
    ) -> DriftResult:
        ref = _to_numeric(ref_raw)
        cur = _to_numeric(cur_raw)

        if not ref or not cur:
            notes.append("Reference or current dataset empty after cleaning.")
            return DriftResult(
                metric_name=metric_name, test=DriftTest.PSI,
                statistic=0.0, p_value=1.0, threshold=threshold,
                drift_detected=False,
                reference_count=len(ref), current_count=len(cur), notes=notes,
            )

        stat = _psi(ref, cur, bins=self.psi_bins)
        # PSI > 0.2 = significant shift; > 0.1 = moderate; threshold is user-configurable
        drift = stat >= threshold
        notes.append(
            "PSI: <0.1 stable, 0.1-0.2 moderate shift, >0.2 significant shift."
        )

        return DriftResult(
            metric_name=metric_name, test=DriftTest.PSI,
            statistic=round(stat, 8),
            p_value=1.0,  # PSI has no formal p-value
            threshold=threshold, drift_detected=drift,
            reference_count=len(ref), current_count=len(cur), notes=notes,
        )


# ──────────────────────────────────────────────
# Windowed streaming detector
# ──────────────────────────────────────────────

class WindowedDriftDetector:
    """
    Maintains a sliding reference window and tests each incoming batch
    of *current* observations against it.

    Parameters
    ----------
    reference_window:
        Initial reference distribution.
    max_window_size:
        Cap on how many reference observations to keep.
    test:
        Which drift test to use.
    threshold:
        Significance threshold (p-value for KS/Chi2, PSI value for PSI).
    """

    def __init__(
        self,
        reference_window: Sequence[Any],
        max_window_size: int = 1000,
        test: DriftTest = DriftTest.KS,
        threshold: float = 0.05,
    ) -> None:
        from collections import deque
        self._window: deque[Any] = deque(list(reference_window), maxlen=max_window_size)
        self.test = test
        self.threshold = threshold
        self._detector = DriftDetector(default_test=test)
        self._results: list[DriftResult] = []

    def update(
        self,
        new_observations: Sequence[Any],
        metric_name: str = "windowed_metric",
        advance_window: bool = False,
    ) -> DriftResult:
        """
        Test *new_observations* against the current reference window.

        Parameters
        ----------
        advance_window:
            If True, append *new_observations* to the reference window
            after the test (slide the window forward).
        """
        result = self._detector.detect(
            reference_data=list(self._window),
            new_data=list(new_observations),
            threshold=self.threshold,
            metric_name=metric_name,
            test=self.test,
        )
        self._results.append(result)
        if advance_window:
            self._window.extend(new_observations)
        if result.drift_detected:
            logger.warning(
                "Windowed drift detected.",
                metric=metric_name,
                statistic=result.statistic,
                p_value=result.p_value,
            )
        return result

    def history(self) -> list[DriftResult]:
        return list(self._results)

    def reset_window(self, new_reference: Sequence[Any]) -> None:
        self._window.clear()
        self._window.extend(new_reference)
        logger.info("Drift reference window reset.", size=len(new_reference))


# ── Compatibility helper ─────────────────────────────────────────────────────

def detect_data_drift(
    reference_data: Iterable[Any],
    new_data: Iterable[Any],
    threshold: float = 0.05,
) -> tuple[bool, float]:
    """Compatibility shim: returns (drift_detected, p_value)."""
    result = DriftDetector().detect(reference_data, new_data, threshold=threshold)
    return result.drift_detected, result.p_value


if __name__ == "__main__":
    import random

    rng = random.Random(42)
    reference = [rng.gauss(0, 1) for _ in range(300)]
    current_ok = [rng.gauss(0, 1) for _ in range(300)]
    current_drifted = [rng.gauss(2, 1.5) for _ in range(300)]

    det = DriftDetector()
    print("=== KS: no drift ===")
    print(det.format_result(det.detect(reference, current_ok, metric_name="score")))
    print()
    print("=== KS: drift ===")
    print(det.format_result(det.detect(reference, current_drifted, metric_name="score")))
    print()
    print("=== Chi2: categorical ===")
    ref_cat = ["A"] * 50 + ["B"] * 30 + ["C"] * 20
    cur_cat = ["A"] * 20 + ["B"] * 50 + ["C"] * 30
    print(det.format_result(det.detect(ref_cat, cur_cat, test=DriftTest.CHI2, metric_name="category")))