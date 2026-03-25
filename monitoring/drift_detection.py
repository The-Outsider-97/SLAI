from __future__ import annotations

from dataclasses import dataclass, asdict
from math import exp, sqrt
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class DriftResult:
    metric_name: str
    statistic: float
    p_value: float
    threshold: float
    drift_detected: bool
    reference_count: int
    current_count: int
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DriftDetector:
    """KS‑test based drift detection with robust input sanitisation."""

    def detect(
        self,
        reference_data: Iterable[float],
        new_data: Iterable[float],
        threshold: float = 0.05,
        metric_name: str = "default_metric",
    ) -> DriftResult:
        notes: List[str] = []

        # Validate threshold
        if threshold <= 0 or threshold >= 1:
            notes.append("Invalid threshold; using default 0.05.")
            threshold = 0.05

        # Convert to numeric lists, discarding non‑numeric values
        ref = self._to_numeric_list(reference_data)
        cur = self._to_numeric_list(new_data)

        if not ref or not cur:
            notes.append("Reference or current dataset empty after cleaning.")
            return DriftResult(
                metric_name=metric_name,
                statistic=0.0,
                p_value=1.0,
                threshold=threshold,
                drift_detected=False,
                reference_count=len(ref),
                current_count=len(cur),
                notes=notes,
            )

        statistic = self._ks_statistic(ref, cur)
        p_value = self._approximate_ks_pvalue(statistic, len(ref), len(cur))
        drift_detected = p_value < threshold

        if not drift_detected:
            notes.append("No significant drift detected based on current threshold.")

        return DriftResult(
            metric_name=metric_name,
            statistic=float(statistic),
            p_value=float(p_value),
            threshold=threshold,
            drift_detected=drift_detected,
            reference_count=len(ref),
            current_count=len(cur),
            notes=notes,
        )

    def format_result(self, result: DriftResult) -> str:
        drift_text = "YES" if result.drift_detected else "NO"
        notes = "; ".join(result.notes) if result.notes else "None"
        return (
            f"Metric: {result.metric_name}\n"
            f"Statistic: {result.statistic:.6f}\n"
            f"P-value: {result.p_value:.6f} (threshold={result.threshold:.3f})\n"
            f"Drift detected: {drift_text}\n"
            f"Samples: reference={result.reference_count}, current={result.current_count}\n"
            f"Notes: {notes}"
        )

    @staticmethod
    def _to_numeric_list(values: Iterable[float]) -> List[float]:
        if values is None:
            return []
        cleaned = []
        for v in values:
            try:
                f = float(v)
                if f == f and f not in (float("inf"), float("-inf")):
                    cleaned.append(f)
            except (TypeError, ValueError):
                continue
        return cleaned

    @staticmethod
    def _ks_statistic(sample_a: List[float], sample_b: List[float]) -> float:
        a = sorted(sample_a)
        b = sorted(sample_b)
        n, m = len(a), len(b)
        i = j = 0
        cdf_a = cdf_b = 0.0
        max_diff = 0.0

        while i < n and j < m:
            if a[i] <= b[j]:
                i += 1
                cdf_a = i / n
            else:
                j += 1
                cdf_b = j / m
            diff = abs(cdf_a - cdf_b)
            if diff > max_diff:
                max_diff = diff
        return max_diff

    @staticmethod
    def _approximate_ks_pvalue(statistic: float, n: int, m: int) -> float:
        en = sqrt((n * m) / (n + m))
        if en <= 0:
            return 1.0
        value = 2.0 * exp(-2.0 * ((en * statistic) ** 2))
        return max(0.0, min(1.0, value))


def detect_data_drift(reference_data, new_data, threshold=0.05):
    """Compatibility helper: returns (drift_detected, p_value)."""
    result = DriftDetector().detect(reference_data, new_data, threshold=threshold)
    return result.drift_detected, result.p_value


if __name__ == "__main__":
    reference = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    current = [0.4, 0.5, 0.6, 0.8, 0.9, 1.0]
    detector = DriftDetector()
    drift = detector.detect(reference, current, metric_name="demo_distribution")
    print(detector.format_result(drift))
