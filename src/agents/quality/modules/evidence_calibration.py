"""Evidence-strength and uncertainty calibration for SLAI quality findings.

This module separates defect magnitude/verdict from the strength of evidence
supporting the measurement. It never changes a quality verdict by itself.

Configuration source:
    src/agents/quality/configs/quality_config.yaml -> evidence_calibration
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.quality_error import *
from ..utils.quality_helpers import *
from logs.logger import get_logger, PrettyPrinter, get_log_queue # pyright: ignore[reportMissingImports]


logger = get_logger("Quality Evidence Calibration")
printer = PrettyPrinter()

_STRENGTH_RANK = {
    "insufficient": 0,
    "weak": 1,
    "moderate": 2,
    "strong": 3,
    "very_strong": 4,
}


@dataclass(frozen=True, slots=True)
class EvidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    successes: int
    trials: int

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["width"] = self.width
        return payload


@dataclass(frozen=True, slots=True)
class EvidenceCalibration:
    confidence: float
    uncertainty: float
    evidence_strength: str
    support_count: int
    population_count: int
    coverage: float
    baseline_trust: float
    source_reliability: float
    detector_agreement: float
    components: Dict[str, float]
    reason_codes: Tuple[str, ...]
    interval: Optional[EvidenceInterval] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        payload["interval"] = None if self.interval is None else self.interval.to_dict()
        return payload


class EvidenceCalibrator:
    """Calibrate quality evidence from support, coverage and trust signals."""

    def __init__(self, *, config: Optional[Mapping[str, Any]] = None) -> None:
        printer.status("INIT", "Initializing Evidence Calibrator", "info")
        global_config = load_global_config()
        section = get_config_section("evidence_calibration", config=global_config)
        if config:
            section.update(dict(config))
        self.config = section
        self.enabled = bool(section.get("enabled", True))
        self.confidence_level = bounded_score(
            section.get("confidence_level", 0.95),
            field_name="evidence_calibration.confidence_level",
        )
        if self.confidence_level <= 0.5:
            raise self._config_error("confidence_level must be greater than 0.5")
        self.sample_support_scale = self._positive_float(
            section.get("sample_support_scale", 50.0),
            "evidence_calibration.sample_support_scale",
        )
        self.component_weights = normalize_weights(
            section.get(
                "component_weights",
                {
                    "sample_support": 0.30,
                    "coverage": 0.20,
                    "baseline_trust": 0.20,
                    "source_reliability": 0.15,
                    "detector_agreement": 0.15,
                },
            ),
            expected_keys=(
                "sample_support",
                "coverage",
                "baseline_trust",
                "source_reliability",
                "detector_agreement",
            ),
            field_name="evidence_calibration.component_weights",
            reject_unknown=True,
        )
        thresholds = section.get("strength_thresholds", {})
        if not isinstance(thresholds, Mapping):
            raise self._config_error("strength_thresholds must be a mapping")
        self.strength_thresholds = {
            "weak": bounded_score(thresholds.get("weak", 0.35), field_name="strength.weak"),
            "moderate": bounded_score(thresholds.get("moderate", 0.55), field_name="strength.moderate"),
            "strong": bounded_score(thresholds.get("strong", 0.75), field_name="strength.strong"),
            "very_strong": bounded_score(thresholds.get("very_strong", 0.90), field_name="strength.very_strong"),
        }
        ordered = [
            self.strength_thresholds["weak"],
            self.strength_thresholds["moderate"],
            self.strength_thresholds["strong"],
            self.strength_thresholds["very_strong"],
        ]
        if ordered != sorted(ordered):
            raise self._config_error("strength thresholds must be monotonically non-decreasing")

        aggregation = section.get("batch_aggregation", {})
        if not isinstance(aggregation, Mapping):
            raise self._config_error("batch_aggregation must be a mapping")
        mean_weight = bounded_score(aggregation.get("mean_weight", 0.70), field_name="batch_aggregation.mean_weight")
        minimum_weight = bounded_score(
            aggregation.get("minimum_weight", 0.30),
            field_name="batch_aggregation.minimum_weight",
        )
        total = mean_weight + minimum_weight
        if total <= 0.0:
            raise self._config_error("batch aggregation weights must sum to a positive value")
        self.batch_mean_weight = mean_weight / total
        self.batch_minimum_weight = minimum_weight / total

        baseline_map = section.get("baseline_trust_scores", {})
        if not isinstance(baseline_map, Mapping):
            raise self._config_error("baseline_trust_scores must be a mapping")
        self.baseline_trust_scores = {
            str(key).strip().lower(): bounded_score(
                value,
                field_name=f"baseline_trust_scores.{key}",
            )
            for key, value in baseline_map.items()
        }
        logger.info(f"Evidence Calibrator successfully initialized")

    def wilson_interval(
        self,
        successes: int,
        trials: int,
        *,
        confidence_level: Optional[float] = None,
    ) -> EvidenceInterval:
        """Return a Wilson score interval for a binomial proportion."""
        successes_i = self._nonnegative_int(successes, "successes")
        trials_i = self._nonnegative_int(trials, "trials")
        if trials_i <= 0:
            raise ValueError("trials must be > 0")
        if successes_i > trials_i:
            raise ValueError("successes cannot exceed trials")

        level = self.confidence_level if confidence_level is None else bounded_score(
            confidence_level,
            field_name="confidence_level",
        )
        if level <= 0.5:
            raise ValueError("confidence_level must be > 0.5")

        z_value = self._normal_quantile(0.5 + level / 2.0)
        n = float(trials_i)
        estimate = successes_i / n
        z2 = z_value * z_value
        denominator = 1.0 + z2 / n
        center = (estimate + z2 / (2.0 * n)) / denominator
        margin = (
            z_value
            * math.sqrt((estimate * (1.0 - estimate) / n) + (z2 / (4.0 * n * n)))
            / denominator
        )
        return EvidenceInterval(
            estimate=estimate,
            lower=max(0.0, center - margin),
            upper=min(1.0, center + margin),
            confidence_level=level,
            successes=successes_i,
            trials=trials_i,
        )

    def calibrate(
        self,
        *,
        support_count: int,
        population_count: Optional[int] = None,
        coverage: Optional[float] = None,
        baseline_status: Optional[str] = None,
        source_reliability: Optional[float] = None,
        detector_agreement: Optional[float] = None,
        rate_successes: Optional[int] = None,
        rate_trials: Optional[int] = None,
    ) -> EvidenceCalibration:
        support = self._nonnegative_int(support_count, "support_count")
        population = support if population_count is None else self._nonnegative_int(population_count, "population_count")
        population = max(population, support)
        coverage_value = (
            0.0 if population <= 0 else min(1.0, support / population)
        ) if coverage is None else bounded_score(coverage, field_name="coverage")

        baseline_key = str(baseline_status or "unestablished").strip().lower()
        baseline_trust = self.baseline_trust_scores.get(baseline_key, 0.0)
        reliability = 0.5 if source_reliability is None else bounded_score(
            source_reliability,
            field_name="source_reliability",
        )
        agreement = 0.5 if detector_agreement is None else bounded_score(
            detector_agreement,
            field_name="detector_agreement",
        )
        sample_support = 1.0 - math.exp(-support / self.sample_support_scale)

        interval = None
        interval_precision = None
        if rate_successes is not None and rate_trials is not None:
            interval = self.wilson_interval(rate_successes, rate_trials)
            interval_precision = max(0.0, min(1.0, 1.0 - interval.width))
            sample_support = 0.5 * sample_support + 0.5 * interval_precision

        components = {
            "sample_support": bounded_score(sample_support, field_name="sample_support"),
            "coverage": coverage_value,
            "baseline_trust": baseline_trust,
            "source_reliability": reliability,
            "detector_agreement": agreement,
        }
        confidence = sum(
            components[name] * self.component_weights[name]
            for name in self.component_weights
        )
        confidence = bounded_score(confidence, field_name="evidence_confidence")

        reasons = []
        if support == 0:
            reasons.append("no_sample_support")
        elif sample_support < 0.50:
            reasons.append("limited_sample_support")
        else:
            reasons.append("adequate_sample_support")
        if coverage_value < 0.50:
            reasons.append("low_measurement_coverage")
        if baseline_trust < 0.50:
            reasons.append("weak_or_unestablished_baseline")
        if reliability < 0.50:
            reasons.append("low_source_reliability")
        if agreement < 0.50:
            reasons.append("limited_detector_agreement")
        if interval_precision is not None:
            reasons.append("binomial_interval_calibrated")

        return EvidenceCalibration(
            confidence=confidence,
            uncertainty=1.0 - confidence,
            evidence_strength=self._strength(confidence),
            support_count=support,
            population_count=population,
            coverage=coverage_value,
            baseline_trust=baseline_trust,
            source_reliability=reliability,
            detector_agreement=agreement,
            components=components,
            reason_codes=tuple(reasons),
            interval=interval,
        )

    def calibrate_finding(
        self,
        finding: Mapping[str, Any],
        *,
        population_count: Optional[int] = None,
        baseline_status: Optional[str] = None,
        source_reliability: Optional[float] = None,
        detector_agreement: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Attach evidence metadata without overwriting detector verdict/confidence."""
        payload = dict(finding)
        metrics = payload.get("metrics", {})
        metrics = metrics if isinstance(metrics, Mapping) else {}
        support = self._infer_support(payload, metrics)
        population = self._infer_population(population_count, payload, metrics, support)
        affected = self._infer_affected(payload, metrics)
        rate_successes = None
        rate_trials = None
        if affected is not None and population > 0 and affected <= population:
            rate_successes = affected
            rate_trials = population

        calibration = self.calibrate(
            support_count=support,
            population_count=population,
            baseline_status=baseline_status,
            source_reliability=source_reliability,
            detector_agreement=detector_agreement,
            rate_successes=rate_successes,
            rate_trials=rate_trials,
        )
        payload["evidence"] = calibration.to_dict()
        payload.setdefault("confidence", calibration.confidence)
        return payload

    def aggregate(self, calibrations: Sequence[Union[Mapping[str, Any], EvidenceCalibration]]) -> Dict[str, Any]:
        if not calibrations:
            return {
                "confidence": 0.0,
                "uncertainty": 1.0,
                "evidence_strength": "insufficient",
                "finding_count": 0,
                "mean_confidence": 0.0,
                "minimum_confidence": 0.0,
                "weakest_strength": "insufficient",
            }

        values = []
        strengths = []
        for item in calibrations:
            if isinstance(item, EvidenceCalibration):
                confidence = item.confidence
                strength = item.evidence_strength
            else:
                source = item.get("evidence", item)
                if not isinstance(source, Mapping):
                    continue
                confidence = bounded_score(source.get("confidence", 0.0), field_name="calibration.confidence")
                strength = str(source.get("evidence_strength", self._strength(confidence)))
            values.append(confidence)
            strengths.append(strength)

        if not values:
            return self.aggregate(())
        mean_value = sum(values) / len(values)
        minimum = min(values)
        aggregate_confidence = bounded_score(
            self.batch_mean_weight * mean_value + self.batch_minimum_weight * minimum,
            field_name="aggregate_evidence_confidence",
        )
        weakest = min(strengths, key=lambda item: _STRENGTH_RANK.get(item, -1)) if strengths else "insufficient"
        return {
            "confidence": aggregate_confidence,
            "uncertainty": 1.0 - aggregate_confidence,
            "evidence_strength": self._strength(aggregate_confidence),
            "finding_count": len(values),
            "mean_confidence": mean_value,
            "minimum_confidence": minimum,
            "weakest_strength": weakest,
        }

    @staticmethod
    def _infer_support(finding: Mapping[str, Any], metrics: Mapping[str, Any]) -> int:
        for key in ("support_count", "reviewed_record_count", "record_count", "count", "observations", "current_support"):
            value = finding.get(key, metrics.get(key))
            if value is not None:
                try:
                    return max(0, int(value))
                except (TypeError, ValueError):
                    pass
        return 0

    def _infer_population(
        self,
        explicit: Optional[int],
        finding: Mapping[str, Any],
        metrics: Mapping[str, Any],
        support: int,
    ) -> int:
        if explicit is not None:
            return max(support, self._nonnegative_int(explicit, "population_count"))
        for key in ("record_count", "population_count", "total_records", "trials"):
            value = finding.get(key, metrics.get(key))
            if value is not None:
                try:
                    return max(support, int(value))
                except (TypeError, ValueError):
                    pass
        return support

    @staticmethod
    def _infer_affected(finding: Mapping[str, Any], metrics: Mapping[str, Any]) -> Optional[int]:
        for key in (
            "affected_records",
            "affected_record_count",
            "duplicate_records",
            "outlier_record_count",
            "required_missing_records",
        ):
            value = finding.get(key, metrics.get(key))
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return len(value)
            if value is not None:
                try:
                    return max(0, int(value))
                except (TypeError, ValueError):
                    pass
        return None

    def _strength(self, confidence: float) -> str:
        value = bounded_score(confidence, field_name="confidence")
        if value >= self.strength_thresholds["very_strong"]:
            return "very_strong"
        if value >= self.strength_thresholds["strong"]:
            return "strong"
        if value >= self.strength_thresholds["moderate"]:
            return "moderate"
        if value >= self.strength_thresholds["weak"]:
            return "weak"
        return "insufficient"

    @staticmethod
    def _normal_quantile(p: float) -> float:
        """Acklam rational approximation for inverse standard-normal CDF."""
        if not 0.0 < p < 1.0:
            raise ValueError("p must be within (0, 1)")
        a = (-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239)
        b = (-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572)
        c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783)
        d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416)
        plow = 0.02425
        phigh = 1.0 - plow
        if p < plow:
            q = math.sqrt(-2.0 * math.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
        if p > phigh:
            q = math.sqrt(-2.0 * math.log(1.0 - p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)

    @staticmethod
    def _nonnegative_int(value: Any, field_name: str) -> int:
        if isinstance(value, bool):
            raise TypeError(f"{field_name} must be an integer")
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"{field_name} must be >= 0")
        return parsed

    @staticmethod
    def _positive_float(value: Any, field_name: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise EvidenceCalibrator._config_error(
                f"{field_name} must be a finite positive number",
                context={"value": value},
            )
        return parsed

    @staticmethod
    def _config_error(message: str, *, context: Optional[Mapping[str, Any]] = None) -> DataQualityError:
        return DataQualityError(
            message=message,
            error_type=QualityErrorType.CONFIGURATION_INVALID,
            severity=QualitySeverity.HIGH,
            retryable=False,
            stage=QualityStage.VALIDATION,
            domain=QualityDomain.SYSTEM,
            disposition=QualityDisposition.ESCALATE,
            context=dict(context or {}),
            remediation="Correct evidence_calibration in quality_config.yaml.",
        )


__all__ = ["EvidenceInterval", "EvidenceCalibration", "EvidenceCalibrator"]