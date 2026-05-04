from __future__ import annotations

"""
SLAI Interpretability Helper
============================

Production-oriented utilities for translating model evaluation, certification,
security, validation, and attention-analysis signals into readable operational
explanations.

Design goals:
- Keep explanations deterministic, auditable, and easy to test.
- Preserve the existing public helper API while making it safer and richer.
- Validate inputs before interpretation so bad telemetry is surfaced early.
- Avoid over-compressing interpretability output; important context is retained.
"""

import math
import sys
import numpy as np

from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple, Union

from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI Interpretability Helper")
printer = PrettyPrinter()

Numeric = Union[int, float, np.integer, np.floating]
MetricMapping = Mapping[str, Any]


class InterpretabilityHelper:
    """
    Helper to explain evaluation results in plain language with enhanced
    interpretability features.

    The class intentionally exposes static methods so it remains lightweight and
    backwards-compatible with the original helper usage pattern:

        helper = InterpretabilityHelper()
        helper.explain_performance(0.85)

    Each method returns a string that can be logged, rendered in reports, or used
    directly in CLI output. Input validation is strict enough for production use,
    but common optional fields remain optional to support partial telemetry.
    """

    DEFAULT_PERFORMANCE_THRESHOLD = 0.8
    DEFAULT_RISK_BANDS: Tuple[Tuple[str, float], ...] = (
        ("CRITICAL", 0.85),
        ("HIGH", 0.70),
        ("MODERATE", 0.40),
        ("LOW", 0.00),
    )
    DEFAULT_ANOMALY_THRESHOLDS: Dict[str, float] = {
        "critical": 0.90,
        "high": 0.70,
        "medium": 0.50,
    }
    EPSILON = 1e-12

    # ------------------------------------------------------------------
    # Generic validation and formatting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_float(value: Any, name: str, *, allow_nan: bool = False) -> float:
        """Coerce a value to float and reject non-finite telemetry by default."""
        if isinstance(value, bool):
            raise TypeError(f"{name} must be numeric, not bool.")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be numeric; received {type(value).__name__}.") from exc
        if not allow_nan and not math.isfinite(result):
            raise ValueError(f"{name} must be finite; received {result!r}.")
        return result

    @staticmethod
    def _as_int(value: Any, name: str, *, minimum: int = 0) -> int:
        """Coerce a value to int while rejecting bools and negative counts."""
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer count, not bool.")
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be an integer count.") from exc
        if result < minimum:
            raise ValueError(f"{name} must be >= {minimum}; received {result}.")
        return result

    @staticmethod
    def _ensure_mapping(value: Any, name: str) -> MetricMapping:
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping/dict-like object.")
        return value

    @staticmethod
    def _format_float(value: float, digits: int = 3) -> str:
        return f"{value:.{digits}f}"

    @staticmethod
    def _format_percent(value: float, digits: int = 1) -> str:
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}%}"

    @staticmethod
    def _join_non_empty(lines: Sequence[str]) -> str:
        return "\n".join(line for line in lines if line is not None and line != "")

    @staticmethod
    def _severity_from_score(score: float, bands: Sequence[Tuple[str, float]]) -> str:
        """Return the first severity whose threshold is met by score."""
        ordered_bands = sorted(bands, key=lambda item: item[1], reverse=True)
        for label, threshold in ordered_bands:
            if score >= threshold:
                return label
        return ordered_bands[-1][0] if ordered_bands else "UNKNOWN"

    @staticmethod
    def _validate_probability_like(value: float, name: str) -> None:
        """Validate metrics that are expected to be in [0, 1]."""
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be between 0 and 1; received {value}.")

    @classmethod
    def _normalize_rows(cls, matrix: np.ndarray) -> np.ndarray:
        """Return a row-normalized copy, preserving zero rows as zeros."""
        row_sums = matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.divide(
                matrix,
                row_sums,
                out=np.zeros_like(matrix, dtype=float),
                where=np.abs(row_sums) > cls.EPSILON,
            )
        return normalized

    @classmethod
    def _coerce_attention_matrix(
        cls,
        attention: Union[np.ndarray, Sequence[Sequence[Numeric]]],
        tokens: Optional[Sequence[str]] = None,
        *,
        require_square: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """Validate and optionally row-normalize an attention matrix."""
        matrix = np.asarray(attention, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"attention must be a 2D matrix; received shape {matrix.shape}.")
        if matrix.size == 0:
            raise ValueError("attention matrix cannot be empty.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("attention matrix contains NaN or infinite values.")
        if np.any(matrix < 0):
            raise ValueError("attention weights must be non-negative.")
        if require_square and matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"attention must be square; received shape {matrix.shape}.")
        if tokens is not None and len(tokens) != matrix.shape[0]:
            raise ValueError(
                f"tokens length ({len(tokens)}) must match attention rows ({matrix.shape[0]})."
            )
        return cls._normalize_rows(matrix) if normalize else matrix.copy()

    @staticmethod
    def _validate_tokens(tokens: Sequence[str], expected_length: int) -> List[str]:
        if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)):
            raise TypeError("tokens must be a sequence of token strings.")
        if len(tokens) != expected_length:
            raise ValueError(f"Expected {expected_length} tokens, received {len(tokens)}.")
        sanitized = [str(token) for token in tokens]
        if any(token == "" for token in sanitized):
            raise ValueError("tokens cannot contain empty strings.")
        return sanitized

    @classmethod
    def _entropy_from_probabilities(cls, probabilities: np.ndarray) -> float:
        probabilities = np.asarray(probabilities, dtype=float)
        probabilities = probabilities[probabilities > cls.EPSILON]
        if probabilities.size == 0:
            return 0.0
        return float(-np.sum(probabilities * np.log2(probabilities)))

    @staticmethod
    def _safe_rate(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else float("nan")

    # ------------------------------------------------------------------
    # Evaluation, risk, compliance, and certification explanations
    # ------------------------------------------------------------------
    @staticmethod
    def explain_performance(score: float, threshold: float = DEFAULT_PERFORMANCE_THRESHOLD) -> str:
        """
        Explain whether a performance score passes its acceptance threshold.

        The output includes pass/fail status, margin, severity, and actionability
        instead of only saying that performance is acceptable or low.
        """
        score_value = InterpretabilityHelper._as_float(score, "score")
        threshold_value = InterpretabilityHelper._as_float(threshold, "threshold")
        InterpretabilityHelper._validate_probability_like(threshold_value, "threshold")

        margin = score_value - threshold_value
        if margin >= 0.05:
            status = "PASS"
            severity = "healthy"
            action = "Continue monitoring; no immediate remediation is required."
        elif margin >= 0.0:
            status = "PASS"
            severity = "near-threshold"
            action = "Monitor closely because the acceptance margin is small."
        elif margin >= -0.10:
            status = "FAIL"
            severity = "degraded"
            action = "Prioritize targeted evaluation and regression checks before release."
        else:
            status = "FAIL"
            severity = "critical"
            action = "Block production promotion until the reliability gap is remediated."

        return InterpretabilityHelper._join_non_empty(
            [
                f"Performance Status: {status} ({severity})",
                f"- Score: {score_value:.3f}",
                f"- Required Threshold: {threshold_value:.3f}",
                f"- Margin: {margin:+.3f}",
                f"- Interpretation: Performance is {'above' if margin >= 0 else 'below'} the configured acceptance threshold.",
                f"- Recommended Action: {action}",
            ]
        )

    @staticmethod
    def explain_risk(risk: dict) -> str:
        """Explain aggregate risk statistics with severity and uncertainty context."""
        risk_data = InterpretabilityHelper._ensure_mapping(risk, "risk")
        mean = InterpretabilityHelper._as_float(risk_data.get("mean", 0.0), "risk.mean")
        std = InterpretabilityHelper._as_float(
            risk_data.get("std_dev", risk_data.get("std", 0.0)), "risk.std_dev"
        )
        if std < 0:
            raise ValueError("risk.std_dev must be non-negative.")

        threshold = InterpretabilityHelper._as_float(risk_data.get("threshold", 0.70), "risk.threshold")
        p95 = risk_data.get("p95", risk_data.get("percentile_95"))
        p95_value = None if p95 is None else InterpretabilityHelper._as_float(p95, "risk.p95")
        level = InterpretabilityHelper._severity_from_score(mean, InterpretabilityHelper.DEFAULT_RISK_BANDS)
        volatility = "stable" if std < 0.05 else "variable" if std < 0.15 else "volatile"
        confidence_band = (max(0.0, mean - 1.96 * std), mean + 1.96 * std)

        if mean >= threshold:
            action = "Escalate: mean risk breaches the configured threshold."
        elif p95_value is not None and p95_value >= threshold:
            action = "Review tail cases: average risk is acceptable but high-percentile risk breaches the threshold."
        elif std >= 0.15:
            action = "Investigate instability: risk variance is high even though the mean may be acceptable."
        else:
            action = "Continue routine monitoring and retain current controls."

        lines = [
            f"System Risk: {level}",
            f"- Mean Risk: {mean:.3f}",
            f"- Deviation: {std:.3f} ({volatility})",
            f"- 95% Approx. Band: [{confidence_band[0]:.3f}, {confidence_band[1]:.3f}]",
            f"- Threshold: {threshold:.3f}",
        ]
        if p95_value is not None:
            lines.append(f"- Tail Risk p95: {p95_value:.3f}")
        lines.append(f"- Recommended Action: {action}")
        return InterpretabilityHelper._join_non_empty(lines)

    @staticmethod
    def summarize_certification(cert_result: dict) -> str:
        """Summarize certification status, level, evidence, and unresolved blockers."""
        cert = InterpretabilityHelper._ensure_mapping(cert_result, "cert_result")
        status = str(cert.get("status", "UNKNOWN")).upper()
        level = cert.get("level", "UNSPECIFIED")
        confidence = cert.get("confidence")
        unmet = list(cert.get("unmet_criteria", cert.get("unmet_requirements", [])) or [])
        warnings = list(cert.get("warnings", [] ) or [])
        evidence = list(cert.get("evidence", [] ) or [])
        expires_at = cert.get("expires_at", cert.get("valid_until"))

        if status == "PASSED":
            interpretation = "All required certification gates are marked as satisfied."
        elif status == "FAILED":
            interpretation = "One or more certification gates failed and production approval should be blocked."
        elif status in {"CONDITIONAL", "PENDING", "REVIEW"}:
            interpretation = "Certification is not final and requires additional validation before unrestricted use."
        else:
            interpretation = "Certification status is not recognized; treat this as requiring manual review."

        lines = [
            f"Certification Status: {status}",
            f"- Level: {level}",
            f"- Interpretation: {interpretation}",
        ]
        if confidence is not None:
            confidence_value = InterpretabilityHelper._as_float(confidence, "cert_result.confidence")
            lines.append(f"- Confidence: {confidence_value:.1%}")
        if expires_at:
            lines.append(f"- Valid Until: {expires_at}")
        if unmet:
            lines.append("- Unmet Criteria:")
            lines.extend(f"  • {item}" for item in unmet)
        if warnings:
            lines.append("- Warnings:")
            lines.extend(f"  • {item}" for item in warnings)
        if evidence:
            lines.append("- Evidence Reviewed:")
            lines.extend(f"  • {item}" for item in evidence)
        return InterpretabilityHelper._join_non_empty(lines)

    @staticmethod
    def generate_compliance_report(metrics: dict) -> str:
        """Generate a concise but complete compliance report across control domains."""
        data = InterpretabilityHelper._ensure_mapping(metrics, "metrics")
        domains = (
            "safety",
            "security",
            "ethics",
            "privacy",
            "robustness",
            "transparency",
            "governance",
        )
        lines = ["Compliance Summary:"]
        assessed = 0
        passing = 0
        partial = 0
        failing = 0

        for domain in domains:
            raw = data.get(domain, "Not assessed")
            if isinstance(raw, Mapping):
                status = str(raw.get("status", raw.get("result", "Not assessed")))
                notes = raw.get("notes", raw.get("summary"))
            else:
                status = str(raw)
                notes = None

            normalized_status = status.strip().lower()
            if normalized_status not in {"not assessed", "pending", "unknown", "n/a"}:
                assessed += 1
            if normalized_status in {"compliant", "pass", "passed", "ok", "green"}:
                passing += 1
            elif normalized_status in {"partial", "conditional", "warning", "amber"}:
                partial += 1
            elif normalized_status in {"non-compliant", "fail", "failed", "red", "blocked"}:
                failing += 1

            suffix = f" — {notes}" if notes else ""
            lines.append(f"- {domain.title()}: {status}{suffix}")

        overall = data.get("overall")
        if overall is None:
            if failing:
                overall = "Blocked"
            elif partial:
                overall = "Conditional Approval"
            elif assessed and passing == assessed:
                overall = "Compliant"
            else:
                overall = "Pending"

        lines.extend(
            [
                f"- Assessed Domains: {assessed}/{len(domains)}",
                f"- Passing Domains: {passing}",
                f"- Partial Domains: {partial}",
                f"- Failing Domains: {failing}",
                f"Overall Status: {overall}",
            ]
        )

        gaps = list(data.get("gaps", [] ) or [])
        action_items = list(data.get("action_items", data.get("recommendations", [])) or [])
        if gaps:
            lines.append("Open Gaps:")
            lines.extend(f"- {gap}" for gap in gaps)
        if action_items:
            lines.append("Required Actions:")
            lines.extend(f"- {item}" for item in action_items)
        return InterpretabilityHelper._join_non_empty(lines)

    @staticmethod
    def explain_feature_importance(features: dict, top_n: int = 3) -> str:
        """Explain the strongest feature contributions with ordering and share of impact."""
        feature_data = InterpretabilityHelper._ensure_mapping(features, "features")
        if not feature_data:
            return "No feature importance data available."
        if top_n <= 0:
            raise ValueError("top_n must be greater than zero.")

        numeric_features: List[Tuple[str, float]] = []
        for feature, importance in feature_data.items():
            numeric_features.append(
                (str(feature), InterpretabilityHelper._as_float(importance, f"features[{feature!r}]"))
            )

        sorted_features = sorted(numeric_features, key=lambda item: abs(item[1]), reverse=True)
        selected = sorted_features[: min(top_n, len(sorted_features))]
        total_abs = sum(abs(score) for _, score in numeric_features)
        lines = ["Most Influential Factors:"]

        for index, (feature, importance) in enumerate(selected, 1):
            direction = "positive" if importance >= 0 else "negative"
            share = abs(importance) / total_abs if total_abs > 0 else float("nan")
            lines.append(
                f"{index}. {feature}: {importance:.3f} impact ({direction}, {InterpretabilityHelper._format_percent(share)} of absolute influence)"
            )

        remaining = len(sorted_features) - len(selected)
        if remaining > 0:
            residual = sum(abs(score) for _, score in sorted_features[len(selected):])
            residual_share = residual / total_abs if total_abs > 0 else float("nan")
            lines.append(
                f"- Remaining Factors: {remaining} feature(s), {InterpretabilityHelper._format_percent(residual_share)} of absolute influence."
            )
        return InterpretabilityHelper._join_non_empty(lines)

    @staticmethod
    def explain_confusion_matrix(matrix: dict) -> str:
        """Explain confusion-matrix counts and derived classification rates."""
        data = InterpretabilityHelper._ensure_mapping(matrix, "matrix")
        tp = InterpretabilityHelper._as_int(data.get("tp", data.get("true_positive", 0)), "tp")
        fp = InterpretabilityHelper._as_int(data.get("fp", data.get("false_positive", 0)), "fp")
        tn = InterpretabilityHelper._as_int(data.get("tn", data.get("true_negative", 0)), "tn")
        fn = InterpretabilityHelper._as_int(data.get("fn", data.get("false_negative", 0)), "fn")
        total = tp + fp + tn + fn

        accuracy = InterpretabilityHelper._safe_rate(tp + tn, total)
        precision = InterpretabilityHelper._safe_rate(tp, tp + fp)
        recall = InterpretabilityHelper._safe_rate(tp, tp + fn)
        specificity = InterpretabilityHelper._safe_rate(tn, tn + fp)
        fpr = InterpretabilityHelper._safe_rate(fp, fp + tn)
        fnr = InterpretabilityHelper._safe_rate(fn, fn + tp)
        f1 = (
            2 * precision * recall / (precision + recall)
            if math.isfinite(precision) and math.isfinite(recall) and (precision + recall) > 0
            else float("nan")
        )
        balanced_accuracy = (
            (recall + specificity) / 2
            if math.isfinite(recall) and math.isfinite(specificity)
            else float("nan")
        )

        if total == 0:
            quality_note = "No examples were provided; derived metrics are unavailable."
        elif fnr > 0.20:
            quality_note = "False negatives are elevated; missed positive cases require investigation."
        elif fpr > 0.20:
            quality_note = "False positives are elevated; review decision thresholds and negative-class coverage."
        elif math.isfinite(f1) and f1 >= 0.80:
            quality_note = "Classification behavior is strong across precision and recall."
        else:
            quality_note = "Classification behavior is mixed and should be reviewed by class and segment."

        return InterpretabilityHelper._join_non_empty(
            [
                "Error Analysis:",
                f"- True Positives: {tp}",
                f"- False Positives: {fp} (Type I errors)",
                f"- True Negatives: {tn}",
                f"- False Negatives: {fn} (Type II errors)",
                f"- Total Examples: {total}",
                f"- Accuracy: {InterpretabilityHelper._format_percent(accuracy)}",
                f"- Precision: {InterpretabilityHelper._format_percent(precision)}",
                f"- Recall/Sensitivity: {InterpretabilityHelper._format_percent(recall)}",
                f"- Specificity: {InterpretabilityHelper._format_percent(specificity)}",
                f"- F1 Score: {InterpretabilityHelper._format_percent(f1)}",
                f"- Balanced Accuracy: {InterpretabilityHelper._format_percent(balanced_accuracy)}",
                f"- False Positive Rate: {InterpretabilityHelper._format_percent(fpr)}",
                f"- False Negative Rate: {InterpretabilityHelper._format_percent(fnr)}",
                f"- Interpretation: {quality_note}",
            ]
        )

    @staticmethod
    def explain_validation_metrics(metrics: Dict) -> str:
        """Explain validation pass/fail/coverage metrics without assuming all keys exist."""
        data = InterpretabilityHelper._ensure_mapping(metrics, "metrics")
        passed = InterpretabilityHelper._as_int(data.get("passed", 0), "metrics.passed")
        failed = InterpretabilityHelper._as_int(data.get("failed", 0), "metrics.failed")
        skipped = InterpretabilityHelper._as_int(data.get("skipped", 0), "metrics.skipped")
        total = data.get("total")
        total_count = (
            InterpretabilityHelper._as_int(total, "metrics.total")
            if total is not None
            else passed + failed + skipped
        )
        coverage_raw = data.get("coverage")
        coverage = (
            InterpretabilityHelper._as_float(coverage_raw, "metrics.coverage")
            if coverage_raw is not None
            else InterpretabilityHelper._safe_rate(passed + failed, total_count)
        )
        if coverage > 1.0:
            # Preserve compatibility with callers that pass percentage values such as 92.5.
            coverage = coverage / 100.0

        pass_rate = InterpretabilityHelper._safe_rate(passed, passed + failed)
        blockers = list(data.get("blockers", [] ) or [])
        warnings = list(data.get("warnings", [] ) or [])

        if failed > 0:
            interpretation = "Validation is incomplete because one or more requirements failed."
        elif total_count and skipped / total_count > 0.10:
            interpretation = "Validation passed executed checks, but skipped coverage is material."
        elif math.isfinite(coverage) and coverage < 0.80:
            interpretation = "Requirement coverage is below the recommended minimum."
        else:
            interpretation = "Validation evidence is acceptable for the executed requirement set."

        lines = [
            "Validation Summary:",
            f"- Passed: {passed}",
            f"- Failed: {failed}",
            f"- Skipped: {skipped}",
            f"- Total Requirements: {total_count}",
            f"- Requirement Coverage: {InterpretabilityHelper._format_percent(coverage)}",
            f"- Executed Pass Rate: {InterpretabilityHelper._format_percent(pass_rate)}",
            f"- Interpretation: {interpretation}",
        ]
        if blockers:
            lines.append("- Blockers:")
            lines.extend(f"  • {item}" for item in blockers)
        if warnings:
            lines.append("- Warnings:")
            lines.extend(f"  • {item}" for item in warnings)
        return InterpretabilityHelper._join_non_empty(lines)

    # ------------------------------------------------------------------
    # Attention and model-behavior explanations
    # ------------------------------------------------------------------
    @staticmethod
    def explain_attention_pattern(attention: np.ndarray, tokens: List[str]) -> str:
        """Explain attention patterns in natural language with validation and summary statistics."""
        matrix = InterpretabilityHelper._coerce_attention_matrix(attention, tokens, normalize=True)
        token_list = InterpretabilityHelper._validate_tokens(tokens, matrix.shape[0])
        max_idx = tuple(int(i) for i in np.unravel_index(np.argmax(matrix), matrix.shape))
        min_idx = tuple(int(i) for i in np.unravel_index(np.argmin(matrix), matrix.shape))
        diagonal = np.diag(matrix)
        diagonal_mean = float(np.mean(diagonal))
        off_diagonal = matrix.copy()
        np.fill_diagonal(off_diagonal, -1.0)
        strongest_cross_idx = tuple(int(i) for i in np.unravel_index(np.argmax(off_diagonal), off_diagonal.shape))
        row_entropies = [InterpretabilityHelper._entropy_from_probabilities(row) for row in matrix]
        mean_entropy = float(np.mean(row_entropies))
        max_entropy = math.log2(matrix.shape[1]) if matrix.shape[1] > 1 else 0.0
        normalized_entropy = mean_entropy / max_entropy if max_entropy > 0 else 0.0

        lines = [
            "Attention Pattern Explanation:",
            (
                f"- Strongest Focus: '{token_list[max_idx[0]]}' attends most to "
                f"'{token_list[max_idx[1]]}' ({matrix[max_idx]:.3f})."
            ),
            (
                f"- Weakest Link: '{token_list[min_idx[0]]}' to "
                f"'{token_list[min_idx[1]]}' ({matrix[min_idx]:.3f})."
            ),
            (
                f"- Strongest Cross-Token Link: '{token_list[strongest_cross_idx[0]]}' to "
                f"'{token_list[strongest_cross_idx[1]]}' ({matrix[strongest_cross_idx]:.3f})."
            ),
            f"- Mean Diagonal Attention: {diagonal_mean:.3f}",
            f"- Mean Attention Entropy: {mean_entropy:.3f} ({normalized_entropy:.1%} of maximum)",
        ]

        if diagonal_mean > 0.70:
            lines.append("- Interpretation: Strong diagonal pattern suggests token-to-self attention dominance.")
        elif normalized_entropy > 0.80:
            lines.append("- Interpretation: Attention is broadly distributed and may lack a sharply focused rationale.")
        elif normalized_entropy < 0.35:
            lines.append("- Interpretation: Attention is narrow; verify the model is not over-focusing on isolated tokens.")
        else:
            lines.append("- Interpretation: Attention distribution is balanced between focus and context spread.")
        return InterpretabilityHelper._join_non_empty(lines)

    @staticmethod
    def explain_anomaly(anomaly_score: float, thresholds: Dict[str, float]) -> str:
        """Explain attention anomaly scores with configurable threshold semantics."""
        score = InterpretabilityHelper._as_float(anomaly_score, "anomaly_score")
        provided = dict(thresholds or {})
        merged = {**InterpretabilityHelper.DEFAULT_ANOMALY_THRESHOLDS, **provided}
        critical = InterpretabilityHelper._as_float(merged["critical"], "thresholds.critical")
        high = InterpretabilityHelper._as_float(merged["high"], "thresholds.high")
        medium = InterpretabilityHelper._as_float(merged["medium"], "thresholds.medium")
        if not (critical >= high >= medium):
            raise ValueError("Anomaly thresholds must satisfy critical >= high >= medium.")

        if score >= critical:
            label = "CRITICAL"
            interpretation = "Possible adversarial manipulation, severe distribution shift, or telemetry corruption."
            action = "Block automated trust decisions and trigger incident review."
        elif score >= high:
            label = "HIGH"
            interpretation = "Potential attention hijacking or unstable model focus."
            action = "Review examples, segments, and recent model/data changes before deployment."
        elif score >= medium:
            label = "MODERATE"
            interpretation = "Unusual attention patterns were detected but are not yet critical."
            action = "Monitor closely and inspect representative samples."
        else:
            label = "NORMAL"
            interpretation = "Attention patterns are within the configured anomaly range."
            action = "Continue routine monitoring."

        return InterpretabilityHelper._join_non_empty(
            [
                f"Anomaly Status: {label}",
                f"- Score: {score:.3f}",
                f"- Thresholds: medium={medium:.3f}, high={high:.3f}, critical={critical:.3f}",
                f"- Interpretation: {interpretation}",
                f"- Recommended Action: {action}",
            ]
        )

    @staticmethod
    def explain_entropy(entropy: float, normal_range: Tuple[float, float]) -> str:
        """Explain attention entropy values against an expected operating range."""
        entropy_value = InterpretabilityHelper._as_float(entropy, "entropy")
        if len(normal_range) != 2:
            raise ValueError("normal_range must contain exactly two values: (low, high).")
        low = InterpretabilityHelper._as_float(normal_range[0], "normal_range.low")
        high = InterpretabilityHelper._as_float(normal_range[1], "normal_range.high")
        if low > high:
            raise ValueError("normal_range.low must be <= normal_range.high.")

        if entropy_value < low:
            status = "LOW"
            interpretation = "Model focus is too narrow and may ignore contextual information."
            action = "Inspect over-dominant tokens and evaluate context-ablation sensitivity."
        elif entropy_value > high:
            status = "HIGH"
            interpretation = "Model focus is too broad and may lack decisive attribution."
            action = "Inspect diffuse attention heads and compare against confidence/calibration metrics."
        else:
            status = "NORMAL"
            interpretation = "Attention distribution is within the expected operating range."
            action = "No entropy-specific intervention is required."

        return InterpretabilityHelper._join_non_empty(
            [
                f"Attention Entropy Status: {status}",
                f"- Entropy: {entropy_value:.3f}",
                f"- Expected Range: [{low:.3f}, {high:.3f}]",
                f"- Interpretation: {interpretation}",
                f"- Recommended Action: {action}",
            ]
        )

    @staticmethod
    def explain_security_assessment(assessment: Dict) -> str:
        """Generate a human-readable security assessment without assuming mandatory fields."""
        data = InterpretabilityHelper._ensure_mapping(assessment, "assessment")
        secure = bool(data.get("secure", False))
        status = "SECURE" if secure else "VULNERABLE"
        confidence = InterpretabilityHelper._as_float(data.get("confidence", 0.0), "assessment.confidence")
        severity = str(data.get("severity", "LOW" if secure else "HIGH")).upper()
        findings = list(data.get("findings", [] ) or [])
        recommendations = list(data.get("recommendations", [] ) or [])
        controls = list(data.get("controls", [] ) or [])

        lines = [
            f"Security Status: {status}",
            f"- Severity: {severity}",
            f"- Confidence: {confidence:.1%}",
        ]

        lines.append("Findings:")
        if findings:
            lines.extend(f"- {finding}" for finding in findings)
        else:
            lines.append("- No explicit findings were provided.")

        if controls:
            lines.append("Validated Controls:")
            lines.extend(f"- {control}" for control in controls)
        if recommendations:
            lines.append("Recommendations:")
            lines.extend(f"- {rec}" for rec in recommendations)
        elif not secure:
            lines.append("Recommendations:")
            lines.append("- Add concrete remediation steps before approving the assessment.")
        return InterpretabilityHelper._join_non_empty(lines)

    @staticmethod
    def explain_head_importance(importances: List[float]) -> str:
        """Explain attention-head importance distribution across one layer or aggregate view."""
        if importances is None or len(importances) == 0:
            return "No head importance data available."
        values = np.asarray(importances, dtype=float)
        if values.ndim != 1:
            raise ValueError("importances must be a one-dimensional sequence.")
        if not np.all(np.isfinite(values)):
            raise ValueError("importances contain NaN or infinite values.")

        abs_values = np.abs(values)
        max_idx = int(np.argmax(abs_values))
        min_idx = int(np.argmin(abs_values))
        variance = float(np.var(values))
        mean = float(np.mean(values))
        total_abs = float(np.sum(abs_values))
        normalized = abs_values / total_abs if total_abs > 0 else np.zeros_like(abs_values)
        entropy = InterpretabilityHelper._entropy_from_probabilities(normalized)
        max_entropy = math.log2(values.size) if values.size > 1 else 0.0
        concentration = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

        top_indices = np.argsort(abs_values)[::-1][: min(3, values.size)]
        lines = [
            "Head Importance Explanation:",
            f"- Strongest Head: #{max_idx} ({values[max_idx]:.3f})",
            f"- Weakest Head: #{min_idx} ({values[min_idx]:.3f})",
            f"- Mean Importance: {mean:.3f}",
            f"- Variance: {variance:.3f}",
            f"- Concentration: {concentration:.1%}",
            "- Top Heads: "
            + ", ".join(f"#{int(index)}={values[int(index)]:.3f}" for index in top_indices),
        ]
        if variance > 0.10 or concentration > 0.45:
            lines.append("- Interpretation: Head importance is concentrated, suggesting specialized roles.")
        elif total_abs == 0:
            lines.append("- Interpretation: All heads have zero measured importance; verify attribution inputs.")
        else:
            lines.append("- Interpretation: Head importance is broadly distributed, suggesting redundancy or shared processing.")
        return InterpretabilityHelper._join_non_empty(lines)

    @staticmethod
    def attention_to_text(attention: np.ndarray, tokens: List[str]) -> str:
        """Convert attention matrix to a per-token human-readable representation."""
        matrix = InterpretabilityHelper._coerce_attention_matrix(attention, tokens, normalize=True)
        token_list = InterpretabilityHelper._validate_tokens(tokens, matrix.shape[0])
        output = []
        for row_index, row in enumerate(matrix):
            focus_idx = int(np.argmax(row))
            second_idx = int(np.argsort(row)[-2]) if row.size > 1 else focus_idx
            output.append(
                f"When processing '{token_list[row_index]}', the model focuses most on "
                f"'{token_list[focus_idx]}' (attention: {row[focus_idx]:.2f}); "
                f"secondary focus is '{token_list[second_idx]}' (attention: {row[second_idx]:.2f})."
            )
        return "\n".join(output)

    @staticmethod
    def compute_attention_entropy(attention: np.ndarray) -> Dict[str, Union[float, List[float]]]:
        """Compute row-level and aggregate entropy for an attention matrix."""
        matrix = InterpretabilityHelper._coerce_attention_matrix(attention, normalize=True)
        row_entropies = [InterpretabilityHelper._entropy_from_probabilities(row) for row in matrix]
        max_entropy = math.log2(matrix.shape[1]) if matrix.shape[1] > 1 else 0.0
        mean_entropy = float(np.mean(row_entropies)) if row_entropies else 0.0
        return {
            "row_entropy": row_entropies,
            "mean_entropy": mean_entropy,
            "max_entropy": max_entropy,
            "normalized_entropy": mean_entropy / max_entropy if max_entropy > 0 else 0.0,
        }

    @staticmethod
    def attention_summary(
        attention: np.ndarray,
        tokens: List[str],
        *,
        layer: Optional[int] = None,
        head: Optional[int] = None,
        top_k: int = 5,
    ) -> str:
        """Generate a richer attention summary suitable for reports and debugging."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        matrix = InterpretabilityHelper._coerce_attention_matrix(attention, tokens, normalize=True)
        token_list = InterpretabilityHelper._validate_tokens(tokens, matrix.shape[0])
        entropy_stats = InterpretabilityHelper.compute_attention_entropy(matrix)
        pairs: List[Tuple[float, str, str]] = []
        for source_idx, row in enumerate(matrix):
            for target_idx, weight in enumerate(row):
                pairs.append((float(weight), token_list[source_idx], token_list[target_idx]))
        pairs.sort(key=lambda item: item[0], reverse=True)
        selected_pairs = pairs[: min(top_k, len(pairs))]

        context = []
        if layer is not None:
            context.append(f"Layer {layer}")
        if head is not None:
            context.append(f"Head {head}")
        title_suffix = f" ({', '.join(context)})" if context else ""

        lines = [
            f"Attention Summary{title_suffix}:",
            f"- Tokens: {len(token_list)}",
            f"- Normalized Entropy: {float(entropy_stats['normalized_entropy']):.1%}", # type: ignore
            "- Strongest Attention Links:",
        ]
        lines.extend(f"  • {src} → {dst}: {weight:.3f}" for weight, src, dst in selected_pairs)
        return InterpretabilityHelper._join_non_empty(lines)

    # ------------------------------------------------------------------
    # Additional production interpretability report helpers
    # ------------------------------------------------------------------
    @staticmethod
    def explain_fairness_metrics(metrics: Dict[str, Any]) -> str:
        """Explain fairness metrics such as demographic parity and equal opportunity gaps."""
        data = InterpretabilityHelper._ensure_mapping(metrics, "fairness metrics")
        parity_gap = data.get("demographic_parity_gap", data.get("parity_gap"))
        opportunity_gap = data.get("equal_opportunity_gap", data.get("opportunity_gap"))
        group_metrics = data.get("groups", {})
        threshold = InterpretabilityHelper._as_float(data.get("threshold", 0.10), "fairness.threshold")

        lines = ["Fairness Summary:"]
        worst_gap = 0.0
        if parity_gap is not None:
            parity_value = abs(InterpretabilityHelper._as_float(parity_gap, "demographic_parity_gap"))
            worst_gap = max(worst_gap, parity_value)
            lines.append(f"- Demographic Parity Gap: {parity_value:.3f}")
        if opportunity_gap is not None:
            opportunity_value = abs(InterpretabilityHelper._as_float(opportunity_gap, "equal_opportunity_gap"))
            worst_gap = max(worst_gap, opportunity_value)
            lines.append(f"- Equal Opportunity Gap: {opportunity_value:.3f}")
        if isinstance(group_metrics, Mapping) and group_metrics:
            lines.append("- Group Metrics:")
            for group, values in group_metrics.items():
                lines.append(f"  • {group}: {values}")

        status = "PASS" if worst_gap <= threshold else "REVIEW_REQUIRED"
        interpretation = (
            "Observed group-level gaps are within tolerance."
            if status == "PASS"
            else "At least one fairness gap exceeds tolerance and requires mitigation review."
        )
        lines.extend(
            [
                f"- Threshold: {threshold:.3f}",
                f"- Status: {status}",
                f"- Interpretation: {interpretation}",
            ]
        )
        return InterpretabilityHelper._join_non_empty(lines)

    @staticmethod
    def explain_drift(drift: Dict[str, Any]) -> str:
        """Explain feature, label, or embedding drift measurements."""
        data = InterpretabilityHelper._ensure_mapping(drift, "drift")
        score = InterpretabilityHelper._as_float(data.get("score", 0.0), "drift.score")
        threshold = InterpretabilityHelper._as_float(data.get("threshold", 0.20), "drift.threshold")
        method = data.get("method", "unspecified")
        affected = list(data.get("affected_features", data.get("features", [])) or [])
        reference_window = data.get("reference_window", "baseline")
        current_window = data.get("current_window", "current")

        status = "DRIFT_DETECTED" if score >= threshold else "STABLE"
        action = (
            "Run segment analysis, confirm data pipeline health, and consider retraining or recalibration."
            if status == "DRIFT_DETECTED"
            else "Continue scheduled drift monitoring."
        )
        lines = [
            f"Drift Status: {status}",
            f"- Score: {score:.3f}",
            f"- Threshold: {threshold:.3f}",
            f"- Method: {method}",
            f"- Windows: {reference_window} → {current_window}",
            f"- Recommended Action: {action}",
        ]
        if affected:
            lines.append("- Affected Features:")
            lines.extend(f"  • {feature}" for feature in affected)
        return InterpretabilityHelper._join_non_empty(lines)

    @staticmethod
    def explain_calibration(calibration: Dict[str, Any]) -> str:
        """Explain probability calibration quality."""
        data = InterpretabilityHelper._ensure_mapping(calibration, "calibration")
        ece = InterpretabilityHelper._as_float(data.get("ece", data.get("expected_calibration_error", 0.0)), "ece")
        brier = data.get("brier", data.get("brier_score"))
        threshold = InterpretabilityHelper._as_float(data.get("threshold", 0.05), "calibration.threshold")
        status = "CALIBRATED" if ece <= threshold else "MIS_CALIBRATED"
        lines = [
            f"Calibration Status: {status}",
            f"- Expected Calibration Error: {ece:.3f}",
            f"- Threshold: {threshold:.3f}",
        ]
        if brier is not None:
            brier_value = InterpretabilityHelper._as_float(brier, "brier_score")
            lines.append(f"- Brier Score: {brier_value:.3f}")
        lines.append(
            "- Interpretation: "
            + (
                "Predicted probabilities are within the configured calibration tolerance."
                if status == "CALIBRATED"
                else "Predicted probabilities may be over- or under-confident and should be recalibrated."
            )
        )
        return InterpretabilityHelper._join_non_empty(lines)

    @staticmethod
    def generate_interpretability_report(
        *,
        performance: Optional[Dict[str, Any]] = None,
        risk: Optional[Dict[str, Any]] = None,
        certification: Optional[Dict[str, Any]] = None,
        compliance: Optional[Dict[str, Any]] = None,
        feature_importance: Optional[Dict[str, Any]] = None,
        confusion_matrix: Optional[Dict[str, Any]] = None,
        validation: Optional[Dict[str, Any]] = None,
        security: Optional[Dict[str, Any]] = None,
        fairness: Optional[Dict[str, Any]] = None,
        drift: Optional[Dict[str, Any]] = None,
        calibration: Optional[Dict[str, Any]] = None,
        attention: Optional[np.ndarray] = None,
        tokens: Optional[List[str]] = None,
    ) -> str:
        """Compile available interpretability sections into one production report."""
        sections: List[str] = ["SLAI Interpretability Report"]

        if performance is not None:
            sections.append(
                InterpretabilityHelper.explain_performance(
                    performance.get("score", 0.0),
                    performance.get("threshold", InterpretabilityHelper.DEFAULT_PERFORMANCE_THRESHOLD),
                )
            )
        if risk is not None:
            sections.append(InterpretabilityHelper.explain_risk(risk))
        if certification is not None:
            sections.append(InterpretabilityHelper.summarize_certification(certification))
        if compliance is not None:
            sections.append(InterpretabilityHelper.generate_compliance_report(compliance))
        if feature_importance is not None:
            sections.append(InterpretabilityHelper.explain_feature_importance(feature_importance))
        if confusion_matrix is not None:
            sections.append(InterpretabilityHelper.explain_confusion_matrix(confusion_matrix))
        if validation is not None:
            sections.append(InterpretabilityHelper.explain_validation_metrics(validation))
        if security is not None:
            sections.append(InterpretabilityHelper.explain_security_assessment(security))
        if fairness is not None:
            sections.append(InterpretabilityHelper.explain_fairness_metrics(fairness))
        if drift is not None:
            sections.append(InterpretabilityHelper.explain_drift(drift))
        if calibration is not None:
            sections.append(InterpretabilityHelper.explain_calibration(calibration))
        if attention is not None:
            if tokens is None:
                raise ValueError("tokens must be provided when attention is provided.")
            sections.append(InterpretabilityHelper.explain_attention_pattern(attention, tokens))
            sections.append(InterpretabilityHelper.attention_summary(attention, tokens))

        return "\n\n".join(sections)


if __name__ == "__main__":
    print("\n=== Running Interpretability Helper ===\n")
    printer.status("TEST", "Interpretability Helper initialized", "info")

    helper = InterpretabilityHelper()

    # Shared fixtures
    attention_matrix = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.10, 0.70, 0.20],
            [0.10, 0.30, 0.60],
        ]
    )
    tokens = ["The", "movie", "was"]
    features = {
        "data_quality": 0.92,
        "model_complexity": 0.85,
        "training_samples": 0.78,
        "feature_engineering": 0.65,
    }

    # Test performance explanations
    printer.status("TEST", "Performance explanations", "info")
    performance_ok = helper.explain_performance(0.85)
    performance_low = helper.explain_performance(0.72)
    print(performance_ok)
    print(performance_low)
    assert "Performance Status: PASS" in performance_ok
    assert "Performance Status: FAIL" in performance_low

    # Test risk assessment
    printer.status("TEST", "Risk assessments", "info")
    risk_high = helper.explain_risk({"mean": 0.80, "std_dev": 0.15, "p95": 0.93})
    risk_low = helper.explain_risk({"mean": 0.35, "std_dev": 0.02})
    print(risk_high)
    print(risk_low)
    assert "System Risk: HIGH" in risk_high
    assert "System Risk: LOW" in risk_low

    # Test certification summaries
    printer.status("TEST", "Certification summaries", "info")
    cert_passed = helper.summarize_certification(
        {"status": "PASSED", "level": "A", "confidence": 0.97, "evidence": ["unit tests", "security scan"]}
    )
    cert_failed = helper.summarize_certification(
        {"status": "FAILED", "level": "B", "unmet_criteria": ["robustness threshold"]}
    )
    cert_pending = helper.summarize_certification({"status": "PENDING", "level": "C"})
    print(cert_passed)
    print(cert_failed)
    print(cert_pending)
    assert "Certification Status: PASSED" in cert_passed
    assert "Unmet Criteria" in cert_failed
    assert "requires additional validation" in cert_pending

    # Test feature importance
    printer.status("TEST", "Feature importance", "info")
    feature_text = helper.explain_feature_importance(features)
    print(feature_text)
    assert "data_quality" in feature_text
    assert "Remaining Factors" in feature_text

    # Test compliance report
    printer.status("TEST", "Compliance report", "info")
    compliance_text = helper.generate_compliance_report(
        {
            "safety": "Compliant",
            "security": "Partial",
            "ethics": "Compliant",
            "privacy": {"status": "Compliant", "notes": "PII controls active"},
            "overall": "Conditional Approval",
            "gaps": ["Security review follow-up required"],
        }
    )
    print(compliance_text)
    assert "Compliance Summary" in compliance_text
    assert "Conditional Approval" in compliance_text

    # Test confusion matrix
    printer.status("TEST", "Confusion matrix", "info")
    confusion_text = helper.explain_confusion_matrix({"tp": 120, "fp": 15, "tn": 200, "fn": 25})
    print(confusion_text)
    assert "F1 Score" in confusion_text
    assert "False Negative Rate" in confusion_text

    # Test validation metrics
    printer.status("TEST", "Validation metrics", "info")
    validation_text = helper.explain_validation_metrics(
        {
            "passed": 42,
            "failed": 1,
            "skipped": 2,
            "coverage": 0.92,
            "blockers": ["One failed robustness check"],
        }
    )
    print(validation_text)
    assert "Validation Summary" in validation_text
    assert "Blockers" in validation_text

    # Test attention explanations
    printer.status("TEST", "Attention pattern explanation", "info")
    attention_text = helper.explain_attention_pattern(attention_matrix, tokens)
    print(attention_text)
    assert "Strongest Focus" in attention_text
    assert "Mean Attention Entropy" in attention_text

    printer.status("TEST", "Attention to text", "info")
    attention_lines = helper.attention_to_text(attention_matrix, tokens)
    print(attention_lines)
    assert "When processing 'The'" in attention_lines

    printer.status("TEST", "Attention entropy stats", "info")
    entropy_stats = helper.compute_attention_entropy(attention_matrix)
    print(entropy_stats)
    assert "mean_entropy" in entropy_stats
    assert entropy_stats["normalized_entropy"] >= 0.0 # type: ignore

    printer.status("TEST", "Attention summary", "info")
    summary_text = helper.attention_summary(attention_matrix, tokens, layer=1, head=2, top_k=4)
    print(summary_text)
    assert "Layer 1" in summary_text
    assert "Strongest Attention Links" in summary_text

    # Test anomaly and entropy explanations
    printer.status("TEST", "Anomaly explanation", "info")
    anomaly_text = helper.explain_anomaly(0.85, {"critical": 0.90, "high": 0.70, "medium": 0.50})
    print(anomaly_text)
    assert "Anomaly Status: HIGH" in anomaly_text

    printer.status("TEST", "Entropy explanation", "info")
    entropy_text = helper.explain_entropy(0.25, (0.40, 0.80))
    print(entropy_text)
    assert "Attention Entropy Status: LOW" in entropy_text

    # Test security assessment
    printer.status("TEST", "Security assessment", "info")
    security_text = helper.explain_security_assessment(
        {
            "secure": False,
            "confidence": 0.65,
            "severity": "high",
            "findings": ["Low attention entropy detected", "High variance in head importance"],
            "recommendations": ["Investigate attention patterns in layer 4", "Apply attention regularization"],
            "controls": ["input validation", "audit logging"],
        }
    )
    print(security_text)
    assert "Security Status: VULNERABLE" in security_text
    assert "Recommendations" in security_text

    # Test head importance
    printer.status("TEST", "Head importance", "info")
    head_text = helper.explain_head_importance([0.40, 0.10, 0.30, 0.20])
    print(head_text)
    assert "Strongest Head" in head_text

    # Test additional production helpers
    printer.status("TEST", "Fairness, drift, and calibration", "info")
    fairness_text = helper.explain_fairness_metrics(
        {
            "demographic_parity_gap": 0.08,
            "equal_opportunity_gap": 0.06,
            "threshold": 0.10,
            "groups": {"segment_a": {"tpr": 0.91}, "segment_b": {"tpr": 0.85}},
        }
    )
    drift_text = helper.explain_drift(
        {
            "score": 0.24,
            "threshold": 0.20,
            "method": "PSI",
            "affected_features": ["input_length", "source_channel"],
            "reference_window": "train-2025Q4",
            "current_window": "prod-2026Q2",
        }
    )
    calibration_text = helper.explain_calibration({"ece": 0.04, "brier_score": 0.11, "threshold": 0.05})
    print(fairness_text)
    print(drift_text)
    print(calibration_text)
    assert "Fairness Summary" in fairness_text
    assert "Drift Status: DRIFT_DETECTED" in drift_text
    assert "Calibration Status: CALIBRATED" in calibration_text

    # Test compiled report
    printer.status("TEST", "Full interpretability report", "info")
    full_report = helper.generate_interpretability_report(
        performance={"score": 0.86, "threshold": 0.80},
        risk={"mean": 0.41, "std_dev": 0.08},
        certification={"status": "CONDITIONAL", "level": "B", "warnings": ["Pending fairness sign-off"]},
        compliance={"safety": "Compliant", "security": "Partial", "ethics": "Compliant"},
        feature_importance=features,
        confusion_matrix={"tp": 120, "fp": 15, "tn": 200, "fn": 25},
        validation={"passed": 42, "failed": 1, "skipped": 2, "coverage": 0.92},
        security={"secure": False, "confidence": 0.65, "findings": ["Entropy anomaly"]},
        fairness={"demographic_parity_gap": 0.08, "equal_opportunity_gap": 0.06},
        drift={"score": 0.10, "threshold": 0.20},
        calibration={"ece": 0.04, "threshold": 0.05},
        attention=attention_matrix,
        tokens=tokens,
    )
    print(full_report)
    print()
    assert "SLAI Interpretability Report" in full_report
    assert "Attention Pattern Explanation" in full_report

    printer.status("TEST", "All Interpretability Helper checks passed", "success")
    sys.stdout.flush()
    logger.info("Interpretability helper smoke test completed successfully.")
    print("\n=== Test ran successfully ===\n")
