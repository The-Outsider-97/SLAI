"""
Production-grade attention monitor for the Safety Agent subsystem.

This module performs mechanistic interpretability and security monitoring over
attention tensors emitted by transformer-style models. It is an observation and
analysis layer only: it does not own model training, intervention policy,
allow/block orchestration, or long-term persistence policy. Those concerns stay
with the neural network, adaptive security, safety guard, secure memory, and
orchestration layers.
"""

from __future__ import annotations

import base64
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dataclasses import asdict, dataclass, field
from io import BytesIO
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from src.utils.interpretability import InterpretabilityHelper  # pyright: ignore[reportMissingImports]
from .utils.config_loader import load_global_config, get_config_section
from .utils.safety_helpers import *
from .utils.security_error import *
from .secure_memory import SecureMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Security Attention Monitor")
printer = PrettyPrinter()

MODULE_VERSION = "2.1.0"
ANALYSIS_SCHEMA_VERSION = "attention_monitor.analysis.v3"
ASSESSMENT_SCHEMA_VERSION = "attention_monitor.security_assessment.v2"
EPSILON = 1e-12


@dataclass(frozen=True)
class AttentionTensorSummary:
    """Audit-safe tensor shape and provenance summary."""

    original_shape: Tuple[int, ...]
    canonical_shape: Tuple[int, int, int, int]
    dtype: str
    device: str
    fingerprint: str
    normalized: bool
    reduced: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttentionSecurityAssessment:
    """Security implications derived from attention behavior."""

    secure: bool
    confidence: float
    severity: str
    risk_score: float
    decision: str
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["confidence"] = clamp_score(data["confidence"])
        data["risk_score"] = clamp_score(data["risk_score"])
        data["findings"] = sanitize_for_logging(data["findings"])
        data["recommendations"] = sanitize_for_logging(data["recommendations"])
        return data


@dataclass(frozen=True)
class AttentionAnalysisResult:
    """Structured attention analysis envelope returned by the monitor."""

    schema_version: str
    analysis_id: str
    timestamp: str
    matrix_summary: AttentionTensorSummary
    metrics: Dict[str, Any]
    security_assessment: AttentionSecurityAssessment
    context: Dict[str, Any] = field(default_factory=dict)
    attention_plot: Optional[str] = None
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp,
            "matrix_summary": self.matrix_summary.to_dict(),
            "metrics": sanitize_for_logging(self.metrics),
            "security_assessment": self.security_assessment.to_dict(),
            "context": sanitize_for_logging(self.context),
            "interpretation": redact_text(self.interpretation),
        }
        if self.attention_plot is not None:
            data["attention_plot"] = self.attention_plot
        return data


class AttentionMonitor(nn.Module):
    """
    Mechanistic interpretability and security monitor for attention tensors.

    The monitor accepts 2D, 3D, or 4D attention tensors and normalizes them into
    a canonical shape of [batch, heads, query_positions, key_positions]. It
    computes global, per-head, and per-row statistics, derives a bounded anomaly
    score, produces a structured security assessment, optionally stores an
    audit-safe record in SecureMemory, and can render a base64 heatmap when
    visualization is enabled in secure_config.yaml.
    """

    def __init__(self, device: Union[str, torch.device] = "cpu"):
        super().__init__()
        self.config = load_global_config()
        self.attention_config = get_config_section("attention_monitor")
        self._validate_configuration()

        self.entropy_threshold = coerce_float(self._cfg("entropy_threshold"), 0.0, minimum=0.0)
        self.uniformity_threshold = coerce_float(self._cfg("uniformity_threshold"), 0.0, minimum=0.0)
        self.anomaly_threshold = coerce_float(self._cfg("anomaly_threshold"), 0.0, minimum=0.0, maximum=1.0)
        self.review_threshold = coerce_float(self._cfg("review_threshold"), 0.0, minimum=0.0, maximum=1.0)
        self.anomaly_detection = coerce_bool(self._cfg("anomaly_detection"), True)
        self.store_analysis = coerce_bool(self._cfg("store_analysis"), True)
        self.visualization = coerce_bool(self._cfg("visualization.enabled"), False)
        self.device = torch.device(device)

        self.memory = SecureMemory()
        self.interpreter = InterpretabilityHelper()
        self._recent_analysis_ids: List[str] = []

        logger.info(
            "Attention Monitor initialized: %s",
            stable_json(
                safe_log_payload(
                    "attention_monitor.initialized",
                    {
                        "module_version": MODULE_VERSION,
                        "entropy_threshold": self.entropy_threshold,
                        "uniformity_threshold": self.uniformity_threshold,
                        "anomaly_threshold": self.anomaly_threshold,
                        "device": str(self.device),
                    },
                )
            ),
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _cfg(self, path: Union[str, Sequence[str]], default: Any = None) -> Any:
        return get_nested(self.attention_config, path, default)

    def _validate_configuration(self) -> None:
        require_keys(
            self.attention_config,
            [
                "entropy_threshold",
                "uniformity_threshold",
                "anomaly_threshold",
                "review_threshold",
                "anomaly_detection",
                "store_analysis",
                "visualization",
                "tensor_validation",
                "security",
                "anomaly_weights",
                "memory",
            ],
            context="attention_monitor",
        )
        for key in ("entropy_threshold", "uniformity_threshold", "anomaly_threshold", "review_threshold"):
            value = coerce_float(self.attention_config.get(key), -1.0)
            if value < 0.0:
                raise ConfigurationTamperingError(
                    config_file_path=str(self.config.get("__config_path__", "secure_config.yaml")),
                    suspicious_change=f"attention_monitor.{key} must be non-negative",
                    component="attention_monitor",
                )
        if coerce_float(self.attention_config["review_threshold"], 0.0) > coerce_float(self.attention_config["anomaly_threshold"], 1.0):
            raise ConfigurationTamperingError(
                config_file_path=str(self.config.get("__config_path__", "secure_config.yaml")),
                suspicious_change="attention_monitor.review_threshold must be <= anomaly_threshold",
                component="attention_monitor",
            )
        validation = self.attention_config.get("tensor_validation", {})
        require_keys(validation, ["max_dimensions", "max_elements", "allow_negative", "normalize_rows"], context="attention_monitor.tensor_validation")
        if coerce_int(validation.get("max_dimensions"), 0) < 2:
            raise ConfigurationTamperingError(
                config_file_path=str(self.config.get("__config_path__", "secure_config.yaml")),
                suspicious_change="attention_monitor.tensor_validation.max_dimensions must be at least 2",
                component="attention_monitor",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_attention(self, attention_matrix: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Comprehensive attention pattern analysis with security insights."""
        printer.status("MONITOR", "Analyzing attention", "info")

        try:
            canonical, aggregate, summary = self._prepare_attention(attention_matrix)
            metrics = self._build_metrics(canonical, aggregate)
            if self.anomaly_detection:
                metrics["anomaly_score"] = self._detect_anomalies_from_metrics(metrics)
                metrics["anomaly"] = metrics["anomaly_score"] >= self.anomaly_threshold
            else:
                metrics["anomaly_score"] = 0.0
                metrics["anomaly"] = False

            assessment = self._assess_security(metrics)
            interpretation = self.get_anomaly_interpretation({**metrics, "security_assessment": assessment.to_dict()})
            analysis_id = generate_identifier("attn")
            safe_context = sanitize_for_logging(context or {})
            attention_plot = None
            if self.visualization:
                attention_plot = self.visualize_attention(aggregate)

            result = AttentionAnalysisResult(
                schema_version=ANALYSIS_SCHEMA_VERSION,
                analysis_id=analysis_id,
                timestamp=utc_iso(),
                matrix_summary=summary,
                metrics=metrics,
                security_assessment=assessment,
                context=safe_context,
                attention_plot=attention_plot,
                interpretation=interpretation,
            )
            result_dict = result.to_dict()
            # Preserve legacy top-level metric keys expected by existing callers.
            result_dict.update(metrics)
            result_dict["security_assessment"] = assessment.to_dict()
            result_dict["anomaly_interpretation"] = interpretation
            if attention_plot is not None:
                result_dict["attention_plot"] = attention_plot

            if self.store_analysis:
                self._store_analysis(result_dict, safe_context)
            self._track_recent_analysis(analysis_id)
            return result_dict
        except SecurityError:
            raise
        except Exception as exc:
            raise wrap_security_exception(
                exc,
                operation="analyze_attention",
                component="attention_monitor",
                context={"context": sanitize_for_logging(context or {})},
                error_type=SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                severity=SecuritySeverity.HIGH,
            ) from exc

    def generate_report(self, analysis: Mapping[str, Any]) -> str:
        """Generate an audit-safe attention analysis report."""
        printer.status("MONITOR", "Generating report", "info")

        safe_analysis = sanitize_for_logging(dict(analysis or {}))
        security = safe_analysis.get("security_assessment", {}) if isinstance(safe_analysis.get("security_assessment"), Mapping) else {}
        findings = security.get("findings", []) if isinstance(security, Mapping) else []
        recommendations = security.get("recommendations", []) if isinstance(security, Mapping) else []
        entropy = coerce_float(safe_analysis.get("entropy"), 0.0)
        normalized_entropy = coerce_float(safe_analysis.get("normalized_entropy"), 0.0)
        uniformity = coerce_float(safe_analysis.get("uniformity"), 0.0)
        anomaly_score = coerce_float(safe_analysis.get("anomaly_score"), 0.0)
        head_importance = safe_analysis.get("head_importance", [])
        focus_pattern = safe_analysis.get("focus_pattern", "unknown")

        report = [
            "# Attention Analysis Report",
            f"**Generated**: {utc_iso()}",
            f"**Analysis ID**: `{safe_analysis.get('analysis_id', 'unknown')}`",
            f"**Decision**: `{security.get('decision', threshold_decision(anomaly_score, block_threshold=self.anomaly_threshold, review_threshold=self.review_threshold))}`",
            f"**Risk Level**: `{categorize_risk(anomaly_score)}`",
            "",
            "## Core Metrics",
            f"- **Entropy**: {entropy:.3f}",
            f"- **Normalized Entropy**: {normalized_entropy:.3f}",
            f"- **Uniformity / Coefficient of Variation**: {uniformity:.3f}",
            f"- **Dispersion**: {coerce_float(safe_analysis.get('dispersion'), 0.0):.3f}",
            f"- **Anomaly Score**: {anomaly_score:.3f}",
            f"- **Focus Pattern**: {focus_pattern}",
            "",
            "## Security Assessment",
            f"- **Status**: {'Secure' if security.get('secure') else 'Needs Review'}",
            f"- **Severity**: {security.get('severity', 'unknown')}",
            f"- **Confidence**: {coerce_float(security.get('confidence'), 0.0):.1%}",
            f"- **Findings**: {', '.join(findings) if isinstance(findings, list) and findings else 'No significant issues detected'}",
        ]
        if isinstance(recommendations, list) and recommendations:
            report.append("- **Recommendations**:")
            report.extend(f"  - {item}" for item in recommendations)

        report.extend([
            "",
            "## Interpretability Insights",
            self.interpreter.explain_risk({
                "mean": anomaly_score,
                "std_dev": uniformity,
                "threshold": self.anomaly_threshold,
            }),
        ])

        if isinstance(head_importance, list) and head_importance:
            report.extend(["", "## Attention Head Importance", self.interpreter.explain_head_importance([float(x) for x in head_importance])])

        if self._cfg("reports.include_context", False):
            report.extend(["", "## Sanitized Context", f"```json\n{stable_json(safe_analysis.get('context', {}))}\n```"])

        if "attention_plot" in analysis and coerce_bool(self._cfg("reports.include_visualization", False)):
            report.extend(["", "## Attention Visualization", f"![Attention Map](data:image/png;base64,{analysis['attention_plot']})"])

        return "\n".join(report)

    def get_anomaly_interpretation(self, analysis: Mapping[str, Any]) -> str:
        """Generate a human-readable interpretation of attention anomalies."""
        printer.status("MONITOR", "Generating interpretation", "info")

        findings: List[str] = []
        entropy = coerce_float(analysis.get("entropy"), 0.0)
        normalized_entropy = coerce_float(analysis.get("normalized_entropy"), 0.0)
        uniformity = coerce_float(analysis.get("uniformity"), 0.0)
        anomaly_score = coerce_float(analysis.get("anomaly_score"), 0.0)
        diagonal_focus = coerce_float(analysis.get("diagonal_focus"), 0.0)
        max_attention = coerce_float(analysis.get("max_attention"), 0.0)

        low_entropy_multiplier = coerce_float(self._cfg("security.low_entropy_multiplier"), 0.7, minimum=0.0)
        high_uniformity_multiplier = coerce_float(self._cfg("security.high_uniformity_multiplier"), 1.5, minimum=0.0)
        sharp_focus_threshold = coerce_float(self._cfg("security.sharp_focus_threshold"), 0.95, minimum=0.0, maximum=1.0)

        if entropy < self.entropy_threshold * low_entropy_multiplier:
            findings.append("Low attention entropy indicates potential over-focusing on a narrow token set")
        if normalized_entropy > coerce_float(self._cfg("security.diffuse_entropy_threshold"), 0.90, minimum=0.0, maximum=1.0):
            findings.append("Very high normalized entropy indicates diffuse attention with weak explanatory focus")
        if uniformity > self.uniformity_threshold * high_uniformity_multiplier:
            findings.append("High attention dispersion suggests inconsistent or unstable attention patterns")
        if max_attention >= sharp_focus_threshold:
            findings.append("Extremely sharp attention peak detected; review for prompt or token hijacking")
        if diagonal_focus >= coerce_float(self._cfg("security.diagonal_dominance_threshold"), 0.80, minimum=0.0, maximum=1.0):
            findings.append("Strong diagonal dominance may indicate token self-focus over contextual reasoning")
        if anomaly_score >= self.anomaly_threshold:
            findings.append("Severe attention anomaly detected; possible adversarial manipulation or telemetry drift")

        return "; ".join(dedupe_preserve_order(findings)) if findings else "Normal attention patterns"

    def visualize_attention(self, matrix: torch.Tensor) -> str:
        """Generate a bounded attention visualization and return it as base64."""
        printer.status("MONITOR", "Visualizing attention", "info")

        if not self.visualization and not coerce_bool(self._cfg("visualization.allow_manual_call", True), True):
            raise UnauthorizedAccessError(
                resource="attention_visualization",
                policy_violated="attention_monitor.visualization.enabled=false",
                attempted_action="visualize_attention",
                component="attention_monitor",
            )
        _, aggregate, _ = self._prepare_attention(matrix) if matrix.dim() != 2 else (None, self._sanitize_matrix(matrix), None)
        max_dim = coerce_int(self._cfg("visualization.max_dimension", 128), 128, minimum=2)
        render_matrix = aggregate.detach().cpu()
        if render_matrix.shape[0] > max_dim or render_matrix.shape[1] > max_dim:
            render_matrix = torch.nn.functional.interpolate(
                render_matrix.unsqueeze(0).unsqueeze(0),
                size=(min(max_dim, render_matrix.shape[0]), min(max_dim, render_matrix.shape[1])),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        figure_size = self._cfg("visualization.figure_size", [8, 6])
        if not isinstance(figure_size, Sequence) or len(figure_size) != 2:
            figure_size = [8, 6]
        plt.figure(figsize=(coerce_float(figure_size[0], 8.0, minimum=1.0), coerce_float(figure_size[1], 6.0, minimum=1.0)))
        plt.imshow(render_matrix.numpy(), interpolation="nearest")
        plt.colorbar()
        plt.title(normalize_text(self._cfg("visualization.title", "Attention Heatmap"), max_length=80))
        plt.xlabel("Key Positions")
        plt.ylabel("Query Positions")
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format="png", dpi=coerce_int(self._cfg("visualization.dpi", 120), 120, minimum=72, maximum=300))
        plt.close()
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("ascii")

    # ------------------------------------------------------------------
    # Compatibility metric helpers
    # ------------------------------------------------------------------

    def _calculate_entropy(self, matrix: torch.Tensor) -> float:
        """Information-theoretic attention entropy with probability normalization."""
        clean = self._sanitize_matrix(matrix)
        flat = clean.flatten()
        total = flat.sum()
        if total <= EPSILON:
            return 0.0
        probabilities = flat / total
        probabilities = probabilities[probabilities > EPSILON]
        return float((-(probabilities * torch.log2(probabilities)).sum()).item())

    def _calculate_uniformity(self, matrix: torch.Tensor) -> float:
        """Coefficient of variation for attention dispersion."""
        clean = self._sanitize_matrix(matrix)
        mean = clean.mean()
        if mean <= EPSILON:
            return 0.0
        return float((clean.std(unbiased=False) / mean).item())

    def _calculate_head_importance(self, matrix: torch.Tensor) -> List[float]:
        """Calculate normalized importance of each attention head."""
        canonical, _, _ = self._prepare_attention(matrix)
        # Variance/entropy hybrid. High-variance heads and low-entropy focused heads receive more weight.
        importances: List[float] = []
        for head_idx in range(canonical.shape[1]):
            head = canonical[:, head_idx, :, :].mean(dim=0)
            dispersion = self._calculate_uniformity(head)
            entropy = self._calculate_entropy(head)
            max_entropy = math.log2(max(head.numel(), 2))
            focus = 1.0 - clamp_score(entropy / max_entropy if max_entropy > 0 else 0.0)
            importances.append(max(0.0, (0.6 * dispersion) + (0.4 * focus)))
        total = sum(importances)
        if total <= EPSILON:
            return [1.0 / max(1, len(importances)) for _ in importances]
        return [value / total for value in importances]

    def _calculate_dispersion(self, matrix: torch.Tensor) -> float:
        """Calculate bounded attention dispersion index."""
        clean = self._sanitize_matrix(matrix)
        max_val = clean.max()
        min_val = clean.min()
        denominator = max_val + min_val
        return float(((max_val - min_val) / denominator).item()) if denominator > EPSILON else 0.0

    def _identify_focus_pattern(self, matrix: torch.Tensor) -> str:
        """Identify dominant focus pattern for a 2D aggregate matrix."""
        clean = self._matrix_to_2d(matrix)
        row_max = clean.max(dim=1).values.mean().item()
        col_max = clean.max(dim=0).values.mean().item()
        diagonal = torch.diag(clean).mean().item() if clean.shape[0] == clean.shape[1] else 0.0
        off_diagonal = clean.clone()
        if clean.shape[0] == clean.shape[1]:
            off_diagonal.fill_diagonal_(0.0)
        off_diag_mean = off_diagonal.mean().item() if off_diagonal.numel() else 0.0
        if diagonal > max(row_max, col_max, off_diag_mean):
            return "diagonal"
        if row_max > col_max:
            return "row-focused"
        if col_max > row_max:
            return "column-focused"
        return "diffuse"

    def _detect_anomalies(self, matrix: torch.Tensor) -> float:
        """Detect anomalous attention patterns from raw matrix statistics."""
        canonical, aggregate, _ = self._prepare_attention(matrix)
        return self._detect_anomalies_from_metrics(self._build_metrics(canonical, aggregate))

    def _assess_security(self, metrics: Mapping[str, Any]) -> AttentionSecurityAssessment:
        """Assess security implications of attention patterns."""
        printer.status("MONITOR", "Assessing security", "info")

        findings: List[str] = []
        recommendations: List[str] = []
        confidence = 1.0
        anomaly_score = clamp_score(metrics.get("anomaly_score", 0.0))
        entropy = coerce_float(metrics.get("entropy"), 0.0)
        normalized_entropy = clamp_score(metrics.get("normalized_entropy", 0.0))
        uniformity = coerce_float(metrics.get("uniformity"), 0.0)
        max_attention = clamp_score(metrics.get("max_attention", 0.0))
        diagonal_focus = clamp_score(metrics.get("diagonal_focus", 0.0))
        head_dominance = clamp_score(metrics.get("head_dominance", 0.0))

        if entropy < self.entropy_threshold * coerce_float(self._cfg("security.low_entropy_multiplier"), 0.7):
            findings.append("Low attention entropy detected - potential overfocusing")
            recommendations.append("Inspect dominant tokens and compare with prompt-injection indicators.")
            confidence *= 0.82
        if normalized_entropy > coerce_float(self._cfg("security.diffuse_entropy_threshold"), 0.90):
            findings.append("Diffuse attention detected - weak or unstable attribution signal")
            recommendations.append("Review examples with confidence calibration and token ablation tests.")
            confidence *= 0.88
        if uniformity > self.uniformity_threshold * coerce_float(self._cfg("security.high_uniformity_multiplier"), 1.5):
            findings.append("High attention variance detected - potential erratic behavior")
            recommendations.append("Compare recent model/data changes and distribution-shift telemetry.")
            confidence *= 0.78
        if max_attention >= coerce_float(self._cfg("security.sharp_focus_threshold"), 0.95):
            findings.append("Extreme attention peak detected - possible token hijacking or brittle focus")
            recommendations.append("Review the triggering token span and prompt boundary handling.")
            confidence *= 0.80
        if diagonal_focus >= coerce_float(self._cfg("security.diagonal_dominance_threshold"), 0.80):
            findings.append("Diagonal attention dominance detected - context may be underused")
            recommendations.append("Inspect whether the model is over-relying on self-token attention.")
            confidence *= 0.90
        if head_dominance >= coerce_float(self._cfg("security.head_dominance_threshold"), 0.75):
            findings.append("Single-head dominance detected - attention behavior may be fragile")
            recommendations.append("Inspect per-head saliency and validate against adversarial examples.")
            confidence *= 0.86
        if anomaly_score >= self.anomaly_threshold:
            findings.append("High anomaly score detected - possible adversarial manipulation")
            recommendations.append("Route this analysis to security review before trusting automated decisions.")
            confidence *= 0.55

        decision = threshold_decision(anomaly_score, block_threshold=self.anomaly_threshold, review_threshold=self.review_threshold)
        severity = categorize_risk(anomaly_score)
        secure = decision == "allow" and not findings
        if not recommendations and secure:
            recommendations.append("Continue routine monitoring and retain baseline comparison telemetry.")
        return AttentionSecurityAssessment(
            secure=secure,
            confidence=clamp_score(confidence),
            severity=severity,
            risk_score=anomaly_score,
            decision=decision,
            findings=dedupe_preserve_order(findings),
            recommendations=dedupe_preserve_order(recommendations),
        )

    # ------------------------------------------------------------------
    # Internal tensor and metric logic
    # ------------------------------------------------------------------

    def _sanitize_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        if not isinstance(matrix, torch.Tensor):
            raise SecurityError(
                SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                "Attention monitor expected a torch.Tensor.",
                severity=SecuritySeverity.HIGH,
                context={"received_type": type(matrix).__name__},
                component="attention_monitor",
                response_action=SecurityResponseAction.BLOCK,
            )
        validation = self._cfg("tensor_validation", {})
        max_dimensions = coerce_int(validation.get("max_dimensions"), 4, minimum=2)
        max_elements = coerce_int(validation.get("max_elements"), 4_000_000, minimum=1)
        if matrix.dim() < 2 or matrix.dim() > max_dimensions:
            raise SecurityError(
                SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                "Attention tensor has unsupported rank.",
                severity=SecuritySeverity.HIGH,
                context={"shape": tuple(matrix.shape), "max_dimensions": max_dimensions},
                component="attention_monitor",
                response_action=SecurityResponseAction.BLOCK,
            )
        if matrix.numel() > max_elements:
            raise ResourceExhaustionError(
                resource_type="attention_tensor_elements",
                current_usage=float(matrix.numel()),
                limit=float(max_elements),
                source_identifier="attention_monitor",
                component="attention_monitor",
            )
        clean = matrix.detach().to(self.device).float()
        if not torch.isfinite(clean).all():
            raise SystemIntegrityError(
                component="attention_monitor.telemetry",
                anomaly_description="Attention tensor contains NaN or infinite values.",
                context={"shape": tuple(matrix.shape)},
            )
        if torch.any(clean < 0.0):
            if coerce_bool(validation.get("allow_negative"), False):
                clean = torch.clamp(clean, min=0.0)
            else:
                raise SecurityError(
                    SecurityErrorType.ADVERSARIAL_INPUT,
                    "Attention tensor contains negative weights.",
                    severity=SecuritySeverity.HIGH,
                    context={"shape": tuple(matrix.shape), "min_value": float(clean.min().item())},
                    component="attention_monitor",
                    response_action=SecurityResponseAction.BLOCK,
                )
        return clean

    def _matrix_to_2d(self, matrix: torch.Tensor) -> torch.Tensor:
        clean = self._sanitize_matrix(matrix)
        if clean.dim() == 2:
            return clean
        if clean.dim() == 3:
            return clean.mean(dim=0)
        if clean.dim() == 4:
            return clean.mean(dim=(0, 1))
        return clean.reshape(clean.shape[-2], clean.shape[-1])

    def _prepare_attention(self, attention_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, AttentionTensorSummary]:
        clean = self._sanitize_matrix(attention_matrix)
        original_shape = tuple(int(x) for x in clean.shape)
        reduced = False
        if clean.dim() == 2:
            canonical = clean.unsqueeze(0).unsqueeze(0)
        elif clean.dim() == 3:
            # Interpret [heads, query, key].
            canonical = clean.unsqueeze(0)
        elif clean.dim() == 4:
            canonical = clean
        else:
            reduced = True
            canonical = clean.reshape(1, 1, clean.shape[-2], clean.shape[-1])

        validation = self._cfg("tensor_validation", {})
        require_square = coerce_bool(validation.get("require_square"), True)
        if require_square and canonical.shape[-1] != canonical.shape[-2]:
            raise SecurityError(
                SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                "Attention tensor must be square on query/key dimensions.",
                severity=SecuritySeverity.HIGH,
                context={"canonical_shape": tuple(canonical.shape)},
                component="attention_monitor",
                response_action=SecurityResponseAction.BLOCK,
            )
        normalized = False
        if coerce_bool(validation.get("normalize_rows"), True):
            canonical = self._normalize_rows(canonical)
            normalized = True
        aggregate = canonical.mean(dim=(0, 1))
        summary = AttentionTensorSummary(
            original_shape=original_shape,
            canonical_shape=tuple(int(x) for x in canonical.shape), # type: ignore
            dtype=str(attention_matrix.dtype),
            device=str(self.device),
            fingerprint=fingerprint(canonical.detach().cpu().tolist()),
            normalized=normalized,
            reduced=reduced,
        )
        return canonical, aggregate, summary

    @staticmethod
    def _normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
        row_sums = matrix.sum(dim=-1, keepdim=True)
        return torch.where(row_sums > EPSILON, matrix / torch.clamp(row_sums, min=EPSILON), torch.zeros_like(matrix))

    def _build_metrics(self, canonical: torch.Tensor, aggregate: torch.Tensor) -> Dict[str, Any]:
        head_importance = self._calculate_head_importance(canonical)
        row_entropies = self._row_entropies(aggregate)
        entropy = self._calculate_entropy(aggregate)
        max_entropy = math.log2(max(aggregate.numel(), 2))
        normalized_entropy = clamp_score(entropy / max_entropy if max_entropy > 0 else 0.0)
        diagonal_focus = self._diagonal_focus(aggregate)
        off_diagonal_mass = self._off_diagonal_mass(aggregate)
        row_max_mean = float(aggregate.max(dim=1).values.mean().item())
        col_max_mean = float(aggregate.max(dim=0).values.mean().item())
        head_entropy_values = [self._calculate_entropy(canonical[:, idx, :, :].mean(dim=0)) for idx in range(canonical.shape[1])]
        head_dominance = max(head_importance) if head_importance else 0.0
        metrics: Dict[str, Any] = {
            "max_attention": float(aggregate.max().item()),
            "min_attention": float(aggregate.min().item()),
            "mean_attention": float(aggregate.mean().item()),
            "std_attention": float(aggregate.std(unbiased=False).item()),
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "row_entropy_mean": float(sum(row_entropies) / len(row_entropies)) if row_entropies else 0.0,
            "row_entropy_min": min(row_entropies) if row_entropies else 0.0,
            "row_entropy_max": max(row_entropies) if row_entropies else 0.0,
            "uniformity": self._calculate_uniformity(aggregate),
            "dispersion": self._calculate_dispersion(aggregate),
            "diagonal_focus": diagonal_focus,
            "off_diagonal_mass": off_diagonal_mass,
            "row_max_mean": row_max_mean,
            "column_max_mean": col_max_mean,
            "focus_pattern": self._identify_focus_pattern(aggregate),
            "head_count": int(canonical.shape[1]),
            "batch_count": int(canonical.shape[0]),
            "sequence_length": int(canonical.shape[-1]),
            "head_importance": [float(x) for x in head_importance],
            "head_dominance": head_dominance,
            "head_entropy": [float(x) for x in head_entropy_values],
            "anomaly_score": 0.0,
            "anomaly": False,
        }
        return metrics

    def _row_entropies(self, matrix: torch.Tensor) -> List[float]:
        entropies: List[float] = []
        for row in matrix:
            total = row.sum()
            if total <= EPSILON:
                entropies.append(0.0)
                continue
            probabilities = row / total
            probabilities = probabilities[probabilities > EPSILON]
            entropies.append(float((-(probabilities * torch.log2(probabilities)).sum()).item()))
        return entropies

    def _diagonal_focus(self, matrix: torch.Tensor) -> float:
        if matrix.shape[0] != matrix.shape[1] or matrix.numel() == 0:
            return 0.0
        total = matrix.sum()
        return clamp_score(float(torch.diag(matrix).sum().item() / total.item())) if total > EPSILON else 0.0

    def _off_diagonal_mass(self, matrix: torch.Tensor) -> float:
        if matrix.shape[0] != matrix.shape[1] or matrix.numel() == 0:
            return 0.0
        return clamp_score(1.0 - self._diagonal_focus(matrix))

    def _detect_anomalies_from_metrics(self, metrics: Mapping[str, Any]) -> float:
        entropy = coerce_float(metrics.get("entropy"), 0.0)
        normalized_entropy = clamp_score(metrics.get("normalized_entropy"), default=0.0)
        uniformity = coerce_float(metrics.get("uniformity"), 0.0)
        dispersion = clamp_score(metrics.get("dispersion"), default=0.0)
        max_attention = clamp_score(metrics.get("max_attention"), default=0.0)
        diagonal_focus = clamp_score(metrics.get("diagonal_focus"), default=0.0)
        head_dominance = clamp_score(metrics.get("head_dominance"), default=0.0)

        entropy_low_dev = clamp_score((self.entropy_threshold - entropy) / max(self.entropy_threshold, EPSILON)) if entropy < self.entropy_threshold else 0.0
        entropy_high_dev = clamp_score((normalized_entropy - coerce_float(self._cfg("security.diffuse_entropy_threshold"), 0.90)) / 0.10)
        uniformity_dev = clamp_score((uniformity - self.uniformity_threshold) / max(self.uniformity_threshold, EPSILON)) if uniformity > self.uniformity_threshold else 0.0
        sharp_focus_dev = clamp_score((max_attention - coerce_float(self._cfg("security.sharp_focus_threshold"), 0.95)) / 0.05)
        diagonal_dev = clamp_score((diagonal_focus - coerce_float(self._cfg("security.diagonal_dominance_threshold"), 0.80)) / 0.20)
        head_dev = clamp_score((head_dominance - coerce_float(self._cfg("security.head_dominance_threshold"), 0.75)) / 0.25)
        component_scores = {
            "entropy_low": entropy_low_dev,
            "entropy_high": entropy_high_dev,
            "uniformity": uniformity_dev,
            "dispersion": dispersion,
            "sharp_focus": sharp_focus_dev,
            "diagonal_dominance": diagonal_dev,
            "head_dominance": head_dev,
        }
        weights = self._cfg("anomaly_weights", {}) or {}
        return weighted_average(component_scores, weights, default=0.0)

    def _store_analysis(self, metrics: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> Optional[str]:
        """Store attention analysis in SecureMemory without raw sensitive payloads."""
        printer.status("MONITOR", "Storing analysis", "info")

        memory_cfg = self._cfg("memory", {}) or {}
        safe_metrics = dict(metrics)
        if not coerce_bool(memory_cfg.get("store_visualization", False), False):
            safe_metrics.pop("attention_plot", None)
        record = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "module_version": MODULE_VERSION,
            "metrics": sanitize_for_logging(safe_metrics),
            "context": sanitize_for_logging(dict(context or {})),
            "timestamp": utc_iso(),
        }
        try:
            return self.memory.add(
                record,
                tags=list(memory_cfg.get("analysis_tags", ["attention_analysis", "security"])),
                sensitivity=coerce_float(memory_cfg.get("analysis_sensitivity", 0.6), 0.6, minimum=0.0, maximum=1.0),
                ttl_seconds=memory_cfg.get("analysis_ttl_seconds"),
                purpose="attention_security_monitoring",
                owner="attention_monitor",
                classification=str(memory_cfg.get("classification", "confidential")),
                source="attention_monitor",
                metadata={"analysis_fingerprint": fingerprint(record)},
            )
        except TypeError:
            # Backward-compatible path for older SecureMemory.add signatures.
            return self.memory.add(
                record,
                tags=list(memory_cfg.get("analysis_tags", ["attention_analysis", "security"])),
                sensitivity=coerce_float(memory_cfg.get("analysis_sensitivity", 0.6), 0.6, minimum=0.0, maximum=1.0),
            )
        except SecurityError:
            raise
        except Exception as exc:
            raise AuditLogFailureError(
                logging_target="secure_memory.attention_analysis",
                failure_mode=f"Failed to store attention analysis: {type(exc).__name__}",
                component="attention_monitor",
                cause=exc,
            ) from exc

    def _track_recent_analysis(self, analysis_id: str) -> None:
        self._recent_analysis_ids.append(analysis_id)
        max_recent = coerce_int(self._cfg("recent_analysis_max", 250), 250, minimum=1)
        if len(self._recent_analysis_ids) > max_recent:
            del self._recent_analysis_ids[: len(self._recent_analysis_ids) - max_recent]

    def get_recent_analysis_ids(self) -> List[str]:
        return list(self._recent_analysis_ids)


class AttentionAdapter:
    """Adapter for attention layers that emit observer callbacks."""

    def __init__(self, monitor: AttentionMonitor):
        self.monitor = monitor

    def log_attention(self, attention_matrix: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        printer.status("ADAPT", "Log attention", "info")
        if attention_matrix.dim() == 4:
            # Keep the full 4D tensor for monitor-level head/batch analysis.
            prepared = attention_matrix
        elif attention_matrix.dim() == 3:
            prepared = attention_matrix
        elif attention_matrix.dim() == 2:
            prepared = attention_matrix
        else:
            raise SecurityError(
                SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                "Attention adapter received unsupported attention tensor rank.",
                severity=SecuritySeverity.HIGH,
                context={"shape": tuple(attention_matrix.shape)},
                component="attention_monitor.adapter",
                response_action=SecurityResponseAction.BLOCK,
            )
        return self.monitor.analyze_attention(prepared.to(self.monitor.device), context=context or {"source": "attention_adapter"})


if __name__ == "__main__":
    print("\n=== Running Attention Monitor ===\n")
    printer.status("TEST", "Attention Monitor initialized", "info")

    device = "cpu"
    monitor = AttentionMonitor(device)

    attention_matrix = torch.tensor(
        [
            [0.80, 0.10, 0.10],
            [0.20, 0.70, 0.10],
            [0.10, 0.10, 0.80],
        ],
        dtype=torch.float32,
    )
    analysis = monitor.analyze_attention(
        attention_matrix,
        context={
            "request_id": "attn-test-001",
            "user_email": "analyst@example.com",
            "auth_token": "secret-token-should-not-leak",
            "purpose": "self_test",
        },
    )
    assert analysis["schema_version"] == ANALYSIS_SCHEMA_VERSION
    assert analysis["entropy"] >= 0.0
    assert 0.0 <= analysis["anomaly_score"] <= 1.0
    assert "security_assessment" in analysis
    assert "secret-token-should-not-leak" not in stable_json(analysis)
    assert "analyst@example.com" not in stable_json(analysis)
    printer.status("TEST", f"2D analysis decision: {analysis['security_assessment']['decision']}", "info")

    multi_head = torch.stack(
        [
            attention_matrix,
            torch.tensor(
                [
                    [0.34, 0.33, 0.33],
                    [0.33, 0.34, 0.33],
                    [0.33, 0.33, 0.34],
                ],
                dtype=torch.float32,
            ),
        ],
        dim=0,
    )
    adapter = AttentionAdapter(monitor)
    adapter_result = adapter.log_attention(multi_head, context={"request_id": "attn-test-002"})
    assert adapter_result["head_count"] == 2
    assert len(adapter_result["head_importance"]) == 2
    printer.status("TEST", f"Adapter head importance: {adapter_result['head_importance']}", "info")

    report = monitor.generate_report(analysis)
    assert "Attention Analysis Report" in report
    assert "secret-token-should-not-leak" not in report
    printer.status("TEST", "Report generated safely", "info")

    bad_matrix = torch.tensor([[float("nan"), 1.0], [0.5, 0.5]], dtype=torch.float32)
    try:
        monitor.analyze_attention(bad_matrix, context={"request_id": "attn-test-bad"})
        raise AssertionError("Expected SecurityError for invalid attention tensor")
    except SecurityError as exc:
        public = exc.to_public_response() if hasattr(exc, "to_public_response") else {"message": str(exc)}
        assert public.get("message")
        printer.status("TEST", "Invalid tensor produced structured SecurityError", "info")

    print("\n=== Test ran successfully ===\n")
