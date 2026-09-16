"""Production attention telemetry and anomaly monitoring for Safety Agent."""
from __future__ import annotations

import base64
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from io import BytesIO
from threading import RLock
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..base.utils.interpretability import InterpretabilityHelper
from .utils.config_loader import get_config_section, load_global_config
from .utils.safety_helpers import *
from .utils.security_error import *
from .secure_memory import SecureMemory
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("Security Attention Monitor")
printer = PrettyPrinter()

MODULE_VERSION = "2.3.0"
ANALYSIS_SCHEMA_VERSION = "attention_monitor.analysis.v4"
ASSESSMENT_SCHEMA_VERSION = "attention_monitor.security_assessment.v3"
EPSILON = 1e-12


@dataclass(frozen=True)
class AttentionTensorSummary:
    original_shape: Tuple[int, ...]
    canonical_shape: Tuple[int, int, int, int]
    dtype: str
    device: str
    fingerprint: str
    normalized: bool
    reduced: bool
    layout: str
    attention_kind: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttentionSecurityAssessment:
    secure: bool
    confidence: float
    severity: str
    risk_score: float
    decision: str
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    evidence_class: str = "telemetry_anomaly"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["confidence"] = clamp_score(data["confidence"])
        data["risk_score"] = clamp_score(data["risk_score"])
        return sanitize_for_logging(data)


@dataclass(frozen=True)
class AttentionAnalysisResult:
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
        result = {
            "schema_version": self.schema_version,
            "module_version": MODULE_VERSION,
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp,
            "matrix_summary": self.matrix_summary.to_dict(),
            "metrics": sanitize_for_logging(self.metrics),
            "security_assessment": self.security_assessment.to_dict(),
            "context": sanitize_for_logging(self.context),
            "interpretation": redact_text(self.interpretation, max_length=2048),
        }
        if self.attention_plot is not None:
            result["attention_plot"] = self.attention_plot
        return result


@dataclass
class _RunningBaseline:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def score(self, value: float, minimum_samples: int) -> float:
        if self.count < minimum_samples or self.count < 2:
            return 0.0
        variance = self.m2 / max(self.count - 1, 1)
        std = math.sqrt(max(variance, 0.0))
        return abs(value - self.mean) / std if std > EPSILON else 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def snapshot(self) -> Dict[str, float]:
        variance = self.m2 / max(self.count - 1, 1) if self.count > 1 else 0.0
        return {"count": float(self.count), "mean": self.mean, "std_dev": math.sqrt(max(variance, 0.0))}


class AttentionMonitor(nn.Module):
    """Attention observability layer; anomalies are evidence, not attack attribution."""

    def __init__(
        self,
        device: Union[str, torch.device] = "cpu",
        *,
        memory: Optional[SecureMemory] = None,
    ) -> None:
        super().__init__()
        self.config = load_global_config()
        self.attention_config = get_config_section("attention_monitor")
        self._validate_configuration()
        self.device = torch.device(device)
        self.memory = memory or SecureMemory.shared()
        self.interpreter = InterpretabilityHelper()
        self.entropy_threshold = coerce_float(self._cfg("entropy_threshold", 0.0), 0.0, minimum=0.0)
        self.uniformity_threshold = coerce_float(self._cfg("uniformity_threshold", 0.0), 0.0, minimum=0.0)
        self.anomaly_threshold = coerce_float(self._cfg("anomaly_threshold", 0.75), 0.75, minimum=0.0, maximum=1.0)
        self.review_threshold = coerce_float(self._cfg("review_threshold", 0.45), 0.45, minimum=0.0, maximum=1.0)
        self.anomaly_detection = coerce_bool(self._cfg("anomaly_detection", True), True)
        self.store_analysis = coerce_bool(self._cfg("store_analysis", True), True)
        self.visualization = coerce_bool(self._cfg("visualization.enabled", False), False)
        self._state_lock = RLock()
        self._baselines: Dict[str, Dict[str, _RunningBaseline]] = defaultdict(dict)
        self._recent_analysis_ids: List[str] = []

    def _cfg(self, path: Union[str, Sequence[str]], default: Any = None) -> Any:
        return get_nested(self.attention_config or {}, path, default)

    def _validate_configuration(self) -> None:
        if not isinstance(self.attention_config, Mapping):
            raise ConfigurationTamperingError("attention_monitor", "attention_monitor config must be a mapping", component="attention_monitor")
        review = coerce_float(self.attention_config.get("review_threshold", 0.45), 0.45)
        block = coerce_float(self.attention_config.get("anomaly_threshold", 0.75), 0.75)
        if review > block:
            raise ConfigurationTamperingError("attention_monitor.review_threshold", "review threshold must not exceed anomaly threshold", component="attention_monitor")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_attention(self, attention_matrix: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context_map = dict(context or {})
        canonical, aggregate, summary = self._prepare_attention(attention_matrix, context_map)
        metrics = self._build_metrics(canonical, aggregate)
        baseline = self._baseline_scores(metrics, context_map, summary)
        metrics["baseline_scores"] = baseline
        metrics["anomaly_score"] = self._detect_anomalies_from_metrics(metrics, baseline) if self.anomaly_detection else 0.0
        metrics["anomaly"] = metrics["anomaly_score"] >= self.anomaly_threshold
        assessment = self._assess_security(metrics, baseline)
        interpretation = self.get_anomaly_interpretation({**metrics, "security_assessment": assessment.to_dict()})
        analysis_id = generate_identifier("attn")
        safe_context = sanitize_for_logging(context_map)
        plot = self.visualize_attention(aggregate) if self.visualization else None
        result = AttentionAnalysisResult(
            schema_version=ANALYSIS_SCHEMA_VERSION,
            analysis_id=analysis_id,
            timestamp=utc_iso(),
            matrix_summary=summary,
            metrics=metrics,
            security_assessment=assessment,
            context=safe_context,
            attention_plot=plot,
            interpretation=interpretation,
        ).to_dict()
        # Legacy top-level metrics.
        result.update(metrics)
        result["anomaly_interpretation"] = interpretation
        if self.store_analysis:
            self._store_analysis(result, safe_context)
        self._track_recent_analysis(analysis_id)
        self._update_baselines(metrics, context_map, summary, assessment)
        return result

    def generate_report(self, analysis: Mapping[str, Any]) -> str:
        safe = sanitize_for_logging(dict(analysis or {}))
        security = safe.get("security_assessment", {}) if isinstance(safe.get("security_assessment"), Mapping) else {}
        findings = security.get("findings", []) or []
        recommendations = security.get("recommendations", []) or []
        lines = [
            "# Attention Analysis Report",
            f"**Generated**: {utc_iso()}",
            f"**Analysis ID**: `{safe.get('analysis_id', 'unknown')}`",
            f"**Decision**: `{security.get('decision', 'unknown')}`",
            f"**Risk**: {coerce_float(safe.get('anomaly_score'), 0.0):.3f}",
            "",
            "## Metrics",
            f"- Entropy: {coerce_float(safe.get('entropy'), 0.0):.3f}",
            f"- Normalized entropy: {coerce_float(safe.get('normalized_entropy'), 0.0):.3f}",
            f"- Uniformity: {coerce_float(safe.get('uniformity'), 0.0):.3f}",
            f"- Dispersion: {coerce_float(safe.get('dispersion'), 0.0):.3f}",
            f"- Head dominance: {coerce_float(safe.get('head_dominance'), 0.0):.3f}",
            "",
            "## Findings",
        ]
        lines.extend([f"- {value}" for value in findings] or ["- No material attention-telemetry anomaly detected."])
        if recommendations:
            lines.append("\n## Recommendations")
            lines.extend(f"- {value}" for value in recommendations)
        return "\n".join(lines)

    def get_anomaly_interpretation(self, analysis: Mapping[str, Any]) -> str:
        findings: List[str] = []
        if coerce_float(analysis.get("normalized_entropy"), 0.0) > coerce_float(self._cfg("security.diffuse_entropy_threshold", 0.90), 0.90):
            findings.append("Attention is unusually diffuse relative to the configured reference threshold.")
        if coerce_float(analysis.get("max_attention"), 0.0) > coerce_float(self._cfg("security.sharp_focus_threshold", 0.95), 0.95):
            findings.append("Attention contains unusually sharp local concentration.")
        if coerce_float(analysis.get("head_dominance"), 0.0) > coerce_float(self._cfg("security.head_dominance_threshold", 0.75), 0.75):
            findings.append("A single head contributes an unusually large fraction of attention mass.")
        if coerce_float(analysis.get("anomaly_score"), 0.0) >= self.review_threshold:
            findings.append("The combined telemetry pattern warrants corroboration with model/output safety evidence.")
        return " ".join(findings) if findings else "Attention telemetry is within configured monitoring bounds."

    def visualize_attention(self, attention_matrix: torch.Tensor) -> str:
        matrix = self._matrix_to_2d(attention_matrix).detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 5))
        try:
            image = ax.imshow(matrix, aspect="auto")
            ax.set_title("Attention Map")
            ax.set_xlabel("Key position")
            ax.set_ylabel("Query position")
            fig.colorbar(image, ax=ax)
            buffer = BytesIO()
            fig.tight_layout()
            fig.savefig(buffer, format="png", dpi=coerce_int(self._cfg("visualization.dpi", 110), 110, minimum=50, maximum=300))
            return base64.b64encode(buffer.getvalue()).decode("ascii")
        finally:
            plt.close(fig)

    # ------------------------------------------------------------------
    # Tensor handling
    # ------------------------------------------------------------------
    def _sanitize_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        if not isinstance(matrix, torch.Tensor):
            raise SecurityError(SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT, "Attention monitor requires torch.Tensor input.", component="attention_monitor", response_action=SecurityResponseAction.BLOCK)
        validation = self._cfg("tensor_validation", {}) or {}
        max_dimensions = coerce_int(validation.get("max_dimensions", 4), 4, minimum=2)
        max_elements = coerce_int(validation.get("max_elements", 4_000_000), 4_000_000, minimum=1)
        if matrix.dim() < 2 or matrix.dim() > max_dimensions:
            raise SecurityError(SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT, "Unsupported attention tensor rank.", component="attention_monitor", context={"shape": tuple(matrix.shape)})
        if matrix.numel() > max_elements:
            raise ResourceExhaustionError("attention_tensor_elements", float(matrix.numel()), float(max_elements), source_identifier="attention_monitor", component="attention_monitor")
        clean = matrix.detach().to(self.device).float()
        if not torch.isfinite(clean).all():
            raise SecurityError(SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION, "Attention tensor contains non-finite values.", component="attention_monitor", severity=SecuritySeverity.HIGH)
        if torch.any(clean < 0.0):
            if coerce_bool(validation.get("allow_negative", False), False):
                clean = torch.clamp(clean, min=0.0)
            else:
                raise SecurityError(SecurityErrorType.ADVERSARIAL_INPUT, "Attention tensor contains negative weights.", component="attention_monitor", context={"shape": tuple(clean.shape)})
        return clean

    def _prepare_attention(self, attention_matrix: torch.Tensor, context: Mapping[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, AttentionTensorSummary]:
        clean = self._sanitize_matrix(attention_matrix)
        layout = normalize_identifier(context.get("attention_layout") or context.get("layout") or "", max_length=64)
        attention_kind = normalize_identifier(context.get("attention_kind") or "self_attention", max_length=64)
        inferred = False
        if clean.dim() == 2:
            canonical = clean.unsqueeze(0).unsqueeze(0)
            layout = layout or "qk"
        elif clean.dim() == 3:
            if layout in {"batch_qk", "bqk"}:
                canonical = clean.unsqueeze(1)
                layout = "batch_qk"
            else:
                canonical = clean.unsqueeze(0)
                layout = layout or "heads_qk"
                inferred = not bool(context.get("attention_layout") or context.get("layout"))
        elif clean.dim() == 4:
            canonical = clean
            layout = layout or "batch_heads_qk"
        else:
            raise SecurityError(SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT, "Unsupported attention tensor layout.", component="attention_monitor")

        validation = self._cfg("tensor_validation", {}) or {}
        require_square = coerce_bool(validation.get("require_square", True), True)
        if attention_kind in {"cross_attention", "encoder_decoder_attention"}:
            require_square = False
        if require_square and canonical.shape[-1] != canonical.shape[-2]:
            raise SecurityError(SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT, "Self-attention tensor must be square on query/key axes.", component="attention_monitor", context={"shape": tuple(canonical.shape)})
        normalized = False
        if coerce_bool(validation.get("normalize_rows", True), True):
            canonical = self._normalize_rows(canonical)
            normalized = True
        aggregate = canonical.mean(dim=(0, 1))
        summary = AttentionTensorSummary(
            original_shape=tuple(int(x) for x in clean.shape),
            canonical_shape=tuple(int(x) for x in canonical.shape),  # type: ignore[arg-type]
            dtype=str(attention_matrix.dtype),
            device=str(self.device),
            fingerprint=fingerprint(canonical.detach().cpu().tolist()),
            normalized=normalized,
            reduced=inferred,
            layout=layout,
            attention_kind=attention_kind,
        )
        return canonical, aggregate, summary

    @staticmethod
    def _normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
        sums = matrix.sum(dim=-1, keepdim=True)
        return torch.where(sums > EPSILON, matrix / torch.clamp(sums, min=EPSILON), torch.zeros_like(matrix))

    def _matrix_to_2d(self, matrix: torch.Tensor) -> torch.Tensor:
        clean = self._sanitize_matrix(matrix)
        if clean.dim() == 2: return clean
        if clean.dim() == 3: return clean.mean(dim=0)
        return clean.mean(dim=(0, 1))

    # ------------------------------------------------------------------
    # Metrics and baseline calibration
    # ------------------------------------------------------------------
    def _build_metrics(self, canonical: torch.Tensor, aggregate: torch.Tensor) -> Dict[str, Any]:
        head_importance = self._calculate_head_importance(canonical)
        row_entropies = self._row_entropies(aggregate)
        entropy = self._calculate_entropy(aggregate)
        # Entropy normalized per row/key vocabulary rather than matrix element count.
        max_row_entropy = math.log2(max(int(aggregate.shape[-1]), 2))
        normalized_entropy = clamp_score((sum(row_entropies) / len(row_entropies)) / max_row_entropy if row_entropies and max_row_entropy > 0 else 0.0)
        metrics = {
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
            "diagonal_focus": self._diagonal_focus(aggregate),
            "off_diagonal_mass": self._off_diagonal_mass(aggregate),
            "row_max_mean": float(aggregate.max(dim=1).values.mean().item()),
            "column_max_mean": float(aggregate.max(dim=0).values.mean().item()),
            "focus_pattern": self._identify_focus_pattern(aggregate),
            "head_count": int(canonical.shape[1]),
            "batch_count": int(canonical.shape[0]),
            "query_length": int(canonical.shape[-2]),
            "sequence_length": int(canonical.shape[-1]),
            "head_importance": head_importance,
            "head_dominance": max(head_importance) if head_importance else 0.0,
            "head_entropy": [self._calculate_entropy(canonical[:, i].mean(dim=0)) for i in range(canonical.shape[1])],
        }
        return metrics

    def _baseline_key(self, context: Mapping[str, Any], summary: AttentionTensorSummary) -> str:
        parts = {
            "model": context.get("model_id") or context.get("model_name") or "unknown_model",
            "layer": context.get("layer_id") or context.get("layer") or "unknown_layer",
            "task": context.get("task_type") or context.get("purpose") or "general",
            "layout": summary.layout,
            "kind": summary.attention_kind,
            "heads": summary.canonical_shape[1],
            "q_band": int(summary.canonical_shape[2] // 32),
            "k_band": int(summary.canonical_shape[3] // 32),
        }
        return fingerprint(parts, length=24)

    def _baseline_scores(self, metrics: Mapping[str, Any], context: Mapping[str, Any], summary: AttentionTensorSummary) -> Dict[str, float]:
        key = self._baseline_key(context, summary)
        minimum_samples = coerce_int(self._cfg("baseline.min_samples", 30), 30, minimum=2)
        monitored = list(self._cfg("baseline.metrics", ["normalized_entropy", "uniformity", "max_attention", "head_dominance"]))
        with self._state_lock:
            bucket = self._baselines[key]
            scores: Dict[str, float] = {}
            for name in monitored:
                stat = bucket.get(name)
                if stat is not None:
                    scores[name] = stat.score(coerce_float(metrics.get(name), 0.0), minimum_samples)
            scores["sample_count"] = float(max((value.count for value in bucket.values()), default=0))
            return scores

    def _update_baselines(self, metrics: Mapping[str, Any], context: Mapping[str, Any], summary: AttentionTensorSummary, assessment: AttentionSecurityAssessment) -> None:
        if not coerce_bool(self._cfg("baseline.enabled", True), True):
            return
        if assessment.decision == "block" and not coerce_bool(self._cfg("baseline.learn_from_blocked", False), False):
            return
        key = self._baseline_key(context, summary)
        monitored = list(self._cfg("baseline.metrics", ["normalized_entropy", "uniformity", "max_attention", "head_dominance"]))
        with self._state_lock:
            bucket = self._baselines[key]
            for name in monitored:
                bucket.setdefault(name, _RunningBaseline()).update(coerce_float(metrics.get(name), 0.0))

    def _detect_anomalies_from_metrics(self, metrics: Mapping[str, Any], baseline: Mapping[str, float]) -> float:
        entropy = coerce_float(metrics.get("entropy"), 0.0)
        normalized_entropy = clamp_score(metrics.get("normalized_entropy", 0.0))
        uniformity = coerce_float(metrics.get("uniformity"), 0.0)
        dispersion = clamp_score(metrics.get("dispersion", 0.0))
        max_attention = clamp_score(metrics.get("max_attention", 0.0))
        diagonal_focus = clamp_score(metrics.get("diagonal_focus", 0.0))
        head_dominance = clamp_score(metrics.get("head_dominance", 0.0))
        component_scores = {
            "entropy_low": clamp_score((self.entropy_threshold - entropy) / max(self.entropy_threshold, EPSILON)) if self.entropy_threshold and entropy < self.entropy_threshold else 0.0,
            "entropy_high": clamp_score((normalized_entropy - coerce_float(self._cfg("security.diffuse_entropy_threshold", 0.90), 0.90)) / 0.10),
            "uniformity": clamp_score((uniformity - self.uniformity_threshold) / max(self.uniformity_threshold, EPSILON)) if self.uniformity_threshold and uniformity > self.uniformity_threshold else 0.0,
            "dispersion": dispersion,
            "sharp_focus": clamp_score((max_attention - coerce_float(self._cfg("security.sharp_focus_threshold", 0.95), 0.95)) / 0.05),
            "diagonal_dominance": clamp_score((diagonal_focus - coerce_float(self._cfg("security.diagonal_dominance_threshold", 0.80), 0.80)) / 0.20),
            "head_dominance": clamp_score((head_dominance - coerce_float(self._cfg("security.head_dominance_threshold", 0.75), 0.75)) / 0.25),
        }
        z_threshold = coerce_float(self._cfg("baseline.z_threshold", 3.0), 3.0, minimum=0.1)
        baseline_score = max([clamp_score(coerce_float(v, 0.0) / (z_threshold * 2.0)) for k, v in baseline.items() if k != "sample_count"] or [0.0])
        static_score = weighted_average(component_scores, self._cfg("anomaly_weights", {}) or {}, default=max(component_scores.values() or [0.0]))
        baseline_weight = coerce_float(self._cfg("baseline.risk_weight", 0.45), 0.45, minimum=0.0, maximum=1.0)
        return clamp_score((1.0 - baseline_weight) * static_score + baseline_weight * baseline_score)

    def _assess_security(self, metrics: Mapping[str, Any], baseline: Mapping[str, float]) -> AttentionSecurityAssessment:
        risk = clamp_score(metrics.get("anomaly_score", 0.0))
        findings: List[str] = []
        recommendations: List[str] = []
        if risk >= self.review_threshold:
            findings.append("Attention telemetry deviates from configured or learned reference behavior.")
            recommendations.append("Corroborate this signal with output, prompt-security, and model-integrity evidence before taking action.")
        if coerce_float(metrics.get("head_dominance"), 0.0) >= coerce_float(self._cfg("security.head_dominance_threshold", 0.75), 0.75):
            findings.append("Single-head dominance detected.")
        if coerce_float(baseline.get("sample_count"), 0.0) < coerce_float(self._cfg("baseline.min_samples", 30), 30):
            recommendations.append("Baseline is not yet mature; interpret anomaly scores conservatively.")
        decision = threshold_decision(risk, block_threshold=self.anomaly_threshold, review_threshold=self.review_threshold)
        confidence = coerce_float(self._cfg("security.base_confidence", 0.82), 0.82, minimum=0.0, maximum=1.0)
        if coerce_float(baseline.get("sample_count"), 0.0) < coerce_float(self._cfg("baseline.min_samples", 30), 30):
            confidence *= 0.75
        return AttentionSecurityAssessment(
            secure=decision == "allow" and not findings,
            confidence=clamp_score(confidence),
            severity=categorize_risk(risk),
            risk_score=risk,
            decision=decision,
            findings=dedupe_preserve_order(findings),
            recommendations=dedupe_preserve_order(recommendations or ["Continue routine attention telemetry monitoring."]),
        )

    # ------------------------------------------------------------------
    # Metric primitives
    # ------------------------------------------------------------------
    def _calculate_entropy(self, matrix: torch.Tensor) -> float:
        values = matrix.flatten()
        total = values.sum()
        if total <= EPSILON: return 0.0
        p = values / total
        p = p[p > EPSILON]
        return float((-(p * torch.log2(p))).sum().item())

    def _row_entropies(self, matrix: torch.Tensor) -> List[float]:
        output: List[float] = []
        for row in matrix:
            total = row.sum()
            if total <= EPSILON:
                output.append(0.0); continue
            p = row / total; p = p[p > EPSILON]
            output.append(float((-(p * torch.log2(p))).sum().item()))
        return output

    def _calculate_uniformity(self, matrix: torch.Tensor) -> float:
        mean = float(matrix.mean().item())
        if abs(mean) <= EPSILON: return 0.0
        return float(matrix.std(unbiased=False).item()) / abs(mean)

    def _calculate_dispersion(self, matrix: torch.Tensor) -> float:
        if matrix.numel() == 0: return 0.0
        return clamp_score(float((matrix.max() - matrix.min()).item()))

    def _calculate_head_importance(self, canonical: torch.Tensor) -> List[float]:
        if canonical.shape[1] <= 0: return []
        masses = canonical.sum(dim=(0, 2, 3)).detach().cpu()
        total = float(masses.sum().item())
        if total <= EPSILON:
            return [1.0 / canonical.shape[1]] * canonical.shape[1]
        return [float(v.item() / total) for v in masses]

    def _identify_focus_pattern(self, matrix: torch.Tensor) -> str:
        maximum = float(matrix.max().item()) if matrix.numel() else 0.0
        entropy = self._calculate_entropy(matrix)
        if maximum >= coerce_float(self._cfg("focus_patterns.sharp_threshold", 0.75), 0.75): return "sharp"
        if entropy >= coerce_float(self._cfg("focus_patterns.diffuse_entropy", 4.0), 4.0): return "diffuse"
        return "distributed"

    def _diagonal_focus(self, matrix: torch.Tensor) -> float:
        if matrix.shape[0] != matrix.shape[1] or matrix.numel() == 0: return 0.0
        total = matrix.sum()
        return clamp_score(float(torch.diag(matrix).sum().item() / total.item())) if total > EPSILON else 0.0

    def _off_diagonal_mass(self, matrix: torch.Tensor) -> float:
        if matrix.shape[0] != matrix.shape[1] or matrix.numel() == 0: return 0.0
        return clamp_score(1.0 - self._diagonal_focus(matrix))

    # ------------------------------------------------------------------
    # Persistence and compatibility
    # ------------------------------------------------------------------
    def _store_analysis(self, metrics: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> Optional[str]:
        cfg = self._cfg("memory", {}) or {}
        safe = dict(metrics)
        if not coerce_bool(cfg.get("store_visualization", False), False):
            safe.pop("attention_plot", None)
        return self.memory.add(
            {"schema_version": ANALYSIS_SCHEMA_VERSION, "module_version": MODULE_VERSION, "metrics": sanitize_for_logging(safe), "context": sanitize_for_logging(dict(context or {})), "timestamp": utc_iso()},
            tags=list(cfg.get("analysis_tags", ["attention_analysis", "security"])),
            sensitivity=coerce_float(cfg.get("analysis_sensitivity", 0.6), 0.6, minimum=0.0, maximum=1.0),
            ttl_seconds=cfg.get("analysis_ttl_seconds"),
            purpose="attention_security_monitoring", owner="attention_monitor", classification=str(cfg.get("classification", "confidential")), source="attention_monitor",
            metadata={"analysis_fingerprint": fingerprint(safe), "eligible_for_compliance": True},
        )

    def _track_recent_analysis(self, analysis_id: str) -> None:
        with self._state_lock:
            self._recent_analysis_ids.append(analysis_id)
            max_recent = coerce_int(self._cfg("recent_analysis_max", 250), 250, minimum=1)
            if len(self._recent_analysis_ids) > max_recent:
                del self._recent_analysis_ids[:-max_recent]

    def get_recent_analysis_ids(self) -> List[str]:
        with self._state_lock:
            return list(self._recent_analysis_ids)


class AttentionAdapter:
    def __init__(self, monitor: AttentionMonitor):
        self.monitor = monitor

    def log_attention(self, attention_matrix: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.monitor.analyze_attention(attention_matrix, context=context or {"source": "attention_adapter"})


__all__ = [
    "MODULE_VERSION", "ANALYSIS_SCHEMA_VERSION", "ASSESSMENT_SCHEMA_VERSION",
    "AttentionTensorSummary", "AttentionSecurityAssessment", "AttentionAnalysisResult",
    "AttentionMonitor", "AttentionAdapter",
]


if __name__ == "__main__":
    print("\n=== Running Attention Monitor ===\n")
    printer.status("TEST", "Attention Monitor initialized", "info")
    printer.section_header("Smoke test #1: Initialization")
    device = "cpu"
    memory = SecureMemory()
    monitor = AttentionMonitor(device=device, memory=memory)
    Adapter = AttentionAdapter(monitor=monitor)

    printer.status("START", "Attention Monitor ready", "success" if Adapter is not None else "error")

    printer.section_header("Smoke test #2: Report")
    analysis = {}
    report = monitor.generate_report(analysis=analysis)

    printer.status("REPORT", report, "success" if any(f"**Decision**: `{decision}`" in report for decision in {"allow", "review"}) else "error")

    printer.section_header("Smoke test #3: Tensor handling")
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
    safe_matrix = monitor._sanitize_matrix(matrix=attention_matrix)
    printer.status("MATRIX", safe_matrix, "success" if any(f"**Decision**: `{decision}`" in safe_matrix for decision in {"allow", "review"}) else "error")

    print("\n== Task run successfully ==\n")