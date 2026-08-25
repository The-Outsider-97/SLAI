"""
Production-ready reporting and visualization pipeline for evaluation output.

This module upgrades the original report layer into a structured reporting
component that:
- Maintains rolling performance metrics with validation and history support
- Produces consistent, serializable reports with integrity hashes
- Renders PyQt5-based charts in headless or interactive environments
- Integrates with certification and documentation subsystems
- Uses structured evaluation errors instead of silent failure paths

Design goals
------------
- Preserve the existing configuration access pattern
- Avoid wrapping local imports in try/except blocks
- Remain compatible with both legacy and upgraded certification/documentation
  implementations where possible
- Provide detailed, production-oriented behavior rather than a minimal wrapper
"""

from __future__ import annotations

import inspect
import json
import os
import hashlib
import importlib
import yaml # type: ignore

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_PYQT_AVAILABLE = False
QApplication = QPainter = QPixmap = QColor = QFont = QPen = Qt = QRect = QPointF = QSize = QBuffer = None


def _initialize_pyqt_bindings() -> bool:
    global _PYQT_AVAILABLE, QApplication, QPainter, QPixmap, QColor, QFont, QPen, Qt, QRect, QPointF, QSize, QBuffer
    try:
        qt_widgets = importlib.import_module("PyQt5.QtWidgets")
        qt_gui = importlib.import_module("PyQt5.QtGui")
        qt_core = importlib.import_module("PyQt5.QtCore")
    except Exception:
        _PYQT_AVAILABLE = False
        return False
    QApplication = qt_widgets.QApplication
    QPainter = qt_gui.QPainter
    QPixmap = qt_gui.QPixmap
    QColor = qt_gui.QColor
    QFont = qt_gui.QFont
    QPen = qt_gui.QPen
    Qt = qt_core.Qt
    QRect = qt_core.QRect
    QPointF = qt_core.QPointF
    QSize = qt_core.QSize
    QBuffer = qt_core.QBuffer
    _PYQT_AVAILABLE = True
    return True


_initialize_pyqt_bindings()

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.evaluation_errors import (ConfigLoadError, EvaluationError, OperationalError,
                                MetricCalculationError, ReportGenerationError,
                                VisualizationError)
from .certification_framework import CertificationManager
from .documentation import AuditTrail, DocumentVersioner, AuditBlock
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Performance Visualizer")
printer = PrettyPrinter()

MODULE_VERSION = "2.2.0"
_global_visualizer: Optional["PerformanceVisualizer"] = None


@dataclass(slots=True)
class VisualizationAsset:
    """Serializable chart payload returned inside generated reports."""

    chart_type: str
    encoding: str
    width: int
    height: int
    image: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chart_type": self.chart_type,
            "encoding": self.encoding,
            "width": self.width,
            "height": self.height,
            "image": self.image,
        }


@dataclass(slots=True)
class ReportArtifactSummary:
    """Summary of audit/versioning side effects performed during generation."""

    audit_block_hash: Optional[str] = None
    audit_block_index: Optional[int] = None
    version_id: Optional[str] = None
    version_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_block_hash": self.audit_block_hash,
            "audit_block_index": self.audit_block_index,
            "version_id": self.version_id,
            "version_hash": self.version_hash,
        }


class PerformanceVisualizer:
    """
    Production-grade visualization and report assembly component.

    Responsibilities
    ----------------
    - Normalize and maintain rolling evaluation metrics
    - Render chart assets from historical performance data
    - Build structured reports with integrity and traceability metadata
    - Coordinate with certification and documentation layers
    - Surface failures through structured evaluation errors
    """

    def __init__(self) -> None:
        self.config = load_global_config()
        self.visualizer_config = get_config_section("performance_visualizer")
        if not isinstance(self.visualizer_config, Mapping):
            raise ConfigLoadError(
                config_path=str(self.config.get("__config_path__", "<unknown>")),
                section="performance_visualizer",
                error_details="Section must be a mapping.",
            )

        self.max_points = self._load_positive_int(
            self.visualizer_config.get("max_points", 100),
            field_name="performance_visualizer.max_points",
        )

        initial_metrics = self.visualizer_config.get("initial_metrics", {})
        if not isinstance(initial_metrics, Mapping):
            raise ConfigLoadError(
                config_path=str(self.config.get("__config_path__", "<unknown>")),
                section="performance_visualizer.initial_metrics",
                error_details="Initial metrics must be a mapping.",
            )

        self.metrics: Dict[str, Any] = {
            "rewards": deque(self._coerce_numeric_sequence(initial_metrics.get("rewards", [])), maxlen=self.max_points),
            "risks": deque(self._coerce_numeric_sequence(initial_metrics.get("risks", [])), maxlen=self.max_points),
            "successes": self._coerce_non_negative_int(initial_metrics.get("successes", 0), "successes"),
            "failures": self._coerce_non_negative_int(initial_metrics.get("failures", 0), "failures"),
            "hazard_rates": deque(
                self._coerce_numeric_sequence(initial_metrics.get("hazard_rates", [])),
                maxlen=self.max_points,
            ),
            "operational_times": deque(
                self._coerce_numeric_sequence(initial_metrics.get("operational_times", [])),
                maxlen=self.max_points,
            ),
            "pass_rates": deque(
                self._coerce_numeric_sequence(initial_metrics.get("pass_rates", [])),
                maxlen=self.max_points,
            ),
            "update_count": 0,
        }

        self.colors = self._load_colors(self.visualizer_config.get("colors", {}))
        self.line_thickness = self._load_positive_int(
            self.visualizer_config.get("line_styles", {}).get("thickness", 2),
            field_name="performance_visualizer.line_styles.thickness",
        )
        grid_style = str(self.visualizer_config.get("line_styles", {}).get("grid_style", "dot")).strip().lower()
        self.grid_style = Qt.DotLine if grid_style == "dot" else Qt.SolidLine

        self.chart_dimensions = self._load_chart_dimensions(
            self.visualizer_config.get("chart_dimensions", {})
        )
        self.grid_config = self._load_grid_config(self.visualizer_config.get("grid", {}))
        self.font_config = self._load_font_config(self.visualizer_config.get("font", {}))

        self.cert_manager = CertificationManager()
        self.audit_trail = AuditTrail()
        self.doc_versioner = DocumentVersioner()

        logger.info("Performance Visualizer successfully initialized")

    # ------------------------------------------------------------------
    # Metric lifecycle management
    # ------------------------------------------------------------------

    def reset_metrics(self) -> None:
        """Reset rolling metric state while preserving configuration."""
        self.metrics["rewards"].clear()
        self.metrics["risks"].clear()
        self.metrics["hazard_rates"].clear()
        self.metrics["operational_times"].clear()
        self.metrics["pass_rates"].clear()
        self.metrics["successes"] = 0
        self.metrics["failures"] = 0
        self.metrics["update_count"] = 0
        logger.info("Performance metrics reset")

    def update_metrics(self, evaluation_data: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Update rolling metrics from evaluation output.

        Supported fields
        ----------------
        successes: int
        failures: int
        hazards: {system_failure: float, ...}
        operational_time: float
        reward: float
        risk: float
        pass_rate: float
        """
        if not isinstance(evaluation_data, Mapping):
            raise MetricCalculationError(
                metric_name="evaluation_update",
                inputs=type(evaluation_data).__name__,
                reason="Expected a mapping of evaluation data.",
            )

        successes = self._coerce_non_negative_int(evaluation_data.get("successes", 0), "successes")
        failures = self._coerce_non_negative_int(evaluation_data.get("failures", 0), "failures")
        self.metrics["successes"] += successes
        self.metrics["failures"] += failures

        if "hazards" in evaluation_data:
            hazard_rate = self._extract_hazard_rate(evaluation_data["hazards"])
            self.metrics["hazard_rates"].append(hazard_rate)

        if "operational_time" in evaluation_data:
            self.metrics["operational_times"].append(
                self._coerce_non_negative_float(evaluation_data["operational_time"], "operational_time")
            )

        if "reward" in evaluation_data:
            self.metrics["rewards"].append(
                self._coerce_float(evaluation_data["reward"], "reward")
            )

        if "risk" in evaluation_data:
            self.metrics["risks"].append(
                self._coerce_non_negative_float(evaluation_data["risk"], "risk")
            )

        if "pass_rate" in evaluation_data:
            self.metrics["pass_rates"].append(
                self._coerce_probability(evaluation_data["pass_rate"], "pass_rate")
            )
        else:
            cumulative_rate = self.compute_metric("success_rate")
            self.metrics["pass_rates"].append(cumulative_rate)

        self.metrics["update_count"] += 1
        snapshot = self.get_current_metrics()
        logger.info(
            "Metrics updated: successes=%d failures=%d updates=%d",
            self.metrics["successes"],
            self.metrics["failures"],
            self.metrics["update_count"],
        )
        return snapshot

    def add_metrics(self, metric_type: str, values: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Generic metric update helper retained for API compatibility.

        The original implementation accepted `metric_type` but did not use it.
        This version retains the parameter for compatibility while validating
        and applying the supplied values map.
        """
        if not isinstance(metric_type, str) or not metric_type.strip():
            raise MetricCalculationError(
                metric_name="metric_type",
                inputs=metric_type,
                reason="metric_type must be a non-empty string.",
            )
        if not isinstance(values, Mapping):
            raise MetricCalculationError(
                metric_name=metric_type,
                inputs=type(values).__name__,
                reason="values must be a mapping.",
            )

        normalized_payload: Dict[str, Any] = {}
        for key, value in values.items():
            normalized_payload[str(key)] = value

        if normalized_payload:
            self.update_metrics(normalized_payload)

        unknown_keys = [key for key in normalized_payload if key not in {
            "successes", "failures", "hazards", "operational_time", "reward", "risk", "pass_rate"
        }]
        for key in unknown_keys:
            logger.warning("Ignoring unknown metric key during '%s' update: %s", metric_type, key)

        return self.get_current_metrics()

    def compute_metric(self, metric_name: str = "success_rate") -> float:
        """Compute a normalized or aggregate metric from current state."""
        try:
            normalized_name = str(metric_name).strip().lower()
            if normalized_name == "success_rate":
                total = self.metrics["successes"] + self.metrics["failures"]
                return round(self.metrics["successes"] / total, 6) if total > 0 else 0.0
            if normalized_name == "current_risk":
                return float(self.metrics["hazard_rates"][-1]) if self.metrics["hazard_rates"] else 0.0
            if normalized_name == "average_risk":
                return round(_safe_mean(self.metrics["hazard_rates"]), 6)
            if normalized_name == "operational_time":
                return float(self.metrics["operational_times"][-1]) if self.metrics["operational_times"] else 0.0
            if normalized_name == "average_operational_time":
                return round(_safe_mean(self.metrics["operational_times"]), 6)
            if normalized_name == "pass_rate":
                return round(_safe_mean(self.metrics["pass_rates"]), 6)

            raise MetricCalculationError(
                metric_name=metric_name,
                inputs=list(self.metrics.keys()),
                reason="Unsupported metric requested.",
            )
        except EvaluationError:
            raise
        except Exception as exc:
            raise MetricCalculationError(
                metric_name=metric_name,
                inputs=self.get_metric_history(),
                reason=str(exc),
            ) from exc

    def get_current_metrics(self) -> Dict[str, Any]:
        """Return a structured summary of current performance metrics."""
        total_events = self.metrics["successes"] + self.metrics["failures"]
        return {
            "updates": self.metrics["update_count"],
            "successes": self.metrics["successes"],
            "failures": self.metrics["failures"],
            "total_events": total_events,
            "success_rate": self.compute_metric("success_rate"),
            "current_risk": self.compute_metric("current_risk"),
            "average_risk": self.compute_metric("average_risk"),
            "operational_time": self.compute_metric("operational_time"),
            "average_operational_time": self.compute_metric("average_operational_time"),
            "average_pass_rate": self.compute_metric("pass_rate"),
            "current_reward": float(self.metrics["rewards"][-1]) if self.metrics["rewards"] else 0.0,
            "current_risk_score": float(self.metrics["risks"][-1]) if self.metrics["risks"] else 0.0,
        }

    def get_metric_history(self) -> Dict[str, Any]:
        """Return the retained metric history in serializable form."""
        return {
            "rewards": list(self.metrics["rewards"]),
            "risks": list(self.metrics["risks"]),
            "hazard_rates": list(self.metrics["hazard_rates"]),
            "operational_times": list(self.metrics["operational_times"]),
            "pass_rates": list(self.metrics["pass_rates"]),
        }

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_full_report(
        self,
        include_visualizations: bool = True,
        include_audit_chain: bool = False,
        include_version_history: bool = False,
        audit_report: bool = True,
        version_report: bool = True,
        submit_report_evidence: bool = False,
        strict: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report.

        strict=False preserves partial output and records structured errors in the
        report body. strict=True raises the first section failure as an exception.
        """
        errors: List[Dict[str, Any]] = []
        generated_at = _utcnow().isoformat()

        report: Dict[str, Any] = {
            "metadata": {
                "timestamp": generated_at,
                "report_version": MODULE_VERSION,
                "generator": self.__class__.__name__,
                "config_path": self.config.get("__config_path__"),
                "strict_mode": strict,
            },
            "summary": self._generate_summary_section(),
            "performance_metrics": self.get_current_metrics(),
            "metric_history": self.get_metric_history(),
            "certification": self._run_section(
                section_name="certification",
                builder=self._generate_certification_section,
                errors=errors,
                strict=strict,
            ),
        }

        if include_visualizations:
            report["visualizations"] = self._run_section(
                section_name="visualizations",
                builder=self._generate_visualization_section,
                errors=errors,
                strict=strict,
            )

        report["integrity"] = self._generate_integrity_section(report)

        if submit_report_evidence:
            self._run_section(
                section_name="evidence_submission",
                builder=lambda: self._submit_report_evidence(report),
                errors=errors,
                strict=strict,
            )

        artifact_summary = self._run_section(
            section_name="artifact_persistence",
            builder=lambda: self._persist_report_artifacts(
                report,
                audit_report=audit_report,
                version_report=version_report,
            ),
            errors=errors,
            strict=strict,
        )

        report["documentation"] = self._run_section(
            section_name="documentation",
            builder=lambda: self._generate_documentation_section(
                include_audit_chain=include_audit_chain,
                include_version_history=include_version_history,
                artifact_summary=artifact_summary,
            ),
            errors=errors,
            strict=strict,
        )

        report["errors"] = errors
        report["summary"]["error_count"] = len(errors)
        report["summary"]["report_status"] = "COMPLETE" if not errors else "PARTIAL"

        logger.info(
            "Full report generated: status=%s errors=%d",
            report["summary"]["report_status"],
            len(errors),
        )
        return report

    def export_report(
        self,
        report: Mapping[str, Any],
        format: str = "json",
        destination_path: Optional[str] = None,
    ) -> str:
        """Serialize a generated report to JSON or YAML, optionally writing it to disk."""
        if not isinstance(report, Mapping):
            raise ReportGenerationError(
                report_type="system_report",
                template="serializer",
                error_details="Report payload must be a mapping.",
            )

        normalized_format = str(format).strip().lower()
        try:
            if normalized_format == "json":
                serialized = json.dumps(dict(report), indent=2, sort_keys=False, default=str)
            elif normalized_format == "yaml":
                serialized = yaml.safe_dump(dict(report), default_flow_style=False, sort_keys=False)
            else:
                raise ReportGenerationError(
                    report_type="system_report",
                    template=normalized_format,
                    error_details="Unsupported export format. Use 'json' or 'yaml'.",
                )

            if destination_path:
                with open(destination_path, "w", encoding="utf-8") as handle:
                    handle.write(serialized)

            return serialized
        except EvaluationError:
            raise
        except OSError as exc:
            raise ReportGenerationError(
                report_type="system_report",
                template=normalized_format,
                error_details=f"Failed to write report to disk: {exc}",
            ) from exc
        except Exception as exc:
            raise ReportGenerationError(
                report_type="system_report",
                template=normalized_format,
                error_details=str(exc),
            ) from exc

    # ------------------------------------------------------------------
    # Certification and documentation integration
    # ------------------------------------------------------------------

    def _generate_summary_section(self) -> Dict[str, Any]:
        metrics = self.get_current_metrics()
        total_events = metrics["total_events"]
        success_rate = metrics["success_rate"]
        risk = metrics["current_risk"]

        operational_state = "stable"
        if total_events == 0:
            operational_state = "uninitialized"
        elif risk > 0.3:
            operational_state = "elevated-risk"
        elif success_rate < 0.8:
            operational_state = "degraded"

        return {
            "operational_state": operational_state,
            "evaluation_window": self.metrics["update_count"],
            "success_rate": success_rate,
            "current_risk": risk,
            "total_events": total_events,
            "report_status": "PENDING",
            "error_count": 0,
        }

    def _generate_certification_section(self) -> Dict[str, Any]:
        section: Dict[str, Any] = {
            "current_level": getattr(self.cert_manager.current_level, "name", str(self.cert_manager.current_level)),
        }

        if hasattr(self.cert_manager, "evaluate_certification_detailed"):
            detailed = self.cert_manager.evaluate_certification_detailed()
            section["evaluation"] = detailed.to_dict() if hasattr(detailed, "to_dict") else dict(detailed)
            section["requirements_passed"] = bool(getattr(detailed, "passed", False))
            section["unmet_requirements"] = list(getattr(detailed, "unmet_requirements", []))
        else:
            passed, unmet = self.cert_manager.evaluate_certification()
            section["evaluation"] = {
                "passed": passed,
                "unmet_requirements": unmet,
            }
            section["requirements_passed"] = passed
            section["unmet_requirements"] = unmet

        section["certificate"] = self.cert_manager.generate_certificate()
        if hasattr(self.cert_manager, "get_evidence_inventory"):
            section["evidence_inventory"] = self.cert_manager.get_evidence_inventory()

        return section

    def _generate_documentation_section(
        self,
        include_audit_chain: bool,
        include_version_history: bool,
        artifact_summary: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        latest_version = self.doc_versioner.get_latest()
        latest_version_summary = self._summarize_version_record(latest_version)

        latest_block = self.audit_trail.chain[-1] if getattr(self.audit_trail, "chain", []) else None
        block_hash = self._extract_block_hash(latest_block)
        block_index = getattr(latest_block, "index", len(getattr(self.audit_trail, "chain", [])) - 1) if latest_block else None

        section: Dict[str, Any] = {
            "audit": {
                "chain_length": len(getattr(self.audit_trail, "chain", [])),
                "latest_block_hash": block_hash,
                "latest_block_index": block_index,
                "verified": self._verify_chain_if_supported(),
            },
            "versioning": {
                "stored_versions": len(getattr(self.doc_versioner, "versions", [])),
                "latest_version": latest_version_summary,
            },
            "artifacts": artifact_summary or {},
        }

        if include_audit_chain:
            exported_chain = self.audit_trail.export_chain(
                format=getattr(self.audit_trail, "default_format", "json")
            )
            section["audit"]["chain_export"] = json.loads(exported_chain) if exported_chain.strip().startswith("[") else exported_chain

        if include_version_history and hasattr(self.doc_versioner, "get_version_history"):
            section["versioning"]["history"] = [
                self._summarize_version_record(item)
                for item in self.doc_versioner.get_version_history()
            ]

        return section

    def _persist_report_artifacts(
        self,
        report: Mapping[str, Any],
        audit_report: bool,
        version_report: bool,
    ) -> Dict[str, Any]:
        summary = ReportArtifactSummary()

        if audit_report:
            block = self._audit_report(report)
            summary.audit_block_hash = self._extract_block_hash(block)
            summary.audit_block_index = getattr(block, "index", None)

        if version_report:
            version = self._store_report_version(report)
            summarized = self._summarize_version_record(version)
            summary.version_id = summarized.get("version_id")
            summary.version_hash = summarized.get("hash")

        return summary.to_dict()

    def _audit_report(self, report: Mapping[str, Any]) -> Any:
        audit_payload = {
            "report_hash": hashlib.sha256(_canonical_json(dict(report)).encode("utf-8")).hexdigest(),
            "metrics_snapshot": dict(report.get("performance_metrics", {})),
            "summary": dict(report.get("summary", {})),
            "generated_at": report.get("metadata", {}).get("timestamp"),
        }

        if hasattr(self.audit_trail, "add_entry"):
            add_entry = getattr(self.audit_trail, "add_entry")
            parameters = inspect.signature(add_entry).parameters
            kwargs: Dict[str, Any] = {}
            if "metadata" in parameters:
                kwargs["metadata"] = {
                    "source": self.__class__.__name__,
                    "report_version": report.get("metadata", {}).get("report_version"),
                }
            if "mine" in parameters:
                kwargs["mine"] = False
            if "validate_evidence" in parameters:
                kwargs["validate_evidence"] = False
            block = add_entry(audit_payload, **kwargs)
        else:
            previous_hash = self._extract_block_hash(self.audit_trail.chain[-1]) if getattr(self.audit_trail, "chain", []) else "0" * 64
            block = self._instantiate_legacy_audit_block(audit_payload, previous_hash)
            if hasattr(block, "mine_block"):
                block.mine_block(getattr(self.audit_trail, "difficulty", 0))
            self.audit_trail.chain.append(block)

        logger.info("Audit artifact stored: hash=%s", self._extract_block_hash(block))
        return block

    def _store_report_version(self, report: Mapping[str, Any]) -> Any:
        add_version = getattr(self.doc_versioner, "add_version")
        parameters = inspect.signature(add_version).parameters
        kwargs: Dict[str, Any] = {}
        if "metadata" in parameters:
            kwargs["metadata"] = {
                "source": self.__class__.__name__,
                "report_version": report.get("metadata", {}).get("report_version"),
            }
        if "validate_schema" in parameters:
            kwargs["validate_schema"] = False
        version = add_version(dict(report), **kwargs)
        logger.info("Report version stored")
        return version

    def _submit_report_evidence(self, report: Mapping[str, Any]) -> Dict[str, Any]:
        evidence = self.cert_manager.submit_evidence(
            {
                "timestamp": _utcnow().isoformat(),
                "type": ["performance_report", "system_report"],
                "content": dict(report),
                "source": self.__class__.__name__,
                "metadata": {
                    "report_version": report.get("metadata", {}).get("report_version"),
                },
            }
        )
        if hasattr(evidence, "to_dict"):
            return evidence.to_dict()
        if isinstance(evidence, Mapping):
            return dict(evidence)
        return {"evidence": str(evidence)}

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def render_tradeoff_chart(self, size: Optional[QSize | Tuple[int, int]] = None) -> QPixmap:
        """Render a risk-versus-operational-time tradeoff chart."""
        self._ensure_qt_application()
        try:
            chart_size = self._coerce_size(size, self.chart_dimensions["tradeoff"])
            pixmap = self._create_canvas(chart_size)
            painter = QPainter(pixmap)
            try:
                plot_area = self._chart_rect(chart_size)
                self._draw_grid(
                    painter,
                    plot_area,
                    x_max=max(self.get_metric_history()["hazard_rates"] or [1.0]),
                    y_max=max(self.get_metric_history()["operational_times"] or [1.0]),
                )

                paired_points = list(zip(
                    self.get_metric_history()["hazard_rates"],
                    self.get_metric_history()["operational_times"],
                ))
                self._plot_xy_series(
                    painter,
                    plot_area,
                    paired_points,
                    self.colors["reward_line"],
                )
                self._draw_labels(
                    painter,
                    chart_size,
                    title="Risk / Operational Tradeoff",
                    x_label="Risk Estimate",
                    y_label="Operational Time",
                )
            finally:
                painter.end()
            return pixmap
        except EvaluationError:
            raise
        except Exception as exc:
            raise VisualizationError(
                chart_type="tradeoff",
                data=self.get_metric_history(),
                error_details=str(exc),
            ) from exc

    def render_temporal_chart(
        self,
        size: Optional[QSize | Tuple[int, int]] = None,
        metric: str = "hazard_rates",
        data: Optional[Sequence[float]] = None,
    ) -> QPixmap:
        """Render a temporal chart for a tracked metric series."""
        self._ensure_qt_application()
        normalized_metric = str(metric).strip().lower()
        metric_map = {
            "hazard_rates": ("Hazard Rate", self.colors["risk_line"]),
            "operational_times": ("Operational Time", self.colors["reward_line"]),
            "pass_rate": ("Pass Rate", self.colors["success"]),
            "pass_rates": ("Pass Rate", self.colors["success"]),
            "rewards": ("Reward", self.colors["reward_line"]),
            "risks": ("Risk Score", self.colors["risk_line"]),
        }

        if normalized_metric not in metric_map:
            raise VisualizationError(
                chart_type=normalized_metric,
                data=data or {},
                error_details="Unsupported temporal metric.",
            )

        label, color = metric_map[normalized_metric]
        internal_key = "pass_rates" if normalized_metric == "pass_rate" else normalized_metric
        chart_data = list(data) if data is not None else list(self.metrics.get(internal_key, []))

        try:
            chart_size = self._coerce_size(size, self.chart_dimensions["temporal"])
            pixmap = self._create_canvas(chart_size)
            painter = QPainter(pixmap)
            try:
                plot_area = self._chart_rect(chart_size)
                max_val = max(chart_data) if chart_data else 1.0
                self._draw_grid(painter, plot_area, x_max=max(len(chart_data), 1), y_max=max_val)
                self._plot_series(painter, plot_area, chart_data, color)
                self._draw_labels(
                    painter,
                    chart_size,
                    title=f"{label} Over Time",
                    x_label="Time Step",
                    y_label=label,
                )
            finally:
                painter.end()
            return pixmap
        except EvaluationError:
            raise
        except Exception as exc:
            raise VisualizationError(
                chart_type=normalized_metric,
                data=chart_data,
                error_details=str(exc),
            ) from exc

    def _generate_visualization_section(self) -> Dict[str, Any]:
        tradeoff = self._chart_to_asset("tradeoff", self.render_tradeoff_chart())
        hazard_trend = self._chart_to_asset(
            "hazard_rates",
            self.render_temporal_chart(metric="hazard_rates"),
        )
        operation_trend = self._chart_to_asset(
            "operational_times",
            self.render_temporal_chart(metric="operational_times"),
        )
        pass_rate_trend = self._chart_to_asset(
            "pass_rates",
            self.render_temporal_chart(metric="pass_rates"),
        )
        return {
            "tradeoff_chart": tradeoff.to_dict(),
            "hazard_rate_chart": hazard_trend.to_dict(),
            "operational_time_chart": operation_trend.to_dict(),
            "pass_rate_chart": pass_rate_trend.to_dict(),
        }

    # ------------------------------------------------------------------
    # Low-level chart helpers
    # ------------------------------------------------------------------

    def _ensure_qt_application(self) -> QApplication:
        application = QApplication.instance()
        if application is None:
            application = QApplication([])
        return application

    def _create_canvas(self, size: QSize) -> QPixmap:
        pixmap = QPixmap(size)
        pixmap.fill(self.colors["background"])
        return pixmap

    def _chart_to_asset(self, chart_type: str, pixmap: QPixmap) -> VisualizationAsset:
        encoded = self._chart_to_base64(pixmap)
        return VisualizationAsset(
            chart_type=chart_type,
            encoding="base64-png",
            width=pixmap.width(),
            height=pixmap.height(),
            image=encoded,
        )

    def _chart_to_base64(self, pixmap: QPixmap) -> str:
        if pixmap.isNull():
            raise VisualizationError(
                chart_type="pixmap",
                data={},
                error_details="Attempted to serialize a null pixmap.",
            )

        buffer = QBuffer()
        if not buffer.open(QBuffer.ReadWrite):
            raise VisualizationError(
                chart_type="pixmap",
                data={},
                error_details="Failed to allocate in-memory image buffer.",
            )

        if not pixmap.save(buffer, "PNG"):
            raise VisualizationError(
                chart_type="pixmap",
                data={},
                error_details="Failed to encode pixmap to PNG.",
            )

        return bytes(buffer.data().toBase64()).decode("utf-8")

    def _chart_rect(self, size: QSize) -> QRect:
        left_padding = 70
        top_padding = 45
        right_padding = 25
        bottom_padding = 60
        return QRect(
            left_padding,
            top_padding,
            max(10, size.width() - left_padding - right_padding),
            max(10, size.height() - top_padding - bottom_padding),
        )

    def _draw_grid(self, painter: QPainter, plot_area: QRect, x_max: float = 1.0, y_max: float = 1.0) -> None:
        pen = QPen(self.colors["text"], 1, self.grid_style)
        painter.setPen(pen)

        x_divisions = self.grid_config["x_divisions"]
        y_divisions = self.grid_config["y_divisions"]
        precision = self.grid_config["label_precision"]

        for index in range(x_divisions + 1):
            x = plot_area.left() + int((index / x_divisions) * plot_area.width())
            painter.drawLine(x, plot_area.top(), x, plot_area.bottom())
            label_value = x_max * (index / x_divisions)
            painter.drawText(x - 18, plot_area.bottom() + 20, f"{label_value:.{precision}f}")

        for index in range(y_divisions + 1):
            y = plot_area.bottom() - int((index / y_divisions) * plot_area.height())
            painter.drawLine(plot_area.left(), y, plot_area.right(), y)
            label_value = y_max * (index / y_divisions)
            painter.drawText(8, y + 5, f"{label_value:.{precision}f}")

    def _plot_series(
        self,
        painter: QPainter,
        plot_area: QRect,
        data: Sequence[float],
        color: QColor,
    ) -> None:
        if not data:
            self._draw_empty_state(painter, plot_area, "No data available")
            return

        pen = QPen(color, self.line_thickness)
        painter.setPen(pen)
        max_val = max(max(data), 1.0)
        x_step = plot_area.width() / max(len(data) - 1, 1)

        if len(data) == 1:
            x = plot_area.left() + plot_area.width() // 2
            y = plot_area.bottom() - int((data[0] / max_val) * plot_area.height())
            painter.drawEllipse(QPointF(x, y), 4, 4)
            return

        for index in range(1, len(data)):
            x1 = plot_area.left() + (index - 1) * x_step
            y1 = plot_area.bottom() - (data[index - 1] / max_val) * plot_area.height()
            x2 = plot_area.left() + index * x_step
            y2 = plot_area.bottom() - (data[index] / max_val) * plot_area.height()
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    def _plot_xy_series(
        self,
        painter: QPainter,
        plot_area: QRect,
        points: Sequence[Tuple[float, float]],
        color: QColor,
    ) -> None:
        if not points:
            self._draw_empty_state(painter, plot_area, "No paired metric data available")
            return

        pen = QPen(color, self.line_thickness)
        painter.setPen(pen)
        max_x = max(max((x for x, _ in points), default=1.0), 1.0)
        max_y = max(max((y for _, y in points), default=1.0), 1.0)

        if len(points) == 1:
            x, y = points[0]
            mapped_x = plot_area.left() + (x / max_x) * plot_area.width()
            mapped_y = plot_area.bottom() - (y / max_y) * plot_area.height()
            painter.drawEllipse(QPointF(mapped_x, mapped_y), 4, 4)
            return

        for index in range(1, len(points)):
            x1, y1 = points[index - 1]
            x2, y2 = points[index]
            mapped_x1 = plot_area.left() + (x1 / max_x) * plot_area.width()
            mapped_y1 = plot_area.bottom() - (y1 / max_y) * plot_area.height()
            mapped_x2 = plot_area.left() + (x2 / max_x) * plot_area.width()
            mapped_y2 = plot_area.bottom() - (y2 / max_y) * plot_area.height()
            painter.drawLine(QPointF(mapped_x1, mapped_y1), QPointF(mapped_x2, mapped_y2))

    def _draw_empty_state(self, painter: QPainter, plot_area: QRect, text: str) -> None:
        painter.setPen(QPen(self.colors["text"]))
        painter.drawText(plot_area, Qt.AlignCenter, text)

    def _draw_labels(
        self,
        painter: QPainter,
        size: QSize,
        title: str,
        x_label: str,
        y_label: str,
    ) -> None:
        font = QFont(self.font_config["family"], self.font_config["size"])
        painter.setFont(font)
        painter.setPen(QPen(self.colors["text"]))

        title_font = QFont(self.font_config["family"], self.font_config["title_size"])
        painter.setFont(title_font)
        painter.drawText(QRect(0, 10, size.width(), 24), Qt.AlignCenter, title)

        painter.setFont(font)
        painter.drawText(QRect(0, size.height() - 30, size.width(), 20), Qt.AlignCenter, x_label)

        painter.save()
        painter.translate(18, size.height() / 2)
        painter.rotate(-90)
        painter.drawText(QRect(0, 0, size.height(), 20), Qt.AlignCenter, y_label)
        painter.restore()

    # ------------------------------------------------------------------
    # Utility and compatibility helpers
    # ------------------------------------------------------------------

    def _generate_integrity_section(self, report: Mapping[str, Any]) -> Dict[str, Any]:
        serialized = _canonical_json(dict(report))
        return {
            "hash_algorithm": "sha256",
            "report_hash": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        }

    def _run_section(
        self,
        section_name: str,
        builder,
        errors: List[Dict[str, Any]],
        strict: bool,
    ) -> Dict[str, Any]:
        try:
            result = builder()
            if result is None:
                return {}
            if isinstance(result, Mapping):
                return dict(result)
            if hasattr(result, "to_dict"):
                return result.to_dict()
            return {"value": result}
        except EvaluationError as exc:
            logger.error("Failed to generate section '%s': %s", section_name, exc)
            if strict:
                raise
            errors.append(exc.to_audit_dict())
            return {"status": "ERROR", "error": exc.to_audit_dict()}
        except Exception as exc:
            wrapped = ReportGenerationError(
                report_type=section_name,
                template="runtime_section_builder",
                error_details=str(exc),
            )
            logger.error("Failed to generate section '%s': %s", section_name, exc)
            if strict:
                raise wrapped from exc
            errors.append(wrapped.to_audit_dict())
            return {"status": "ERROR", "error": wrapped.to_audit_dict()}

    def _verify_chain_if_supported(self) -> Optional[bool]:
        verifier = getattr(self.audit_trail, "verify_chain", None)
        if callable(verifier):
            try:
                return bool(verifier())
            except Exception as exc:
                logger.warning("Audit-chain verification failed during report assembly: %s", exc)
                return False
        return None

    def _summarize_version_record(self, payload: Any) -> Optional[Dict[str, Any]]:
        if payload is None:
            return None

        if hasattr(payload, "to_dict"):
            payload = payload.to_dict()

        if not isinstance(payload, Mapping):
            return {"value": str(payload)}

        result: Dict[str, Any] = {}
        if "timestamp" in payload:
            result["timestamp"] = payload["timestamp"]
        if "version_id" in payload:
            result["version_id"] = payload["version_id"]
        if "metadata" in payload:
            result["metadata"] = dict(payload["metadata"])
        if "hash" in payload:
            result["hash"] = payload["hash"]
        if "content_hash" in payload:
            result["hash"] = payload["content_hash"]
        if "content" in payload and isinstance(payload["content"], Mapping):
            result["content_keys"] = sorted(payload["content"].keys())
        return result

    def _extract_block_hash(self, block: Any) -> Optional[str]:
        if block is None:
            return None
        for field_name in ("block_hash", "hash"):
            value = getattr(block, field_name, None)
            if isinstance(value, str) and value:
                return value
        if isinstance(block, Mapping):
            return block.get("block_hash") or block.get("hash")
        return None

    def _instantiate_legacy_audit_block(self, data: Mapping[str, Any], previous_hash: str) -> Any:
        signature = inspect.signature(AuditBlock)
        parameters = list(signature.parameters.keys())
        if parameters[:2] == ["data", "previous_hash"]:
            return AuditBlock(data=dict(data), previous_hash=previous_hash)

        kwargs: Dict[str, Any] = {
            "index": len(getattr(self.audit_trail, "chain", [])),
            "timestamp": _utcnow().isoformat(),
            "data": dict(data),
            "previous_hash": previous_hash,
        }
        if "hash_algorithm" in signature.parameters:
            kwargs["hash_algorithm"] = getattr(self.audit_trail, "hash_algorithm_name", "sha256")
        return AuditBlock(**kwargs)

    def _load_colors(self, color_config: Mapping[str, Any]) -> Dict[str, QColor]:
        default_colors = {
            "background": (30, 30, 30),
            "text": (255, 255, 255),
            "reward_line": (0, 191, 99),
            "risk_line": (255, 144, 0),
            "success": (0, 191, 99),
            "failure": (231, 76, 60),
        }
        colors: Dict[str, QColor] = {}
        for key, default in default_colors.items():
            raw_value = color_config.get(key, default)
            if not isinstance(raw_value, (list, tuple)) or len(raw_value) < 3:
                raw_value = default
            red, green, blue = (int(raw_value[0]), int(raw_value[1]), int(raw_value[2]))
            colors[key] = QColor(red, green, blue)
        return colors

    def _load_chart_dimensions(self, dimensions_config: Mapping[str, Any]) -> Dict[str, QSize]:
        tradeoff_raw = dimensions_config.get("tradeoff", [800, 600])
        temporal_raw = dimensions_config.get("temporal", [600, 400])
        return {
            "tradeoff": self._coerce_size(tradeoff_raw, QSize(800, 600)),
            "temporal": self._coerce_size(temporal_raw, QSize(600, 400)),
        }

    def _load_grid_config(self, grid_config: Mapping[str, Any]) -> Dict[str, int]:
        return {
            "x_divisions": self._load_positive_int(grid_config.get("x_divisions", 10), "performance_visualizer.grid.x_divisions"),
            "y_divisions": self._load_positive_int(grid_config.get("y_divisions", 10), "performance_visualizer.grid.y_divisions"),
            "label_precision": max(0, self._coerce_non_negative_int(grid_config.get("label_precision", 1), "label_precision")),
        }

    def _load_font_config(self, font_config: Mapping[str, Any]) -> Dict[str, Any]:
        family = str(font_config.get("family", "Arial")).strip() or "Arial"
        size = self._load_positive_int(font_config.get("size", 10), "performance_visualizer.font.size")
        title_size = self._load_positive_int(
            font_config.get("title_size", max(size + 2, 12)),
            "performance_visualizer.font.title_size",
        )
        return {
            "family": family,
            "size": size,
            "title_size": title_size,
        }

    def _coerce_size(self, value: Any, fallback: QSize) -> QSize:
        if isinstance(value, QSize):
            return value
        if isinstance(value, (list, tuple)) and len(value) == 2:
            width = self._load_positive_int(value[0], "chart_width")
            height = self._load_positive_int(value[1], "chart_height")
            return QSize(width, height)
        if isinstance(fallback, QSize):
            return fallback
        raise OperationalError("Invalid chart size configuration.", context={"value": str(value)})

    def _extract_hazard_rate(self, hazards: Any) -> float:
        if isinstance(hazards, Mapping):
            if "system_failure" in hazards:
                return self._coerce_non_negative_float(hazards["system_failure"], "system_failure")
            numeric_values = [
                self._coerce_non_negative_float(value, key)
                for key, value in hazards.items()
                if isinstance(value, (int, float))
            ]
            return _safe_mean(numeric_values)
        return self._coerce_non_negative_float(hazards, "hazards")

    def _load_positive_int(self, value: Any, field_name: str) -> int:
        try:
            integer_value = int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigLoadError(
                config_path=str(self.config.get("__config_path__", "<unknown>")),
                section=field_name,
                error_details="Value must be coercible to a positive integer.",
            ) from exc

        if integer_value <= 0:
            raise ConfigLoadError(
                config_path=str(self.config.get("__config_path__", "<unknown>")),
                section=field_name,
                error_details="Value must be a positive integer.",
            )
        return integer_value

    @staticmethod
    def _coerce_non_negative_int(value: Any, field_name: str) -> int:
        try:
            integer_value = int(value)
        except (TypeError, ValueError) as exc:
            raise MetricCalculationError(field_name, value, "Value must be an integer.") from exc
        if integer_value < 0:
            raise MetricCalculationError(field_name, value, "Value must be non-negative.")
        return integer_value

    @staticmethod
    def _coerce_float(value: Any, field_name: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise MetricCalculationError(field_name, value, "Value must be numeric.") from exc

    @classmethod
    def _coerce_non_negative_float(cls, value: Any, field_name: str) -> float:
        numeric_value = cls._coerce_float(value, field_name)
        if numeric_value < 0:
            raise MetricCalculationError(field_name, value, "Value must be non-negative.")
        return numeric_value

    @classmethod
    def _coerce_probability(cls, value: Any, field_name: str) -> float:
        numeric_value = cls._coerce_float(value, field_name)
        if numeric_value < 0.0 or numeric_value > 1.0:
            raise MetricCalculationError(field_name, value, "Value must be between 0 and 1.")
        return numeric_value

    @classmethod
    def _coerce_numeric_sequence(cls, values: Any) -> List[float]:
        if values is None:
            return []
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise MetricCalculationError("sequence", values, "Expected a numeric sequence.")
        result: List[float] = []
        for item in values:
            result.append(cls._coerce_float(item, "sequence_item"))
        return result


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def get_visualizer(config: Optional[Mapping[str, Any]] = None) -> PerformanceVisualizer:
    """Return a shared visualizer instance, preserving the legacy factory API."""
    del config  # Configuration remains sourced from the central loader by design.

    global _global_visualizer
    if _global_visualizer is None:
        _global_visualizer = PerformanceVisualizer()
    return _global_visualizer


# ----------------------------------------------------------------------
# Module-level utilities
# ----------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _safe_mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return float(mean(materialized)) if materialized else 0.0


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    print("\n=== Running Production Report Pipeline ===\n")
    app = QApplication.instance() or QApplication(sys.argv)

    visualizer = PerformanceVisualizer()
    logger.info(visualizer)

    visualizer.update_metrics(
        {
            "successes": 5,
            "failures": 2,
            "hazards": {"system_failure": 0.12},
            "operational_time": 150.0,
            "reward": 0.82,
            "risk": 0.18,
        }
    )

    report = visualizer.generate_full_report(
        include_visualizations=True,
        include_audit_chain=False,
        include_version_history=False,
        audit_report=True,
        version_report=True,
        submit_report_evidence=False,
        strict=False,
    )

    print("=== Report Summary ===")
    print(json.dumps(report["summary"], indent=2))

    print("\n=== Certification Summary ===")
    print(json.dumps(report.get("certification", {}), indent=2, default=str))

    serialized = visualizer.export_report(report, format="json")
    print("\n=== Report JSON Preview ===")
    print(serialized[:1000] + ("..." if len(serialized) > 1000 else ""))

    print("\n=== Successfully Ran Production Report Pipeline ===\n")
