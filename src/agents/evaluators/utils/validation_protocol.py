from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .evaluation_errors import ConfigLoadError, OperationalError, ValidationFailureError
from .config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Validation Protocol")
printer = PrettyPrinter


@dataclass(slots=True)
class ValidationIssue:
    """Structured validation issue emitted during protocol verification."""

    path: str
    message: str
    severity: str = "error"
    remediation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationProtocolReport:
    """Serializable result of a validation protocol consistency check."""

    is_valid: bool
    issue_count: int
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    validated_sections: List[str] = field(default_factory=list)
    certification_template_summary: Dict[str, Any] = field(default_factory=dict)
    enabled_evaluation_flow: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["errors"] = [issue.to_dict() for issue in self.errors]
        payload["warnings"] = [issue.to_dict() for issue in self.warnings]
        return payload


class ValidationProtocol:
    """
    Production-grade validation protocol manager.

    Responsibilities
    ----------------
    - Load and normalize validation protocol configuration from the global config
    - Preserve sensible defaults while honoring explicit project overrides
    - Validate internal consistency across static analysis, behavioral testing,
      safety, performance, compliance, operational, and evaluation-flow settings
    - Cross-check protocol assumptions against certification templates
    """

    def __init__(self) -> None:
        self.config = load_global_config()
        self.config_path = str(self.config.get("__config_path__", "<config>"))
        self.template_path = self.config.get("template_path")

        self.protocol_config = get_config_section("validation_protocol")
        if not isinstance(self.protocol_config, Mapping):
            raise ConfigLoadError(
                self.config_path,
                "validation_protocol",
                "Section must be a mapping/object.",
            )

        self.certification_config = get_config_section("certification_manager")
        if self.certification_config and not isinstance(self.certification_config, Mapping):
            raise ConfigLoadError(
                self.config_path,
                "certification_manager",
                "Section must be a mapping/object when present.",
            )

        self.static_analysis = self._merge_dicts(
            self._default_static_analysis(),
            self._coerce_mapping(self.protocol_config.get("static_analysis", {}), "validation_protocol.static_analysis"),
        )
        self.behavioral_testing = self._merge_dicts(
            self._default_behavioral_testing(),
            self._coerce_mapping(self.protocol_config.get("behavioral_testing", {}), "validation_protocol.behavioral_testing"),
        )
        self.safety_constraints = self._merge_dicts(
            self._default_safety_constraints(),
            self._coerce_mapping(self.protocol_config.get("safety_constraints", {}), "validation_protocol.safety_constraints"),
        )
        self.performance_metrics = self._merge_dicts(
            self._default_performance_metrics(),
            self._coerce_mapping(self.protocol_config.get("performance_metrics", {}), "validation_protocol.performance_metrics"),
        )
        self.compliance = self._merge_dicts(
            self._default_compliance(),
            self._coerce_mapping(self.protocol_config.get("compliance", {}), "validation_protocol.compliance"),
        )
        self.operational = self._merge_dicts(
            self._default_operational(),
            self._coerce_mapping(self.protocol_config.get("operational", {}), "validation_protocol.operational"),
        )
        self.full_evaluation_flow = self._merge_dicts(
            self._default_full_evaluation_flow(),
            self._coerce_mapping(self.protocol_config.get("full_evaluation_flow", {}), "validation_protocol.full_evaluation_flow"),
        )

        self.domain = str(self.certification_config.get("domain", "automotive")).strip() or "automotive"
        self.system = str(self.certification_config.get("system", "UNKNOWN")).strip() or "UNKNOWN"

        logger.info("Validation Protocol successfully initialized")

    @property
    def validation_protocol(self) -> Dict[str, Any]:
        return self.export()

    def export(self) -> Dict[str, Any]:
        """Return a normalized snapshot of the active validation protocol."""
        return {
            "static_analysis": deepcopy(self.static_analysis),
            "behavioral_testing": deepcopy(self.behavioral_testing),
            "safety_constraints": deepcopy(self.safety_constraints),
            "performance_metrics": deepcopy(self.performance_metrics),
            "compliance": deepcopy(self.compliance),
            "operational": deepcopy(self.operational),
            "full_evaluation_flow": deepcopy(self.full_evaluation_flow),
        }

    def get_enabled_evaluation_flow(self) -> Dict[str, bool]:
        """Return the normalized evaluation-flow toggles."""
        return {
            key: bool(value)
            for key, value in self.full_evaluation_flow.items()
        }

    def validate_configuration(self, raise_on_error: bool = True) -> Dict[str, Any]:
        """
        Validate protocol settings and optionally raise on any hard errors.

        Parameters
        ----------
        raise_on_error:
            When True, raises ValidationFailureError if one or more hard errors
            are found. Warnings are returned but do not raise.
        """
        errors: List[ValidationIssue] = []
        warnings: List[ValidationIssue] = []
        validated_sections: List[str] = []

        self._validate_static_analysis(errors, warnings)
        validated_sections.append("static_analysis")

        self._validate_behavioral_testing(errors, warnings)
        validated_sections.append("behavioral_testing")

        self._validate_safety_constraints(errors, warnings)
        validated_sections.append("safety_constraints")

        self._validate_performance_metrics(errors, warnings)
        validated_sections.append("performance_metrics")

        self._validate_compliance(errors, warnings)
        validated_sections.append("compliance")

        self._validate_operational(errors, warnings)
        validated_sections.append("operational")

        self._validate_full_evaluation_flow(errors, warnings)
        validated_sections.append("full_evaluation_flow")

        template_summary = self._cross_check_certification_templates(errors, warnings)
        validated_sections.append("certification_templates")

        report = ValidationProtocolReport(
            is_valid=not errors,
            issue_count=len(errors) + len(warnings),
            errors=errors,
            warnings=warnings,
            validated_sections=validated_sections,
            certification_template_summary=template_summary,
            enabled_evaluation_flow=self.get_enabled_evaluation_flow(),
        )

        if errors:
            logger.error(
                "Validation Protocol consistency check failed with %d error(s) and %d warning(s)",
                len(errors),
                len(warnings),
            )
            if raise_on_error:
                raise ValidationFailureError(
                    "validation_protocol_consistency",
                    [issue.to_dict() for issue in errors],
                    "protocol configuration with no validation errors",
                )
        else:
            logger.info(
                "Validation Protocol configuration validated successfully with %d warning(s)",
                len(warnings),
            )

        return report.to_dict()

    # ------------------------------------------------------------------
    # Section validation
    # ------------------------------------------------------------------

    def _validate_static_analysis(
        self,
        errors: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ) -> None:
        config = self.static_analysis
        if not isinstance(config.get("enable"), bool):
            self._add_issue(errors, "static_analysis.enable", "Static analysis enable flag must be boolean.")

        security = self._expect_mapping(config, "security", "static_analysis.security", errors)
        code_quality = self._expect_mapping(config, "code_quality", "static_analysis.code_quality", errors)

        if security:
            for key in ("owasp_top_10", "cwe_top_25"):
                if not isinstance(security.get(key), bool):
                    self._add_issue(errors, f"static_analysis.security.{key}", f"'{key}' must be boolean.")
            for key in ("max_critical", "max_high"):
                self._validate_non_negative_integer(
                    security.get(key),
                    f"static_analysis.security.{key}",
                    errors,
                )

        if code_quality:
            tech_debt = code_quality.get("tech_debt_threshold")
            self._validate_probability(tech_debt, "static_analysis.code_quality.tech_debt_threshold", errors)

            coverage = code_quality.get("test_coverage")
            self._validate_probability(coverage, "static_analysis.code_quality.test_coverage", errors)

            complexity = self._expect_mapping(
                code_quality,
                "complexity",
                "static_analysis.code_quality.complexity",
                errors,
            )
            if complexity:
                for key in ("cyclomatic", "cognitive"):
                    self._validate_positive_integer(
                        complexity.get(key),
                        f"static_analysis.code_quality.complexity.{key}",
                        errors,
                    )

    def _validate_behavioral_testing(
        self,
        errors: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ) -> None:
        config = self.behavioral_testing
        test_types = config.get("test_types")
        if not isinstance(test_types, list) or not test_types:
            self._add_issue(errors, "behavioral_testing.test_types", "Test types must be a non-empty list.")
        elif any(not isinstance(item, str) or not item.strip() for item in test_types):
            self._add_issue(errors, "behavioral_testing.test_types", "Every test type must be a non-empty string.")

        sample_size = self._expect_mapping(config, "sample_size", "behavioral_testing.sample_size", errors)
        if sample_size:
            for key in ("nominal", "edge_cases", "adversarial"):
                self._validate_positive_integer(sample_size.get(key), f"behavioral_testing.sample_size.{key}", errors)

        tolerance = self._expect_mapping(config, "failure_tolerance", "behavioral_testing.failure_tolerance", errors)
        if tolerance:
            for key in ("critical", "high", "medium"):
                self._validate_probability(tolerance.get(key), f"behavioral_testing.failure_tolerance.{key}", errors)

            critical = tolerance.get("critical")
            high = tolerance.get("high")
            medium = tolerance.get("medium")
            if self._is_number(critical) and self._is_number(high) and self._is_number(medium):
                if not (float(critical) <= float(high) <= float(medium)):
                    self._add_issue(
                        errors,
                        "behavioral_testing.failure_tolerance",
                        "Failure tolerance thresholds must be monotonic: critical <= high <= medium.",
                    )

    def _validate_safety_constraints(
        self,
        errors: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ) -> None:
        config = self.safety_constraints
        odd = self._expect_mapping(config, "operational_design_domain", "safety_constraints.operational_design_domain", errors)
        mitigation = self._expect_mapping(config, "risk_mitigation", "safety_constraints.risk_mitigation", errors)
        ethical = self._expect_mapping(config, "ethical_requirements", "safety_constraints.ethical_requirements", errors)

        if odd:
            geography = odd.get("geography")
            if not isinstance(geography, str) or not geography.strip():
                self._add_issue(errors, "safety_constraints.operational_design_domain.geography", "Geography must be a non-empty string.")

            speed_range = odd.get("speed_range")
            if not isinstance(speed_range, (list, tuple)) or len(speed_range) != 2:
                self._add_issue(errors, "safety_constraints.operational_design_domain.speed_range", "Speed range must contain exactly two numeric bounds.")
            else:
                lower, upper = speed_range
                if not self._is_number(lower) or not self._is_number(upper):
                    self._add_issue(errors, "safety_constraints.operational_design_domain.speed_range", "Speed range bounds must be numeric.")
                elif float(lower) < 0 or float(upper) < 0 or float(lower) > float(upper):
                    self._add_issue(errors, "safety_constraints.operational_design_domain.speed_range", "Speed range bounds must satisfy 0 <= lower <= upper.")

            weather_conditions = odd.get("weather_conditions")
            if not isinstance(weather_conditions, list) or not weather_conditions:
                self._add_issue(errors, "safety_constraints.operational_design_domain.weather_conditions", "Weather conditions must be a non-empty list.")

        if mitigation:
            margins = self._expect_mapping(mitigation, "safety_margins", "safety_constraints.risk_mitigation.safety_margins", errors)
            if margins:
                for key in ("positional", "temporal"):
                    self._validate_positive_number(
                        margins.get(key),
                        f"safety_constraints.risk_mitigation.safety_margins.{key}",
                        errors,
                    )
            if not isinstance(mitigation.get("fail_operational"), bool):
                self._add_issue(errors, "safety_constraints.risk_mitigation.fail_operational", "Fail-operational flag must be boolean.")

        if ethical:
            self._validate_probability(
                ethical.get("fairness_threshold"),
                "safety_constraints.ethical_requirements.fairness_threshold",
                errors,
            )
            for key in ("bias_detection", "transparency"):
                value = ethical.get(key)
                if not isinstance(value, list) or not value:
                    self._add_issue(errors, f"safety_constraints.ethical_requirements.{key}", f"'{key}' must be a non-empty list.")

    def _validate_performance_metrics(
        self,
        errors: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ) -> None:
        config = self.performance_metrics
        accuracy = self._expect_mapping(config, "accuracy", "performance_metrics.accuracy", errors)
        efficiency = self._expect_mapping(config, "efficiency", "performance_metrics.efficiency", errors)
        robustness = self._expect_mapping(config, "robustness", "performance_metrics.robustness", errors)

        if accuracy:
            for key in ("min_precision", "min_recall", "f1_threshold"):
                self._validate_probability(accuracy.get(key), f"performance_metrics.accuracy.{key}", errors)

        if efficiency:
            self._validate_positive_number(efficiency.get("max_inference_time"), "performance_metrics.efficiency.max_inference_time", errors)
            self._validate_positive_number(efficiency.get("max_memory_usage"), "performance_metrics.efficiency.max_memory_usage", errors)
            self._validate_positive_number(efficiency.get("energy_efficiency"), "performance_metrics.efficiency.energy_efficiency", errors)

        if robustness:
            for key in ("noise_tolerance", "adversarial_accuracy", "distribution_shift"):
                self._validate_probability(robustness.get(key), f"performance_metrics.robustness.{key}", errors)

        if accuracy and self._is_number(accuracy.get("f1_threshold")):
            precision = accuracy.get("min_precision")
            recall = accuracy.get("min_recall")
            f1_threshold = accuracy.get("f1_threshold")
            if self._is_number(precision) and self._is_number(recall):
                upper_bound = min(float(precision), float(recall))
                if float(f1_threshold) > 1.0 or float(f1_threshold) < 0.0:
                    self._add_issue(errors, "performance_metrics.accuracy.f1_threshold", "F1 threshold must be in [0, 1].")
                elif float(f1_threshold) > max(float(precision), float(recall)):
                    self._add_issue(
                        warnings,
                        "performance_metrics.accuracy.f1_threshold",
                        "F1 threshold exceeds at least one component threshold; verify whether this is intentional.",
                        severity="warning",
                    )
                elif float(f1_threshold) > upper_bound + 0.1:
                    self._add_issue(
                        warnings,
                        "performance_metrics.accuracy.f1_threshold",
                        "F1 threshold is materially higher than the minimum precision/recall threshold; check consistency.",
                        severity="warning",
                    )

    def _validate_compliance(
        self,
        errors: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ) -> None:
        config = self.compliance
        frameworks = config.get("regulatory_frameworks")
        if not isinstance(frameworks, list) or not frameworks:
            self._add_issue(errors, "compliance.regulatory_frameworks", "Regulatory frameworks must be a non-empty list.")
        elif any(not isinstance(item, str) or not item.strip() for item in frameworks):
            self._add_issue(errors, "compliance.regulatory_frameworks", "Every regulatory framework must be a non-empty string.")

        certification_level = config.get("certification_level")
        if not isinstance(certification_level, str) or not certification_level.strip():
            self._add_issue(errors, "compliance.certification_level", "Certification level must be a non-empty string.")

        documentation = self._expect_mapping(config, "documentation", "compliance.documentation", errors)
        if documentation:
            required = documentation.get("required")
            if not isinstance(required, list) or not required:
                self._add_issue(errors, "compliance.documentation.required", "Required documentation list must be non-empty.")
            fmt = documentation.get("format")
            if not isinstance(fmt, str) or not fmt.strip():
                self._add_issue(errors, "compliance.documentation.format", "Documentation format must be a non-empty string.")

    def _validate_operational(
        self,
        errors: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ) -> None:
        config = self.operational
        update_policy = self._expect_mapping(config, "update_policy", "operational.update_policy", errors)
        resource_constraints = self._expect_mapping(config, "resource_constraints", "operational.resource_constraints", errors)

        if update_policy:
            self._validate_probability(update_policy.get("retrain_threshold"), "operational.update_policy.retrain_threshold", errors)

            rollback_strategy = update_policy.get("rollback_strategy")
            if not isinstance(rollback_strategy, str) or not rollback_strategy.strip():
                self._add_issue(errors, "operational.update_policy.rollback_strategy", "Rollback strategy must be a non-empty string.")

            validation_frequency = update_policy.get("validation_frequency")
            if not isinstance(validation_frequency, str) or not validation_frequency.strip():
                self._add_issue(errors, "operational.update_policy.validation_frequency", "Validation frequency must be a non-empty string.")

        if resource_constraints:
            self._validate_positive_integer(resource_constraints.get("max_compute_time"), "operational.resource_constraints.max_compute_time", errors)
            for key in ("allowed_hardware", "privacy"):
                value = resource_constraints.get(key)
                if not isinstance(value, list) or not value:
                    self._add_issue(errors, f"operational.resource_constraints.{key}", f"'{key}' must be a non-empty list.")

    def _validate_full_evaluation_flow(
        self,
        errors: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ) -> None:
        flow = self.full_evaluation_flow
        for key in (
            "enable_performance",
            "enable_efficiency",
            "enable_statistical",
            "enable_resource",
        ):
            if key in flow and not isinstance(flow.get(key), bool):
                self._add_issue(errors, f"full_evaluation_flow.{key}", "Evaluation flow flag must be boolean.")

        if flow and not any(bool(value) for value in flow.values() if isinstance(value, bool)):
            self._add_issue(
                warnings,
                "full_evaluation_flow",
                "All evaluation-flow stages are disabled. The protocol may be valid but operationally ineffective.",
                severity="warning",
            )

    def _cross_check_certification_templates(
        self,
        errors: List[ValidationIssue],
        warnings: List[ValidationIssue],
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "checked": False,
            "template_path": None,
            "domain": self.domain,
            "available_levels": [],
        }

        if not isinstance(self.template_path, str) or not self.template_path.strip():
            self._add_issue(
                errors,
                "template_path",
                "Global configuration key 'template_path' is required for certification template cross-checking.",
            )
            return summary

        resolved_path = self._resolve_config_path(self.template_path)
        summary["template_path"] = str(resolved_path)

        try:
            with open(resolved_path, "r", encoding="utf-8") as handle:
                templates = json.load(handle)
        except FileNotFoundError as exc:
            self._add_issue(
                errors,
                "template_path",
                f"Certification template file not found: {resolved_path}",
                remediation="Verify the template_path value and the repository layout.",
            )
            logger.error("Certification template verification failed: %s", exc)
            return summary
        except json.JSONDecodeError as exc:
            self._add_issue(
                errors,
                "template_path",
                f"Certification template file is not valid JSON: {resolved_path}",
                remediation="Fix malformed JSON in the certification template file.",
            )
            logger.error("Certification template verification failed: %s", exc)
            return summary
        except OSError as exc:
            self._add_issue(
                errors,
                "template_path",
                f"Unable to read certification template file: {resolved_path}",
                remediation="Check file permissions and filesystem availability.",
            )
            logger.error("Certification template verification failed: %s", exc)
            return summary

        if not isinstance(templates, Mapping):
            self._add_issue(errors, "template_path", "Certification templates must deserialize to a JSON object.")
            return summary

        summary["checked"] = True
        domain_templates = templates.get(self.domain)
        if not isinstance(domain_templates, Mapping):
            self._add_issue(
                errors,
                "certification_manager.domain",
                f"Configured domain '{self.domain}' not found in certification templates.",
            )
            return summary

        available_levels = sorted(str(level).strip().upper() for level in domain_templates.keys())
        summary["available_levels"] = available_levels

        required_levels = ["DEVELOPMENT", "PILOT", "DEPLOYMENT", "CRITICAL"]
        for level in required_levels:
            if level not in available_levels:
                self._add_issue(
                    errors,
                    f"templates.{self.domain}.{level}",
                    f"Certification template missing phase: {level}",
                )

        configured_level = self.compliance.get("certification_level")
        if isinstance(configured_level, str) and configured_level.strip():
            if configured_level.strip().upper() == "ASIL-D" and "CRITICAL" not in available_levels:
                self._add_issue(
                    warnings,
                    "compliance.certification_level",
                    "Compliance targets ASIL-D-like rigor, but CRITICAL certification templates are unavailable.",
                    severity="warning",
                )

        return summary

    # ------------------------------------------------------------------
    # Defaults and normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _default_static_analysis() -> Dict[str, Any]:
        return {
            "enable": True,
            "security": {
                "owasp_top_10": True,
                "cwe_top_25": True,
                "max_critical": 0,
                "max_high": 3,
            },
            "code_quality": {
                "tech_debt_threshold": 0.15,
                "test_coverage": 0.8,
                "complexity": {
                    "cyclomatic": 15,
                    "cognitive": 20,
                },
            },
        }

    @staticmethod
    def _default_behavioral_testing() -> Dict[str, Any]:
        return {
            "test_types": ["unit", "integration", "adversarial", "stress"],
            "sample_size": {
                "nominal": 1000,
                "edge_cases": 100,
                "adversarial": 50,
            },
            "failure_tolerance": {
                "critical": 0.0,
                "high": 0.01,
                "medium": 0.05,
            },
        }

    @staticmethod
    def _default_safety_constraints() -> Dict[str, Any]:
        return {
            "operational_design_domain": {
                "geography": "global",
                "speed_range": (0, 120),
                "weather_conditions": ["clear", "rain", "snow"],
            },
            "risk_mitigation": {
                "safety_margins": {
                    "positional": 1.5,
                    "temporal": 2.0,
                },
                "fail_operational": True,
            },
            "ethical_requirements": {
                "fairness_threshold": 0.8,
                "bias_detection": ["gender", "ethnicity", "age"],
                "transparency": ["decision_logging", "explanation_generation"],
            },
        }

    @staticmethod
    def _default_performance_metrics() -> Dict[str, Any]:
        return {
            "accuracy": {
                "min_precision": 0.95,
                "min_recall": 0.90,
                "f1_threshold": 0.925,
            },
            "efficiency": {
                "max_inference_time": 100,
                "max_memory_usage": 512,
                "energy_efficiency": 0.5,
            },
            "robustness": {
                "noise_tolerance": 0.2,
                "adversarial_accuracy": 0.85,
                "distribution_shift": 0.15,
            },
        }

    @staticmethod
    def _default_compliance() -> Dict[str, Any]:
        return {
            "regulatory_frameworks": ["ISO 26262", "EU AI Act", "SAE J3016"],
            "certification_level": "ASIL-D",
            "documentation": {
                "required": ["safety_case", "test_reports", "risk_assessment"],
                "format": "ISO/IEC 15288",
            },
        }

    @staticmethod
    def _default_operational() -> Dict[str, Any]:
        return {
            "update_policy": {
                "retrain_threshold": 0.10,
                "rollback_strategy": "versioned",
                "validation_frequency": "continuous",
            },
            "resource_constraints": {
                "max_compute_time": 3600,
                "allowed_hardware": ["CPU", "GPU"],
                "privacy": ["differential_privacy", "on-device_processing"],
            },
        }

    @staticmethod
    def _default_full_evaluation_flow() -> Dict[str, bool]:
        return {
            "enable_performance": True,
            "enable_efficiency": True,
            "enable_statistical": True,
            "enable_resource": True,
        }

    def _resolve_config_path(self, configured_path: str) -> Path:
        if not isinstance(configured_path, str) or not configured_path.strip():
            raise OperationalError(
                "Configured path must be a non-empty string.",
                context={"configured_path": configured_path},
            )

        candidate = Path(configured_path)
        if candidate.is_absolute():
            return candidate

        config_file = self.config.get("__config_path__")
        if not config_file:
            return candidate

        return Path(config_file).resolve().parent / candidate

    @staticmethod
    def _merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
        merged = deepcopy(dict(base))
        for key, value in override.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = ValidationProtocol._merge_dicts(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged

    @staticmethod
    def _coerce_mapping(value: Any, section_name: str) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ConfigLoadError(
                "<config>",
                section_name,
                "Section must be a mapping/object.",
            )
        return dict(value)

    @staticmethod
    def _expect_mapping(
        mapping: Mapping[str, Any],
        key: str,
        path: str,
        errors: List[ValidationIssue],
    ) -> Dict[str, Any]:
        value = mapping.get(key)
        if not isinstance(value, Mapping):
            errors.append(ValidationIssue(path=path, message=f"'{key}' must be a mapping.", severity="error"))
            return {}
        return dict(value)

    @staticmethod
    def _add_issue(
        bucket: List[ValidationIssue],
        path: str,
        message: str,
        severity: str = "error",
        remediation: Optional[str] = None,
    ) -> None:
        bucket.append(
            ValidationIssue(
                path=path,
                message=message,
                severity=severity,
                remediation=remediation,
            )
        )

    def _validate_probability(self, value: Any, path: str, errors: List[ValidationIssue]) -> None:
        if not self._is_number(value):
            self._add_issue(errors, path, "Value must be numeric and within [0, 1].")
            return
        numeric = float(value)
        if numeric < 0.0 or numeric > 1.0:
            self._add_issue(errors, path, "Value must be within [0, 1].")

    def _validate_positive_number(self, value: Any, path: str, errors: List[ValidationIssue]) -> None:
        if not self._is_number(value) or float(value) <= 0.0:
            self._add_issue(errors, path, "Value must be a positive number.")

    def _validate_non_negative_integer(self, value: Any, path: str, errors: List[ValidationIssue]) -> None:
        if not isinstance(value, int) or value < 0:
            self._add_issue(errors, path, "Value must be a non-negative integer.")

    def _validate_positive_integer(self, value: Any, path: str, errors: List[ValidationIssue]) -> None:
        if not isinstance(value, int) or value <= 0:
            self._add_issue(errors, path, "Value must be a positive integer.")

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)


if __name__ == "__main__":
    protocol = ValidationProtocol()
    result = protocol.validate_configuration(raise_on_error=False)
    printer.pretty("Validation Protocol Report", result, "success")
