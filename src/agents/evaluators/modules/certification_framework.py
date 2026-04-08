"""
Production-ready certification framework for multi-stage safety and compliance
assessment.

This module implements a structured certification workflow inspired by:
- UL 4600 safety-case driven assurance
- ISO 26262 functional safety maturity and ASIL reasoning
- ISO 25010 software quality characteristics
- NIST RMF-style risk integration
- EU AI Act-oriented risk documentation support
"""

from __future__ import annotations

import hashlib
import json

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..utils.evaluation_errors import (CertificationFrameworkError, CertificationConfigurationError,
                                TemplateLoadError, CertificationLevel, EvidenceValidationError,
                                RequirementDefinitionError, CertificationEvaluationError)
from ..utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Certification Framework")
printer = PrettyPrinter


@dataclass(slots=True)
class SafetyCase:
    """
    UL 4600-style safety case container.

    The structure keeps the classic separation of goals, arguments, and
    evidence while allowing metadata to be attached for traceability.
    """

    goals: List[str] = field(default_factory=list)
    arguments: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_goal(self, text: str) -> None:
        self._validate_non_empty_string(text, "goal")
        self.goals.append(text.strip())

    def add_argument(self, claim: str) -> None:
        self._validate_non_empty_string(claim, "argument")
        self.arguments.append(claim.strip())

    def add_evidence(self, item: str | Mapping[str, Any], category: str = "generic") -> None:
        if isinstance(item, str):
            if not item.strip():
                raise ValueError("Safety case evidence text cannot be empty.")
            payload = {"category": category, "reference": item.strip()}
        elif isinstance(item, Mapping):
            payload = dict(item)
            payload.setdefault("category", category)
        else:
            raise TypeError("Safety case evidence must be a string or mapping.")

        self.evidence.append(payload)

    def export(self) -> Dict[str, Any]:
        return {
            "goals": list(self.goals),
            "arguments": list(self.arguments),
            "evidence": list(self.evidence),
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def _validate_non_empty_string(value: str, field_name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Safety case {field_name} must be a non-empty string.")


@dataclass(slots=True)
class CertificationRequirement:
    """Single certification requirement loaded from the domain template."""

    description: str
    test_method: str
    passing_condition: str
    evidence_required: List[str]
    requirement_id: str = ""
    level: Optional[CertificationLevel] = None

    def __post_init__(self) -> None:
        for field_name in ("description", "test_method", "passing_condition"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise RequirementDefinitionError(
                    f"Requirement field '{field_name}' must be a non-empty string."
                )
            setattr(self, field_name, value.strip())

        if not isinstance(self.evidence_required, list) or not self.evidence_required:
            raise RequirementDefinitionError(
                "Requirement field 'evidence_required' must be a non-empty list."
            )

        normalized_types: List[str] = []
        seen: set[str] = set()
        for item in self.evidence_required:
            if not isinstance(item, str) or not item.strip():
                raise RequirementDefinitionError(
                    "Each evidence type must be a non-empty string."
                )
            normalized = item.strip()
            key = normalized.casefold()
            if key not in seen:
                normalized_types.append(normalized)
                seen.add(key)
        self.evidence_required = normalized_types

        if not self.requirement_id:
            digest = hashlib.sha256(self.description.encode("utf-8")).hexdigest()[:12]
            self.requirement_id = f"REQ-{digest.upper()}"


@dataclass(slots=True)
class EvidenceRecord:
    """Normalized evidence artifact stored in the certification registry."""

    timestamp: str
    evidence_types: List[str]
    content_hash: str
    content: Any
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    evidence_id: str = ""

    def __post_init__(self) -> None:
        parsed = _coerce_iso8601_timestamp(self.timestamp)
        self.timestamp = parsed.isoformat()

        normalized: List[str] = []
        seen: set[str] = set()
        for item in self.evidence_types:
            if not isinstance(item, str) or not item.strip():
                raise EvidenceValidationError("Evidence types must contain non-empty strings.")
            cleaned = item.strip()
            key = cleaned.casefold()
            if key not in seen:
                normalized.append(cleaned)
                seen.add(key)
        if not normalized:
            raise EvidenceValidationError("At least one evidence type is required.")
        self.evidence_types = normalized

        if not isinstance(self.content_hash, str) or len(self.content_hash) < 16:
            raise EvidenceValidationError("Evidence content hash is missing or invalid.")

        if self.source is not None and (not isinstance(self.source, str) or not self.source.strip()):
            raise EvidenceValidationError("Evidence source must be a non-empty string when provided.")

        if not self.evidence_id:
            digest_source = f"{self.timestamp}:{self.content_hash}:{','.join(self.evidence_types)}"
            self.evidence_id = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RequirementEvaluation:
    """Structured evaluation result for an individual requirement."""

    requirement_id: str
    description: str
    passed: bool
    missing_evidence_types: List[str] = field(default_factory=list)
    matched_evidence_ids: List[str] = field(default_factory=list)
    test_method: str = ""
    passing_condition: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CertificationStatus:
    """Accumulated assurance status across standards and risk lenses."""

    quality_characteristics: Dict[str, str] = field(default_factory=dict)
    iso25010_pass: bool = False
    iso26262_asil: str = "ASIL-A"
    ul4600_valid: bool = False
    nist_rmf_risks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass(slots=True)
class CertificationReport:
    """Detailed output of a certification evaluation cycle."""

    system: str
    domain: str
    level: str
    passed: bool
    evaluated_at: str
    requirement_results: List[RequirementEvaluation]
    unmet_requirements: List[str]
    evidence_count: int
    evidence_inventory: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["requirement_results"] = [result.to_dict() for result in self.requirement_results]
        return payload


class CertificationManager:
    """
    End-to-end certification lifecycle handler.

    Responsibilities
    ----------------
    - Load and validate domain certification requirements
    - Normalize evidence into a registry suitable for auditing
    - Evaluate requirement coverage at a specific certification level
    - Generate structured certification reports and certificate payloads
    """

    def __init__(self) -> None:
        self.config = load_global_config()
        self.template_path = self.config.get("template_path")
        self.certification_config = get_config_section("certification_manager")

        self.system = self.certification_config.get("system", "UNKNOWN")
        self.domain = self.certification_config.get("domain")
        self.security = self.certification_config.get("security", {})
        self.reliability = self.certification_config.get("reliability", {})
        self.performance = self.certification_config.get("performance", {})
        self.maintainability = self.certification_config.get("maintainability", {})

        if not isinstance(self.domain, str) or not self.domain.strip():
            raise CertificationConfigurationError(
                "The 'certification_manager.domain' configuration value is required."
            )

        self.current_level = CertificationLevel.DEVELOPMENT
        self.evidence_registry: List[EvidenceRecord] = []
        self.requirements = self._load_domain_requirements()

        logger.info("Certification Manager successfully initialized")

    def _resolve_config_path(self, configured_path: str) -> Path:
        if not isinstance(configured_path, str) or not configured_path.strip():
            raise CertificationConfigurationError("Configured path must be a non-empty string.")

        candidate = Path(configured_path)
        if candidate.is_absolute():
            return candidate

        config_file = self.config.get("__config_path__")
        if not config_file:
            return candidate

        return Path(config_file).resolve().parent / candidate

    def _load_domain_requirements(self) -> Dict[CertificationLevel, List[CertificationRequirement]]:
        """Load and validate requirement templates for the configured domain."""
        if not self.template_path:
            raise CertificationConfigurationError(
                "Global configuration key 'template_path' is required for certification templates."
            )

        template_path = self._resolve_config_path(self.template_path)
        logger.info("Loading certification templates from %s", template_path)

        try:
            with open(template_path, "r", encoding="utf-8") as handle:
                templates = json.load(handle)
        except FileNotFoundError as exc:
            raise TemplateLoadError(
                f"Certification template file not found: {template_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise TemplateLoadError(
                f"Certification template file is not valid JSON: {template_path}"
            ) from exc
        except OSError as exc:
            raise TemplateLoadError(
                f"Unable to read certification template file: {template_path}"
            ) from exc

        if not isinstance(templates, dict):
            raise TemplateLoadError("Certification templates must deserialize to a JSON object.")

        domain_templates = templates.get(self.domain)
        if not isinstance(domain_templates, dict) or not domain_templates:
            raise TemplateLoadError(
                f"No certification templates found for configured domain '{self.domain}'."
            )

        requirements: Dict[CertificationLevel, List[CertificationRequirement]] = {}
        invalid_levels: List[str] = []

        for level_name, raw_requirements in domain_templates.items():
            try:
                level = CertificationLevel[str(level_name).strip().upper()]
            except KeyError:
                invalid_levels.append(str(level_name))
                logger.warning("Skipping unsupported certification level '%s'", level_name)
                continue

            if not isinstance(raw_requirements, list):
                raise RequirementDefinitionError(
                    f"Requirement list for level '{level_name}' must be a list."
                )

            parsed_requirements: List[CertificationRequirement] = []
            for index, raw_requirement in enumerate(raw_requirements, start=1):
                if not isinstance(raw_requirement, dict):
                    raise RequirementDefinitionError(
                        f"Requirement #{index} in level '{level_name}' must be a JSON object."
                    )

                requirement = CertificationRequirement(
                    description=raw_requirement["description"],
                    test_method=raw_requirement["test_method"],
                    passing_condition=raw_requirement["passing_condition"],
                    evidence_required=raw_requirement["evidence_required"],
                    requirement_id=raw_requirement.get("requirement_id", ""),
                    level=level,
                )
                parsed_requirements.append(requirement)

            requirements[level] = parsed_requirements

        if not requirements:
            raise RequirementDefinitionError(
                f"No valid certification levels were loaded for domain '{self.domain}'."
            )

        if invalid_levels:
            logger.warning(
                "Ignored unsupported certification levels for domain '%s': %s",
                self.domain,
                ", ".join(invalid_levels),
            )

        return requirements

    def set_certification_level(self, level: CertificationLevel | str) -> CertificationLevel:
        """Set the active certification level for subsequent evaluations."""
        if isinstance(level, str):
            try:
                level = CertificationLevel[level.strip().upper()]
            except KeyError as exc:
                raise CertificationConfigurationError(
                    f"Unknown certification level: {level}"
                ) from exc

        self.current_level = level
        logger.info("Active certification level set to %s", level.value)
        return self.current_level

    def submit_evidence(self, evidence: Mapping[str, Any]) -> EvidenceRecord:
        """
        Add validation evidence to the certification package.

        Expected input keys
        -------------------
        timestamp: ISO-8601 timestamp
        type:      string or list of strings describing the evidence artifact types
        content:   payload used to derive a deterministic content hash

        Optional keys
        -------------
        source:    system or subsystem that produced the evidence
        metadata:  free-form structured metadata
        """
        if not isinstance(evidence, Mapping):
            raise EvidenceValidationError("Evidence must be provided as a mapping.")

        missing_fields = [field for field in ("timestamp", "type", "content") if field not in evidence]
        if missing_fields:
            raise EvidenceValidationError(
                f"Evidence is missing required fields: {', '.join(missing_fields)}"
            )

        evidence_types = evidence["type"]
        if isinstance(evidence_types, str):
            normalized_types = [evidence_types]
        elif isinstance(evidence_types, Sequence):
            normalized_types = list(evidence_types)
        else:
            raise EvidenceValidationError(
                "Evidence field 'type' must be a string or a sequence of strings."
            )

        serialized_content = self._canonical_json(evidence["content"])
        content_hash = hashlib.sha256(serialized_content.encode("utf-8")).hexdigest()

        record = EvidenceRecord(
            timestamp=str(evidence["timestamp"]),
            evidence_types=normalized_types,
            content_hash=content_hash,
            content=evidence["content"],
            source=evidence.get("source"),
            metadata=dict(evidence.get("metadata", {})),
        )
        self.evidence_registry.append(record)
        logger.info(
            "Evidence submitted: id=%s types=%s source=%s",
            record.evidence_id,
            record.evidence_types,
            record.source or "<unspecified>",
        )
        return record

    def register_bulk_evidence(self, evidence_items: Iterable[Mapping[str, Any]]) -> List[EvidenceRecord]:
        """Convenience helper for batch evidence submission."""
        records: List[EvidenceRecord] = []
        for evidence in evidence_items:
            records.append(self.submit_evidence(evidence))
        return records

    def get_requirements_for_level(
        self, level: CertificationLevel | str | None = None
    ) -> List[CertificationRequirement]:
        target_level = self.current_level if level is None else self._coerce_level(level)
        return list(self.requirements.get(target_level, []))

    def get_evidence_inventory(self) -> Dict[str, int]:
        """Return a count of evidence types currently registered."""
        inventory: Dict[str, int] = {}
        for record in self.evidence_registry:
            for evidence_type in record.evidence_types:
                inventory[evidence_type] = inventory.get(evidence_type, 0) + 1
        return dict(sorted(inventory.items(), key=lambda item: item[0].casefold()))

    def generate_certificate(self, validity_days: int = 365) -> Dict[str, Any]:
        """Generate an auditable certificate payload from the current evaluation state."""
        if validity_days <= 0:
            raise ValueError("Certificate validity must be greater than zero days.")

        report = self.evaluate_certification_detailed()
        issued_at = _utcnow()
        certificate = {
            "system": self.system,
            "domain": self.domain,
            "level": self.current_level.value,
            "status": "PASSED" if report.passed else "FAILED",
            "issued_at": issued_at.isoformat(),
            "valid_until": (issued_at + timedelta(days=validity_days)).isoformat(),
            "requirements": [req.description for req in self.get_requirements_for_level()],
            "unmet_requirements": list(report.unmet_requirements),
            "evidence_count": report.evidence_count,
        }
        return certificate

    def evaluate_certification(self) -> Tuple[bool, List[str]]:
        """
        Compatibility method returning only pass/fail and unmet requirement text.

        Use :meth:`evaluate_certification_detailed` when a structured report is
        required by downstream automation or dashboards.
        """
        report = self.evaluate_certification_detailed()
        return report.passed, report.unmet_requirements

    def evaluate_certification_detailed(self) -> CertificationReport:
        """Run a detailed certification evaluation for the current level."""
        level_requirements = self.get_requirements_for_level(self.current_level)
        if not level_requirements:
            raise CertificationEvaluationError(
                f"No requirements available for certification level '{self.current_level.value}'."
            )

        evaluation_results: List[RequirementEvaluation] = []
        unmet_requirements: List[str] = []

        for requirement in level_requirements:
            matched_records = self._find_matching_evidence(requirement)
            covered_types = {
                evidence_type.casefold()
                for record in matched_records
                for evidence_type in record.evidence_types
            }
            missing_types = [
                evidence_type
                for evidence_type in requirement.evidence_required
                if evidence_type.casefold() not in covered_types
            ]
            passed = not missing_types
            if not passed:
                unmet_requirements.append(requirement.description)

            evaluation_results.append(
                RequirementEvaluation(
                    requirement_id=requirement.requirement_id,
                    description=requirement.description,
                    passed=passed,
                    missing_evidence_types=missing_types,
                    matched_evidence_ids=[record.evidence_id for record in matched_records],
                    test_method=requirement.test_method,
                    passing_condition=requirement.passing_condition,
                )
            )

        report = CertificationReport(
            system=self.system,
            domain=self.domain,
            level=self.current_level.value,
            passed=not unmet_requirements,
            evaluated_at=_utcnow().isoformat(),
            requirement_results=evaluation_results,
            unmet_requirements=unmet_requirements,
            evidence_count=len(self.evidence_registry),
            evidence_inventory=self.get_evidence_inventory(),
        )
        logger.info(
            "Certification evaluation completed: level=%s passed=%s unmet=%d",
            self.current_level.value,
            report.passed,
            len(unmet_requirements),
        )
        return report

    def _find_matching_evidence(self, requirement: CertificationRequirement) -> List[EvidenceRecord]:
        """
        Return evidence records relevant to the requirement.

        Matching is intentionally cumulative across the registry. A requirement
        can be satisfied by multiple submitted evidence items as long as the
        union of their types covers the required evidence set.
        """
        matched_records: List[EvidenceRecord] = []
        target_types = {item.casefold() for item in requirement.evidence_required}
        for record in self.evidence_registry:
            evidence_types = {item.casefold() for item in record.evidence_types}
            if target_types.intersection(evidence_types):
                matched_records.append(record)
        return matched_records

    def _coerce_level(self, level: CertificationLevel | str) -> CertificationLevel:
        if isinstance(level, CertificationLevel):
            return level
        try:
            return CertificationLevel[level.strip().upper()]
        except KeyError as exc:
            raise CertificationConfigurationError(
                f"Unknown certification level: {level}"
            ) from exc

    @staticmethod
    def _canonical_json(payload: Any) -> str:
        try:
            return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        except (TypeError, ValueError) as exc:
            raise EvidenceValidationError(
                f"Evidence content cannot be serialized deterministically: {exc}"
            ) from exc


class CertificationAuditor:
    """
    Cross-standard assurance evaluator.

    The auditor complements the template-driven manager by calculating
    software-quality checks, safety posture, and risk indicators that can be
    folded into external evidence submissions or reporting workflows.
    """

    def __init__(self) -> None:
        self.config = load_global_config()
        self.certification_config = get_config_section("certification_manager")
        self.safety_case = SafetyCase(metadata={"framework": "UL4600-inspired"})
        self.status = CertificationStatus()

        self.security_thresholds = self.certification_config.get("security", {})
        self.reliability_thresholds = self.certification_config.get("reliability", {})
        self.performance_thresholds = self.certification_config.get("performance", {})
        self.maintainability_thresholds = self.certification_config.get("maintainability", {})

        logger.info("Certification Auditor successfully initialized")

    def assess_iso25010(self, metrics: Mapping[str, float]) -> Dict[str, bool]:
        """
        Assess a subset of ISO 25010 characteristics against configured thresholds.

        Expected metric keys
        --------------------
        mtbf, response_time, tech_debt, vuln_count
        """
        if not isinstance(metrics, Mapping):
            raise ValueError("ISO 25010 metrics must be provided as a mapping.")

        mtbf_threshold = float(self.reliability_thresholds.get("mtbf_threshold", 1000))
        response_threshold = float(self.performance_thresholds.get("max_response_time", 1.0))
        tech_debt_threshold = float(self.maintainability_thresholds.get("max_tech_debt", 0.2))
        max_vulnerabilities = float(self.security_thresholds.get("max_vulnerabilities", 5))

        mtbf = _coerce_non_negative_float(metrics.get("mtbf", 0), "mtbf")
        response_time = _coerce_non_negative_float(metrics.get("response_time", 0), "response_time")
        tech_debt = _coerce_non_negative_float(metrics.get("tech_debt", 0), "tech_debt")
        vuln_count = _coerce_non_negative_float(metrics.get("vuln_count", 0), "vuln_count")

        checks = {
            "reliability": mtbf >= mtbf_threshold,
            "performance_efficiency": response_time <= response_threshold,
            "maintainability": tech_debt <= tech_debt_threshold,
            "security": vuln_count <= max_vulnerabilities,
        }

        self.status.quality_characteristics = {
            key: ("Pass" if passed else "Fail") for key, passed in checks.items()
        }
        self.status.iso25010_pass = all(checks.values())

        if not self.status.iso25010_pass:
            self.status.recommendations.append(
                "Address failing ISO 25010 characteristics before formal certification review."
            )

        return checks

    def evaluate_asil(self, coverage: float, test_count: int) -> str:
        """
        Derive a coarse ASIL readiness band from verification coverage.

        This is not a replacement for a full HARA/TSC/ASIL decomposition process,
        but it is suitable as a readiness indicator inside the evaluation stack.
        """
        coverage_value = _coerce_probability(coverage, "coverage")
        if not isinstance(test_count, int) or test_count < 0:
            raise ValueError("test_count must be a non-negative integer.")

        if coverage_value >= 0.95 and test_count >= 10000:
            level = "ASIL-D"
        elif coverage_value >= 0.90 and test_count >= 5000:
            level = "ASIL-C"
        elif coverage_value >= 0.80 and test_count >= 1000:
            level = "ASIL-B"
        else:
            level = "ASIL-A"

        self.status.iso26262_asil = level
        return level

    def finalize_ul4600(
        self,
        evidence_logs: Sequence[str | Mapping[str, Any]],
        goals: Optional[Sequence[str]] = None,
        arguments: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build and finalize the safety case used for certification reporting."""
        goals = list(goals or ["Prevent catastrophic failures in SLAI agents."])
        arguments = list(arguments or ["All critical paths are tested and analyzed."])

        if metadata:
            self.safety_case.metadata.update(dict(metadata))

        for goal in goals:
            self.safety_case.add_goal(goal)
        for argument in arguments:
            self.safety_case.add_argument(argument)
        for entry in evidence_logs:
            self.safety_case.add_evidence(entry, category="verification")

        self.status.ul4600_valid = bool(self.safety_case.goals and self.safety_case.arguments)
        return self.safety_case.export()

    def integrate_nist_rmf(self, metrics: Mapping[str, float]) -> List[str]:
        """Derive RMF-oriented operational risks from supplied metrics."""
        if not isinstance(metrics, Mapping):
            raise ValueError("NIST RMF metrics must be provided as a mapping.")

        risks: List[str] = []
        distribution_shift = _coerce_non_negative_float(
            metrics.get("distribution_shift", 0), "distribution_shift"
        )
        fairness_score = _coerce_probability(metrics.get("fairness_score", 1.0), "fairness_score")
        residual_risk = _coerce_probability(metrics.get("residual_risk", 0.0), "residual_risk")
        incident_rate = _coerce_non_negative_float(metrics.get("incident_rate", 0.0), "incident_rate")

        if distribution_shift > 0.20:
            risks.append("Concept drift detected")
        if fairness_score < 0.80:
            risks.append("Potential bias risk")
        if residual_risk > 0.30:
            risks.append("Residual operational risk exceeds tolerance")
        if incident_rate > 0.05:
            risks.append("Operational incident rate is elevated")

        self.status.nist_rmf_risks = risks
        if risks:
            self.status.recommendations.append(
                "Mitigate RMF findings and document compensating controls before deployment."
            )
        return risks

    def generate_certificate_report(self) -> Dict[str, Any]:
        """Return a structured assurance report for downstream reporting."""
        report = {
            "timestamp": _utcnow().isoformat(),
            "quality_check": dict(self.status.quality_characteristics),
            "ISO25010": self.status.iso25010_pass,
            "ASIL_Level": self.status.iso26262_asil,
            "UL4600_Safety_Case": self.safety_case.export(),
            "NIST_RMF_Risks": list(self.status.nist_rmf_risks),
            "recommendations": list(dict.fromkeys(self.status.recommendations)),
        }
        return report

    def reset(self) -> None:
        """Reset the auditor state for a fresh evaluation cycle."""
        self.safety_case = SafetyCase(metadata={"framework": "UL4600-inspired"})
        self.status = CertificationStatus()
        logger.info("Certification Auditor state reset")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)



def _coerce_iso8601_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceValidationError("Evidence timestamp must be a non-empty ISO-8601 string.")

    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise EvidenceValidationError(
            f"Evidence timestamp is not valid ISO-8601: {value}"
        ) from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)



def _coerce_non_negative_float(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return number



def _coerce_probability(value: Any, field_name: str) -> float:
    number = _coerce_non_negative_float(value, field_name)
    if number > 1:
        raise ValueError(f"{field_name} must be between 0 and 1 inclusive.")
    return number


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== Running Certification Framework ===\n")

    try:
        auditor = CertificationAuditor()

        iso_metrics = {
            "mtbf": 1200,
            "response_time": 0.8,
            "tech_debt": 0.15,
            "vuln_count": 2,
        }
        printer.pretty("ISO 25010 checks:", auditor.assess_iso25010(iso_metrics), "success")
        printer.pretty("ASIL Level:", auditor.evaluate_asil(0.96, 12000), "success")
        printer.pretty(
            "UL 4600:",
            auditor.finalize_ul4600(
                [
                    {"name": "Unit test log", "artifact_id": "UT-001"},
                    {"name": "Simulation results", "artifact_id": "SIM-042"},
                ],
                metadata={"system": "SLAI"},
            ),
            "success",
        )
        printer.pretty(
            "NIST RMF:",
            auditor.integrate_nist_rmf(
                {"distribution_shift": 0.25, "fairness_score": 0.75, "residual_risk": 0.15}
            ),
            "success",
        )
        printer.pretty("Final Report:", auditor.generate_certificate_report(), "success")

        print("\n* * * * * Phase 2 - Evidence Submission * * * * *\n")
        manager = CertificationManager()
        manager.submit_evidence(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": ["FTA report", "FMEA records"],
                "content": {
                    "summary": "Fail-operational architecture assessed across 10k scenarios",
                    "result": "pass",
                },
                "source": "simulation_pipeline",
                "metadata": {"campaign_id": "SCENARIO-10K"},
            }
        )
        manager.submit_evidence(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": ["Sensor logs", "Failure recovery reports"],
                "content": {
                    "summary": "Redundancy validation completed",
                    "result": "pass",
                },
                "source": "integration_testbed",
            }
        )

        print("\n* * * * * Phase 3 - Certification Check * * * * *\n")
        passed, failures = manager.evaluate_certification()
        detailed_report = manager.evaluate_certification_detailed()

        logger.info("%s", manager)
        print(f"Certification Status: {'PASSED' if passed else 'FAILED'}")
        print(f"Unmet Requirements: {failures}")
        printer.pretty("Detailed Report:", detailed_report.to_dict(), "success")
        printer.pretty("Certificate:", manager.generate_certificate(), "success")
        print("\n=== Successfully Ran Certification Framework ===\n")

    except CertificationFrameworkError as exc:
        logger.error("Certification workflow failed: %s", exc)
        raise
