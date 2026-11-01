"""Residual exposure assessment for SLAI's Privacy subsystem.

This module verifies whether a privacy transformation materially reduced
sensitive-data exposure.  It deliberately does *not* perform PII/PHI detection,
redaction, consent validation, retention enforcement, or persistence.  Instead
it consumes structured evidence produced by ``DataID`` and
``DataMinimization`` and derives an auditable post-control assessment.

Architectural boundary
----------------------
The module is intentionally evidence-driven:

* ``DataID`` owns entity/sensitive-attribute detection and sensitivity scoring.
* ``DataMinimization`` owns keep/drop/mask/tokenize/hash decisions.
* ``ResidualExposureAnalyzer`` compares the before/after evidence and checks
  whether declared transformations achieved the intended privacy effect.

This keeps the module independent from the orchestrator and prevents circular
imports with ``privacy_agent.py`` or the domain components.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


from ..utils.config_loader import get_config_section, load_global_config
from ..utils.privacy_error import *
from ..utils.privacy_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]


logger = get_logger("Privacy Residual Exposure")
printer = PrettyPrinter()

MODULE_VERSION = "2.3.0"
ASSESSMENT_SCHEMA_VERSION = "privacy.residual_exposure.v1"


@dataclass(frozen=True, slots=True)
class ExposureFinding:
    """Audit-safe finding emitted by the residual exposure analyzer."""

    code: str
    severity: str
    message: str
    entity_type: Optional[str] = None
    path: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["evidence"] = sanitize_privacy_mapping(data.get("evidence"), enabled=True)
        return data


@dataclass(frozen=True, slots=True)
class ResidualExposureAssessment:
    """Structured result of a before/after privacy transformation comparison."""

    request_id: Optional[str]
    decision: str
    status: str
    stage: str
    pre_sensitivity: float
    post_sensitivity: float
    exposure_reduction: float
    pre_entity_count: int
    post_entity_count: int
    critical_residual_count: int
    high_residual_count: int
    direct_identifier_residual_count: int
    transformation_coverage: float
    required_field_preservation: float
    removal_ratio: float
    under_redaction: bool
    over_redaction: bool
    aggressive_reduction: bool
    scan_completeness_known: bool
    scan_complete: Optional[bool]
    residual_risk_score: float
    findings: Tuple[ExposureFinding, ...]
    rationale: str
    policy_fingerprint: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["findings"] = [item.to_dict() for item in self.findings]
        return data


class ResidualExposureAnalyzer:
    """Compare pre/post identification evidence around a minimization stage.

    The analyzer does not inspect the raw payload.  It relies on the privacy
    subsystem's structured outputs, which keeps the module auditable and avoids
    creating a second classifier or redaction path.
    """

    CONFIG_SECTION = "residual_exposure"

    def __init__(self, *, config: Optional[Mapping[str, Any]] = None) -> None:
        root_config = dict(config) if config is not None else load_global_config()
        self.config = get_config_section(self.CONFIG_SECTION, config=root_config)
        if not self.config:
            raise PrivacyConfigurationError(
                section=self.CONFIG_SECTION,
                details=(
                    "Missing residual_exposure configuration. Add the documented "
                    "section to src/agents/privacy/configs/privacy_config.yaml."
                ),
            )

        self.enabled = self._require_bool("enabled")
        self.strict_mode = self._require_bool("strict_mode")
        self.default_stage = self._require_text("default_decision_stage")
        self.require_explicit_scan_completeness = self._require_bool(
            "require_explicit_scan_completeness"
        )
        self.block_on_critical_residual = self._require_bool("block_on_critical_residual")
        self.block_on_incomplete_post_scan = self._require_bool("block_on_incomplete_post_scan")
        self.escalate_on_high_residual = self._require_bool("escalate_on_high_residual")
        self.escalate_on_under_redaction = self._require_bool("escalate_on_under_redaction")
        self.modify_on_over_redaction = self._require_bool("modify_on_over_redaction")
        self.max_findings = require_integer(
            self.config.get("max_findings"), "residual_exposure.max_findings", minimum=1
        )
        self.max_required_fields = require_integer(
            self.config.get("max_required_fields"),
            "residual_exposure.max_required_fields",
            minimum=1,
        )
        self.high_residual_saturation_count = require_integer(
            self.config.get("high_residual_saturation_count"),
            "residual_exposure.high_residual_saturation_count",
            minimum=1,
        )

        thresholds = self._require_mapping("thresholds")
        self.max_post_sensitivity_allow = require_probability(
            thresholds.get("max_post_sensitivity_allow"),
            "residual_exposure.thresholds.max_post_sensitivity_allow",
        )
        self.min_exposure_reduction = require_probability(
            thresholds.get("min_exposure_reduction"),
            "residual_exposure.thresholds.min_exposure_reduction",
        )
        self.min_transformation_coverage = require_probability(
            thresholds.get("min_transformation_coverage"),
            "residual_exposure.thresholds.min_transformation_coverage",
        )
        self.min_required_field_preservation = require_probability(
            thresholds.get("min_required_field_preservation"),
            "residual_exposure.thresholds.min_required_field_preservation",
        )
        self.max_removal_ratio = require_probability(
            thresholds.get("max_removal_ratio"),
            "residual_exposure.thresholds.max_removal_ratio",
        )

        scoring = self._require_mapping("risk_weights")
        self.risk_weights = {
            "post_sensitivity": require_probability(
                scoring.get("post_sensitivity"),
                "residual_exposure.risk_weights.post_sensitivity",
            ),
            "critical_residual": require_probability(
                scoring.get("critical_residual"),
                "residual_exposure.risk_weights.critical_residual",
            ),
            "high_residual": require_probability(
                scoring.get("high_residual"),
                "residual_exposure.risk_weights.high_residual",
            ),
            "coverage_gap": require_probability(
                scoring.get("coverage_gap"),
                "residual_exposure.risk_weights.coverage_gap",
            ),
            "scan_uncertainty": require_probability(
                scoring.get("scan_uncertainty"),
                "residual_exposure.risk_weights.scan_uncertainty",
            ),
        }
        self._validate_weights(self.risk_weights, field_name="risk_weights")

        self.critical_severities = set(
            normalize_string_sequence(
                self.config.get("critical_severities"),
                field_name="residual_exposure.critical_severities",
                casefold=True,
            )
        )
        self.high_severities = set(
            normalize_string_sequence(
                self.config.get("high_severities"),
                field_name="residual_exposure.high_severities",
                casefold=True,
            )
        )
        self.direct_identifier_categories = set(
            normalize_string_sequence(
                self.config.get("direct_identifier_categories"),
                field_name="residual_exposure.direct_identifier_categories",
                casefold=True,
            )
        )
        self.coverage_severities = set(
            normalize_string_sequence(
                self.config.get("coverage_severities"),
                field_name="residual_exposure.coverage_severities",
                casefold=True,
            )
        )

        if not self.critical_severities:
            raise PrivacyConfigurationError(
                section=self.CONFIG_SECTION,
                details="critical_severities must contain at least one severity label.",
            )
        if not self.high_severities:
            raise PrivacyConfigurationError(
                section=self.CONFIG_SECTION,
                details="high_severities must contain at least one severity label.",
            )

        self.policy_fingerprint = stable_privacy_fingerprint(
            {"section": self.CONFIG_SECTION, "config": self.config}, length=24
        )
        logger.info(
            "ResidualExposureAnalyzer initialized (schema=%s, policy=%s)",
            ASSESSMENT_SCHEMA_VERSION,
            self.policy_fingerprint,
        )

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------
    def _require_bool(self, key: str) -> bool:
        if key not in self.config or not isinstance(self.config[key], bool):
            raise PrivacyConfigurationError(
                section=self.CONFIG_SECTION,
                details=f"'{key}' must be explicitly configured as a boolean.",
            )
        return bool(self.config[key])

    def _require_text(self, key: str) -> str:
        value = str(self.config.get(key) or "").strip()
        if not value:
            raise PrivacyConfigurationError(
                section=self.CONFIG_SECTION,
                details=f"'{key}' must be a non-empty string.",
            )
        return value

    def _require_mapping(self, key: str) -> Dict[str, Any]:
        value = self.config.get(key)
        if not isinstance(value, Mapping):
            raise PrivacyConfigurationError(
                section=self.CONFIG_SECTION,
                details=f"'{key}' must be a mapping.",
            )
        return dict(value)

    @staticmethod
    def _validate_weights(weights: Mapping[str, float], *, field_name: str) -> None:
        total = sum(float(v) for v in weights.values())
        if abs(total - 1.0) > 1e-9:
            raise PrivacyConfigurationError(
                section=ResidualExposureAnalyzer.CONFIG_SECTION,
                details=f"'{field_name}' must sum to exactly 1.0; received {total:.12f}.",
            )

    # ------------------------------------------------------------------
    # Evidence normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _mapping(value: Any, field_name: str) -> Dict[str, Any]:
        if value is None:
            return {}
        if hasattr(value, "to_dict") and callable(value.to_dict):
            value = value.to_dict()
        if not isinstance(value, Mapping):
            raise TypeError(f"'{field_name}' must be a mapping or expose to_dict().")
        return normalize_mapping(value, field_name=field_name, deep_copy=True)

    @staticmethod
    def _probability_from(mapping: Mapping[str, Any], key: str) -> float:
        value = mapping.get(key, 0.0)
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        if numeric < 0.0:
            return 0.0
        if numeric > 1.0:
            return 1.0
        return numeric

    @staticmethod
    def _entity_signature(entity: Mapping[str, Any]) -> Tuple[str, str]:
        entity_type = normalize_field_name(entity.get("entity_type") or entity.get("type") or "unknown")
        path = ResidualExposureAnalyzer._entity_path(entity)
        path = normalize_field_name(path.replace(".", "__")).replace("__", ".")
        return entity_type, path

    @staticmethod
    def _entity_path(entity: Mapping[str, Any]) -> str:
        path = strip_path_indices(str(entity.get("path") or "")).strip()
        if path == "root":
            return ""
        if path.startswith("root."):
            return path[5:]
        return path

    @staticmethod
    def _entity_severity(entity: Mapping[str, Any]) -> str:
        return str(entity.get("severity") or "low").strip().lower()

    @staticmethod
    def _entity_category(entity: Mapping[str, Any]) -> str:
        return str(entity.get("category") or "").strip().lower()

    @staticmethod
    def _extract_entities(evidence: Mapping[str, Any]) -> List[Dict[str, Any]]:
        raw = evidence.get("detected_entities") or []
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            return []
        result: List[Dict[str, Any]] = []
        for item in raw:
            if isinstance(item, Mapping):
                result.append(dict(item))
        return result

    @staticmethod
    def _coverage_for_paths(entity_paths: Sequence[str], transformed_paths: Sequence[str]) -> float:
        normalized_entities = [path for path in entity_paths if str(path).strip()]
        if not normalized_entities:
            return 1.0
        covered = 0
        for entity_path in normalized_entities:
            if any(
                pattern_matches_path(transform_path, entity_path)
                or pattern_matches_path(entity_path, transform_path)
                for transform_path in transformed_paths
                if str(transform_path).strip()
            ):
                covered += 1
        return covered / float(len(normalized_entities))

    @staticmethod
    def _scan_state(evidence: Mapping[str, Any]) -> Tuple[bool, Optional[bool]]:
        if "scan_complete" not in evidence:
            return False, None
        return True, bool(evidence.get("scan_complete"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def assess(
        self,
        *,
        before_identification: Mapping[str, Any],
        after_identification: Mapping[str, Any],
        minimization: Mapping[str, Any],
        request_id: Optional[str] = None,
        required_fields: Optional[Sequence[str]] = None,
        stage: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Assess post-minimization residual exposure.

        Parameters are structured evidence only; raw payload values are neither
        required nor persisted by this analyzer.
        """

        if not self.enabled:
            result = ResidualExposureAssessment(
                request_id=request_id,
                decision=PrivacyDecision.ALLOW.value,
                status="disabled",
                stage=stage or self.default_stage,
                pre_sensitivity=0.0,
                post_sensitivity=0.0,
                exposure_reduction=0.0,
                pre_entity_count=0,
                post_entity_count=0,
                critical_residual_count=0,
                high_residual_count=0,
                direct_identifier_residual_count=0,
                transformation_coverage=1.0,
                required_field_preservation=1.0,
                removal_ratio=0.0,
                under_redaction=False,
                over_redaction=False,
                aggressive_reduction=False,
                scan_completeness_known=False,
                scan_complete=None,
                residual_risk_score=0.0,
                findings=(),
                rationale="Residual exposure analysis is disabled by configuration.",
                policy_fingerprint=self.policy_fingerprint,
                created_at=utc_iso(),
            )
            return result.to_dict()

        try:
            before = self._mapping(before_identification, "before_identification")
            after = self._mapping(after_identification, "after_identification")
            minimized = self._mapping(minimization, "minimization")
            safe_context = sanitize_privacy_mapping(context, enabled=True)

            before_entities = self._extract_entities(before)
            after_entities = self._extract_entities(after)
            before_signatures = {self._entity_signature(item) for item in before_entities}
            after_signatures = {self._entity_signature(item) for item in after_entities}
            persistent_signatures = before_signatures & after_signatures

            pre_sensitivity = self._probability_from(before, "sensitivity_score")
            post_sensitivity = self._probability_from(after, "sensitivity_score")
            if pre_sensitivity <= 0.0:
                exposure_reduction = 1.0 if post_sensitivity <= 0.0 else 0.0
            else:
                exposure_reduction = max(0.0, min(1.0, 1.0 - (post_sensitivity / pre_sensitivity)))

            critical_residual = [
                entity for entity in after_entities
                if self._entity_severity(entity) in self.critical_severities
            ]
            high_residual = [
                entity for entity in after_entities
                if self._entity_severity(entity) in self.high_severities
            ]
            direct_identifier_residual = [
                entity for entity in after_entities
                if self._entity_category(entity) in self.direct_identifier_categories
            ]

            coverage_entities = [
                entity for entity in before_entities
                if self._entity_severity(entity) in self.coverage_severities
            ]
            coverage_paths = [self._entity_path(entity) for entity in coverage_entities]
            transformed_paths = [
                *[str(v) for v in minimized.get("masked_fields") or []],
                *[str(v) for v in minimized.get("removed_fields") or []],
            ]
            transformation_coverage = self._coverage_for_paths(coverage_paths, transformed_paths)

            normalized_required = normalize_string_sequence(
                required_fields,
                field_name="required_fields",
                max_items=self.max_required_fields,
                overflow="raise",
            )
            missing_required = [str(v) for v in minimized.get("missing_required_fields") or []]
            if normalized_required:
                preserved = max(0, len(normalized_required) - len(set(missing_required)))
                required_field_preservation = preserved / float(len(normalized_required))
            else:
                required_field_preservation = 1.0 if not missing_required else 0.0

            original_count = max(int(minimized.get("original_field_count") or 0), 0)
            removed_count = max(int(minimized.get("removed_field_count") or len(minimized.get("removed_fields") or [])), 0)
            removal_ratio = (removed_count / float(original_count)) if original_count > 0 else 0.0
            removal_ratio = max(0.0, min(1.0, removal_ratio))

            pre_known, pre_complete = self._scan_state(before)
            post_known, post_complete = self._scan_state(after)
            scan_completeness_known = pre_known and post_known
            scan_complete: Optional[bool]
            if scan_completeness_known:
                scan_complete = bool(pre_complete and post_complete)
            else:
                scan_complete = None

            findings: List[ExposureFinding] = []

            def add_finding(
                code: str,
                severity: str,
                message: str,
                *,
                entity_type: Optional[str] = None,
                path: Optional[str] = None,
                evidence: Optional[Mapping[str, Any]] = None,
            ) -> None:
                if len(findings) >= self.max_findings:
                    return
                findings.append(
                    ExposureFinding(
                        code=code,
                        severity=severity,
                        message=message,
                        entity_type=entity_type,
                        path=path,
                        evidence=sanitize_privacy_mapping(evidence, enabled=True),
                    )
                )

            for entity in critical_residual:
                add_finding(
                    "critical_residual_entity",
                    "critical",
                    "A critical sensitive entity remains detectable after minimization.",
                    entity_type=str(entity.get("entity_type") or "unknown"),
                    path=self._entity_path(entity) or None,
                    evidence={"severity": self._entity_severity(entity)},
                )

            if persistent_signatures:
                for entity_type, path in sorted(persistent_signatures):
                    add_finding(
                        "persistent_sensitive_signature",
                        "high",
                        "A sensitive entity signature is present before and after minimization.",
                        entity_type=entity_type,
                        path=path or None,
                    )

            if transformation_coverage < self.min_transformation_coverage and coverage_entities:
                add_finding(
                    "transformation_coverage_below_threshold",
                    "high",
                    "Configured transformation coverage for sensitive evidence is below threshold.",
                    evidence={
                        "coverage": round(transformation_coverage, 6),
                        "required_minimum": self.min_transformation_coverage,
                    },
                )

            if pre_sensitivity > 0 and exposure_reduction < self.min_exposure_reduction:
                add_finding(
                    "insufficient_exposure_reduction",
                    "high",
                    "Post-control sensitivity reduction is below the configured minimum.",
                    evidence={
                        "reduction": round(exposure_reduction, 6),
                        "required_minimum": self.min_exposure_reduction,
                    },
                )

            over_redaction = bool(missing_required) or (
                required_field_preservation < self.min_required_field_preservation
            )
            aggressive_reduction = removal_ratio > self.max_removal_ratio
            if over_redaction:
                add_finding(
                    "required_data_lost",
                    "medium",
                    "Minimization removed or failed to preserve declared required fields.",
                    evidence={
                        "missing_required_fields": missing_required,
                        "required_field_preservation": round(required_field_preservation, 6),
                    },
                )
            elif aggressive_reduction:
                add_finding(
                    "aggressive_data_reduction",
                    "low",
                    "A large fraction of fields was removed; review utility if this was not expected.",
                    evidence={"removal_ratio": round(removal_ratio, 6)},
                )

            if not scan_completeness_known:
                add_finding(
                    "scan_completeness_unknown",
                    "medium" if self.require_explicit_scan_completeness else "low",
                    "Identification evidence does not explicitly report scan completeness.",
                )
            elif scan_complete is False:
                add_finding(
                    "incomplete_privacy_scan",
                    "critical" if self.block_on_incomplete_post_scan else "high",
                    "At least one privacy identification scan was incomplete.",
                )

            under_redaction = bool(
                critical_residual
                or persistent_signatures
                or (
                    coverage_entities
                    and transformation_coverage < self.min_transformation_coverage
                )
                or (
                    pre_sensitivity > 0
                    and exposure_reduction < self.min_exposure_reduction
                    and post_sensitivity > self.max_post_sensitivity_allow
                )
            )

            coverage_gap = 1.0 - transformation_coverage
            scan_uncertainty = 0.0
            if not scan_completeness_known:
                scan_uncertainty = 1.0
            elif scan_complete is False:
                scan_uncertainty = 1.0

            residual_risk_score = (
                self.risk_weights["post_sensitivity"] * post_sensitivity
                + self.risk_weights["critical_residual"] * (1.0 if critical_residual else 0.0)
                + self.risk_weights["high_residual"]
                * min(1.0, len(high_residual) / float(self.high_residual_saturation_count))
                + self.risk_weights["coverage_gap"] * coverage_gap
                + self.risk_weights["scan_uncertainty"] * scan_uncertainty
            )
            residual_risk_score = round(max(0.0, min(1.0, residual_risk_score)), 6)

            decision = PrivacyDecision.ALLOW.value
            if critical_residual and self.block_on_critical_residual:
                decision = PrivacyDecision.BLOCK.value
            elif scan_complete is False and self.block_on_incomplete_post_scan:
                decision = PrivacyDecision.BLOCK.value
            elif under_redaction and self.escalate_on_under_redaction:
                decision = PrivacyDecision.ESCALATE.value
            elif high_residual and self.escalate_on_high_residual:
                decision = PrivacyDecision.ESCALATE.value
            elif post_sensitivity > self.max_post_sensitivity_allow:
                decision = PrivacyDecision.MODIFY.value
            elif over_redaction and self.modify_on_over_redaction:
                decision = PrivacyDecision.MODIFY.value

            if self.require_explicit_scan_completeness and not scan_completeness_known:
                decision = combine_privacy_decisions(decision, PrivacyDecision.ESCALATE.value)

            status = "pass" if decision == PrivacyDecision.ALLOW.value else "review"
            if decision == PrivacyDecision.BLOCK.value:
                status = "fail"

            rationale = (
                "Residual exposure is within configured limits."
                if decision == PrivacyDecision.ALLOW.value
                else "Residual exposure requires privacy intervention based on post-control evidence."
            )

            result = ResidualExposureAssessment(
                request_id=request_id or before.get("request_id") or after.get("request_id"),
                decision=decision_value(decision),
                status=status,
                stage=stage or self.default_stage,
                pre_sensitivity=round(pre_sensitivity, 6),
                post_sensitivity=round(post_sensitivity, 6),
                exposure_reduction=round(exposure_reduction, 6),
                pre_entity_count=len(before_entities),
                post_entity_count=len(after_entities),
                critical_residual_count=len(critical_residual),
                high_residual_count=len(high_residual),
                direct_identifier_residual_count=len(direct_identifier_residual),
                transformation_coverage=round(transformation_coverage, 6),
                required_field_preservation=round(required_field_preservation, 6),
                removal_ratio=round(removal_ratio, 6),
                under_redaction=under_redaction,
                over_redaction=over_redaction,
                aggressive_reduction=aggressive_reduction,
                scan_completeness_known=scan_completeness_known,
                scan_complete=scan_complete,
                residual_risk_score=residual_risk_score,
                findings=tuple(findings),
                rationale=rationale,
                policy_fingerprint=self.policy_fingerprint,
                created_at=utc_iso(),
            )
            output = result.to_dict()
            output["schema_version"] = ASSESSMENT_SCHEMA_VERSION
            if safe_context:
                output["context"] = safe_context
            return output
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise
            normalized = normalize_privacy_exception(
                exc,
                stage="residual_exposure.assess",
                context={"request_id": request_id, "stage": stage or self.default_stage},
            )
            if self.strict_mode:
                raise normalized from exc
            return {
                "schema_version": ASSESSMENT_SCHEMA_VERSION,
                "request_id": request_id,
                "decision": PrivacyDecision.ESCALATE.value,
                "status": "error",
                "stage": stage or self.default_stage,
                "policy_fingerprint": self.policy_fingerprint,
                "created_at": utc_iso(),
                "error": {
                    "error_type": str(getattr(getattr(normalized, "error_type", None), "value", "policy_evaluation_failed")),
                    "error_code": str(getattr(normalized, "error_code", "PRV-3025")),
                    "severity": str(getattr(getattr(normalized, "severity", None), "value", "high")),
                    "retryable": bool(getattr(normalized, "retryable", True)),
                    "message": str(getattr(normalized, "message", "Residual exposure evaluation failed.")),
                },
            }


__all__ = [
    "MODULE_VERSION",
    "ASSESSMENT_SCHEMA_VERSION",
    "ExposureFinding",
    "ResidualExposureAssessment",
    "ResidualExposureAnalyzer",
]
