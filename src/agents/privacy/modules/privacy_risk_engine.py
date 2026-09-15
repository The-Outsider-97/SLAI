"""Evidence-weighted privacy risk aggregation for SLAI.

The ``PrivacyRiskEngine`` combines already-produced privacy evidence into one
explainable residual-risk assessment.  It is intentionally not a generic safety
risk engine and it does not make legal/compliance determinations.  Its inputs
are the stage results owned by the Privacy subsystem (identification, consent,
minimization, residual exposure, retention, and lineage/flow analysis).

Design principles
-----------------
* deterministic and configuration-driven;
* no raw payload inspection;
* no model training or tuning at runtime;
* hard privacy decisions can only be preserved or made more restrictive;
* every numerical score is decomposed into auditable factors;
* incomplete evidence reduces confidence and can trigger escalation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.privacy_error import *
from ..utils.privacy_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Privacy Risk Engine")
printer = PrettyPrinter()

MODULE_VERSION = "2.3.0"
ASSESSMENT_SCHEMA_VERSION = "privacy.risk_assessment.v1"


@dataclass(frozen=True, slots=True)
class PrivacyRiskFactor:
    """One explainable contributor to the aggregate privacy risk score."""

    name: str
    source: str
    raw_score: float
    weight: float
    contribution: float
    rationale: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["evidence"] = sanitize_privacy_mapping(data.get("evidence"), enabled=True)
        return data


@dataclass(frozen=True, slots=True)
class PrivacyRiskAssessment:
    """Final deterministic privacy risk assessment."""

    request_id: Optional[str]
    decision: str
    score_decision: str
    authoritative_decision: str
    stage: str
    overall_risk_score: float
    confidence: float
    evidence_completeness: float
    missing_evidence_sources: Tuple[str, ...]
    factors: Tuple[PrivacyRiskFactor, ...]
    recommended_controls: Tuple[str, ...]
    rationale: str
    policy_fingerprint: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["factors"] = [factor.to_dict() for factor in self.factors]
        return data


class PrivacyRiskEngine:
    """Aggregate privacy-domain evidence without duplicating subsystem policy."""

    CONFIG_SECTION = "privacy_risk_engine"

    def __init__(self, *, config: Optional[Mapping[str, Any]] = None) -> None:
        root_config = dict(config) if config is not None else load_global_config()
        self.config = get_config_section(self.CONFIG_SECTION, config=root_config)
        if not self.config:
            raise PrivacyConfigurationError(
                section=self.CONFIG_SECTION,
                details=(
                    "Missing privacy_risk_engine configuration. Add the documented "
                    "section to src/agents/privacy/configs/privacy_config.yaml."
                ),
            )

        self.enabled = self._require_bool("enabled")
        self.strict_mode = self._require_bool("strict_mode")
        self.default_stage = self._require_text("default_decision_stage")
        self.risk_can_block = self._require_bool("risk_can_block")
        self.escalate_on_incomplete_evidence = self._require_bool(
            "escalate_on_incomplete_evidence"
        )
        self.max_factors = require_integer(
            self.config.get("max_factors"), "privacy_risk_engine.max_factors", minimum=1
        )
        self.max_evidence_sources = require_integer(
            self.config.get("max_evidence_sources"),
            "privacy_risk_engine.max_evidence_sources",
            minimum=1,
        )
        self.max_recommended_controls = require_integer(
            self.config.get("max_recommended_controls"),
            "privacy_risk_engine.max_recommended_controls",
            minimum=1,
        )

        self.required_evidence_sources = tuple(
            normalize_string_sequence(
                self.config.get("required_evidence_sources"),
                field_name="privacy_risk_engine.required_evidence_sources",
                max_items=self.max_evidence_sources,
                overflow="raise",
                casefold=True,
            )
        )
        if not self.required_evidence_sources:
            raise PrivacyConfigurationError(
                section=self.CONFIG_SECTION,
                details="required_evidence_sources must not be empty.",
            )

        weights = self._require_mapping("component_weights")
        expected_weight_keys = {
            "inherent_sensitivity",
            "residual_exposure",
            "authorization",
            "retention",
            "flow",
            "uncertainty",
            "evidence_gap",
        }
        if set(weights) != expected_weight_keys:
            raise PrivacyConfigurationError(
                section=self.CONFIG_SECTION,
                details=(
                    "component_weights must define exactly: "
                    + ", ".join(sorted(expected_weight_keys))
                ),
            )
        self.component_weights = {
            key: require_probability(
                weights.get(key), f"privacy_risk_engine.component_weights.{key}"
            )
            for key in expected_weight_keys
        }
        self._validate_weights(self.component_weights, "component_weights")

        decision_scores = self._require_mapping("decision_scores")
        self.decision_scores = {}
        for name in ("allow", "modify", "escalate", "block"):
            if name not in decision_scores:
                raise PrivacyConfigurationError(
                    section=self.CONFIG_SECTION,
                    details=f"decision_scores.{name} is required.",
                )
            self.decision_scores[name] = require_probability(
                decision_scores.get(name), f"privacy_risk_engine.decision_scores.{name}"
            )
        if not (
            self.decision_scores["allow"]
            <= self.decision_scores["modify"]
            <= self.decision_scores["escalate"]
            <= self.decision_scores["block"]
        ):
            raise PrivacyConfigurationError(
                section=self.CONFIG_SECTION,
                details="decision_scores must be monotonic: allow <= modify <= escalate <= block.",
            )

        thresholds = self._require_mapping("decision_thresholds")
        self.modify_threshold = require_probability(
            thresholds.get("modify"), "privacy_risk_engine.decision_thresholds.modify"
        )
        self.escalate_threshold = require_probability(
            thresholds.get("escalate"), "privacy_risk_engine.decision_thresholds.escalate"
        )
        self.block_threshold = require_probability(
            thresholds.get("block"), "privacy_risk_engine.decision_thresholds.block"
        )
        if not (
            self.modify_threshold <= self.escalate_threshold <= self.block_threshold
        ):
            raise PrivacyConfigurationError(
                section=self.CONFIG_SECTION,
                details="decision_thresholds must be monotonic: modify <= escalate <= block.",
            )

        confidence = self._require_mapping("confidence_penalties")
        self.uncertainty_confidence_penalty = require_probability(
            confidence.get("uncertainty"),
            "privacy_risk_engine.confidence_penalties.uncertainty",
        )
        self.evidence_gap_confidence_penalty = require_probability(
            confidence.get("evidence_gap"),
            "privacy_risk_engine.confidence_penalties.evidence_gap",
        )
        if self.uncertainty_confidence_penalty + self.evidence_gap_confidence_penalty > 1.0 + 1e-9:
            raise PrivacyConfigurationError(
                section=self.CONFIG_SECTION,
                details="confidence_penalties must sum to <= 1.0.",
            )

        self.authoritative_decision_sources = set(
            normalize_string_sequence(
                self.config.get("authoritative_decision_sources"),
                field_name="privacy_risk_engine.authoritative_decision_sources",
                max_items=self.max_evidence_sources,
                overflow="raise",
                casefold=True,
            )
        )
        if not self.authoritative_decision_sources:
            raise PrivacyConfigurationError(
                section=self.CONFIG_SECTION,
                details="authoritative_decision_sources must not be empty.",
            )
        self.recommended_control_map = self._require_mapping("recommended_controls")

        self.policy_fingerprint = stable_privacy_fingerprint(
            {"section": self.CONFIG_SECTION, "config": self.config}, length=24
        )
        logger.info(
            "PrivacyRiskEngine initialized (schema=%s, policy=%s)",
            ASSESSMENT_SCHEMA_VERSION,
            self.policy_fingerprint,
        )

    # ------------------------------------------------------------------
    # Configuration
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
    def _validate_weights(weights: Mapping[str, float], field_name: str) -> None:
        total = sum(float(v) for v in weights.values())
        if abs(total - 1.0) > 1e-9:
            raise PrivacyConfigurationError(
                section=PrivacyRiskEngine.CONFIG_SECTION,
                details=f"'{field_name}' must sum to exactly 1.0; received {total:.12f}.",
            )

    # ------------------------------------------------------------------
    # Evidence helpers
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
    def _probability(mapping: Mapping[str, Any], *keys: str) -> float:
        value: Any = None
        for key in keys:
            if key in mapping:
                value = mapping.get(key)
                break
        if value is None:
            return 0.0
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        return max(0.0, min(1.0, numeric))

    def _decision_score(self, mapping: Mapping[str, Any]) -> float:
        raw = mapping.get("decision")
        if raw is None:
            return 0.0
        try:
            normalized = decision_value(raw)
        except ValueError:
            return 0.0
        return self.decision_scores[normalized]

    def _source_decision(self, source: str, mapping: Mapping[str, Any]) -> Optional[str]:
        if source.casefold() not in self.authoritative_decision_sources:
            return None
        raw = mapping.get("decision")
        if raw is None:
            return None
        try:
            return decision_value(raw)
        except ValueError:
            return None

    def _control_recommendations(
        self,
        *,
        final_decision: str,
        residual: Mapping[str, Any],
        flow: Mapping[str, Any],
        missing_sources: Sequence[str],
    ) -> Tuple[str, ...]:
        keys: List[str] = [final_decision]
        if residual.get("under_redaction"):
            keys.append("under_redaction")
        if residual.get("over_redaction"):
            keys.append("over_redaction")
        if flow.get("unauthorized_destinations"):
            keys.append("unauthorized_destination")
        if int(flow.get("post_deletion_processing_count") or 0) > 0:
            keys.append("post_deletion_processing")
        if missing_sources:
            keys.append("incomplete_evidence")

        controls: List[str] = []
        seen = set()
        for key in keys:
            raw = self.recommended_control_map.get(key, [])
            if isinstance(raw, str):
                raw = [raw]
            if not isinstance(raw, Sequence):
                continue
            for item in raw:
                text = str(item).strip()
                if text and text not in seen:
                    controls.append(text)
                    seen.add(text)
                    if len(controls) >= self.max_recommended_controls:
                        return tuple(controls)
        return tuple(controls)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def assess(
        self,
        *,
        identification: Optional[Mapping[str, Any]] = None,
        consent: Optional[Mapping[str, Any]] = None,
        minimization: Optional[Mapping[str, Any]] = None,
        residual_exposure: Optional[Mapping[str, Any]] = None,
        retention: Optional[Mapping[str, Any]] = None,
        flow_analysis: Optional[Mapping[str, Any]] = None,
        request_id: Optional[str] = None,
        stage: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Aggregate the available privacy evidence into one risk assessment."""

        if not self.enabled:
            result = PrivacyRiskAssessment(
                request_id=request_id,
                decision=PrivacyDecision.ALLOW.value,
                score_decision=PrivacyDecision.ALLOW.value,
                authoritative_decision=PrivacyDecision.ALLOW.value,
                stage=stage or self.default_stage,
                overall_risk_score=0.0,
                confidence=1.0,
                evidence_completeness=1.0,
                missing_evidence_sources=(),
                factors=(),
                recommended_controls=(),
                rationale="Privacy risk aggregation is disabled by configuration.",
                policy_fingerprint=self.policy_fingerprint,
                created_at=utc_iso(),
            )
            return result.to_dict()

        try:
            evidence: Dict[str, Dict[str, Any]] = {
                "identification": self._mapping(identification, "identification"),
                "consent": self._mapping(consent, "consent"),
                "minimization": self._mapping(minimization, "minimization"),
                "residual_exposure": self._mapping(residual_exposure, "residual_exposure"),
                "retention": self._mapping(retention, "retention"),
                "flow": self._mapping(flow_analysis, "flow_analysis"),
            }
            safe_context = sanitize_privacy_mapping(context, enabled=True)

            missing_sources = tuple(
                source
                for source in self.required_evidence_sources
                if not evidence.get(source)
            )
            evidence_completeness = (
                1.0
                - (len(missing_sources) / float(len(self.required_evidence_sources)))
                if self.required_evidence_sources
                else 1.0
            )
            evidence_completeness = max(0.0, min(1.0, evidence_completeness))
            evidence_gap = 1.0 - evidence_completeness

            identification_map = evidence["identification"]
            consent_map = evidence["consent"]
            residual_map = evidence["residual_exposure"]
            retention_map = evidence["retention"]
            flow_map = evidence["flow"]

            inherent_sensitivity = self._probability(
                identification_map, "sensitivity_score"
            )
            residual_score = self._probability(
                residual_map, "residual_risk_score", "post_sensitivity"
            )
            authorization_score = self._decision_score(consent_map)
            retention_score = self._decision_score(retention_map)
            flow_score = self._probability(flow_map, "flow_risk_score")
            uncertainty_score = self._probability(
                identification_map, "uncertainty_score"
            )

            factor_specs = [
                (
                    "inherent_sensitivity",
                    "identification",
                    inherent_sensitivity,
                    "Sensitivity of the data before privacy controls.",
                    {"sensitivity_score": inherent_sensitivity},
                ),
                (
                    "residual_exposure",
                    "residual_exposure",
                    residual_score,
                    "Privacy exposure remaining after minimization and re-identification scan.",
                    {
                        "post_sensitivity": residual_map.get("post_sensitivity"),
                        "under_redaction": bool(residual_map.get("under_redaction")),
                    },
                ),
                (
                    "authorization",
                    "consent",
                    authorization_score,
                    "Consent/purpose stage decision converted to a configured risk score.",
                    {"decision": consent_map.get("decision")},
                ),
                (
                    "retention",
                    "retention",
                    retention_score,
                    "Retention/deletion stage decision converted to a configured risk score.",
                    {
                        "decision": retention_map.get("decision"),
                        "status": retention_map.get("status"),
                    },
                ),
                (
                    "flow",
                    "flow",
                    flow_score,
                    "Cumulative privacy dissemination and lineage risk.",
                    {
                        "flow_risk_score": flow_map.get("flow_risk_score"),
                        "unauthorized_destinations": flow_map.get("unauthorized_destinations"),
                    },
                ),
                (
                    "uncertainty",
                    "identification",
                    uncertainty_score,
                    "Classifier uncertainty in sensitive-data identification.",
                    {"uncertainty_score": uncertainty_score},
                ),
                (
                    "evidence_gap",
                    "risk_engine",
                    evidence_gap,
                    "Penalty for required privacy evidence not available to the aggregator.",
                    {"missing_sources": list(missing_sources)},
                ),
            ]

            factors: List[PrivacyRiskFactor] = []
            overall_risk_score = 0.0
            for name, source, raw_score, rationale, factor_evidence in factor_specs:
                weight = self.component_weights[name]
                contribution = raw_score * weight
                overall_risk_score += contribution
                if len(factors) < self.max_factors:
                    factors.append(
                        PrivacyRiskFactor(
                            name=name,
                            source=source,
                            raw_score=round(raw_score, 6),
                            weight=round(weight, 6),
                            contribution=round(contribution, 6),
                            rationale=rationale,
                            evidence=sanitize_privacy_mapping(factor_evidence, enabled=True),
                        )
                    )
            overall_risk_score = round(max(0.0, min(1.0, overall_risk_score)), 6)

            score_decision = PrivacyDecision.ALLOW.value
            if overall_risk_score >= self.block_threshold:
                score_decision = (
                    PrivacyDecision.BLOCK.value
                    if self.risk_can_block
                    else PrivacyDecision.ESCALATE.value
                )
            elif overall_risk_score >= self.escalate_threshold:
                score_decision = PrivacyDecision.ESCALATE.value
            elif overall_risk_score >= self.modify_threshold:
                score_decision = PrivacyDecision.MODIFY.value

            source_decisions: List[str] = []
            for source_name, mapping in evidence.items():
                normalized = self._source_decision(source_name, mapping)
                if normalized is not None:
                    source_decisions.append(normalized)
            authoritative_decision = combine_privacy_decisions(
                *source_decisions, default=PrivacyDecision.ALLOW.value
            )

            evidence_decision = PrivacyDecision.ALLOW.value
            if missing_sources and self.escalate_on_incomplete_evidence:
                evidence_decision = PrivacyDecision.ESCALATE.value

            final_decision = combine_privacy_decisions(
                authoritative_decision,
                score_decision,
                evidence_decision,
            )

            confidence = 1.0
            confidence -= uncertainty_score * self.uncertainty_confidence_penalty
            confidence -= evidence_gap * self.evidence_gap_confidence_penalty
            confidence = round(max(0.0, min(1.0, confidence)), 6)

            controls = self._control_recommendations(
                final_decision=final_decision,
                residual=residual_map,
                flow=flow_map,
                missing_sources=missing_sources,
            )

            rationale = (
                "Privacy evidence is within configured risk bounds."
                if final_decision == PrivacyDecision.ALLOW.value
                else (
                    "Privacy evidence requires modification, escalation, or blocking; "
                    "the aggregate decision preserves the most restrictive authoritative stage decision."
                )
            )

            result = PrivacyRiskAssessment(
                request_id=request_id or identification_map.get("request_id"),
                decision=decision_value(final_decision),
                score_decision=decision_value(score_decision),
                authoritative_decision=decision_value(authoritative_decision),
                stage=stage or self.default_stage,
                overall_risk_score=overall_risk_score,
                confidence=confidence,
                evidence_completeness=round(evidence_completeness, 6),
                missing_evidence_sources=missing_sources,
                factors=tuple(factors),
                recommended_controls=controls,
                rationale=rationale,
                policy_fingerprint=self.policy_fingerprint,
                created_at=utc_iso(),
            )
            output = result.to_dict()
            output["schema_version"] = ASSESSMENT_SCHEMA_VERSION
            output["stage_decisions"] = {
                source: mapping.get("decision")
                for source, mapping in evidence.items()
                if mapping.get("decision") is not None
            }
            if safe_context:
                output["context"] = safe_context
            return output
        except Exception as exc:
            if isinstance(exc, PrivacyError):
                raise
            normalized = normalize_privacy_exception(
                exc,
                stage="privacy_risk_engine.assess",
                context={"request_id": request_id, "stage": stage or self.default_stage},
            )
            if self.strict_mode:
                raise normalized from exc
            return {
                "schema_version": ASSESSMENT_SCHEMA_VERSION,
                "request_id": request_id,
                "decision": PrivacyDecision.ESCALATE.value,
                "score_decision": PrivacyDecision.ESCALATE.value,
                "authoritative_decision": PrivacyDecision.ESCALATE.value,
                "stage": stage or self.default_stage,
                "overall_risk_score": 1.0,
                "confidence": 0.0,
                "evidence_completeness": 0.0,
                "missing_evidence_sources": list(self.required_evidence_sources),
                "factors": [],
                "recommended_controls": [],
                "rationale": "Privacy risk aggregation failed; manual review is required.",
                "policy_fingerprint": self.policy_fingerprint,
                "created_at": utc_iso(),
                "error": {
                    "error_type": str(getattr(getattr(normalized, "error_type", None), "value", "policy_evaluation_failed")),
                    "error_code": str(getattr(normalized, "error_code", "PRV-3025")),
                    "severity": str(getattr(getattr(normalized, "severity", None), "value", "high")),
                    "retryable": bool(getattr(normalized, "retryable", True)),
                    "message": str(getattr(normalized, "message", "Privacy risk aggregation failed.")),
                },
            }


__all__ = [
    "MODULE_VERSION",
    "ASSESSMENT_SCHEMA_VERSION",
    "PrivacyRiskFactor",
    "PrivacyRiskAssessment",
    "PrivacyRiskEngine",
]
