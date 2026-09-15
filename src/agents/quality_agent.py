"""
The Quality Agent is the agent-level quality gatekeeper between normalized
inputs and downstream SLAI use (knowledge ingestion, training/replay,
inference context, and memory updates).

Interfaces and dependencies
Inputs:
- Reader outputs (documents/files)
- Browser outputs (web content)
- Knowledge ingestion streams
- Training/replay samples

Outputs:
- Quality verdict (pass, warn, block)
- Record-level flags and confidence
- Quarantine queue entries

KPIs
- Bad-record escape rate
- Quarantine precision/recall
- Drift detection latency
- Post-quality-gate incident reduction
- Training stability variance

Failure modes & mitigations
- Overblocking: adaptive thresholds + manual override channel.
- Under-detection: combine rule-based + statistical detectors.
- Source volatility: source-specific reliability scoring and cooldown.
"""

from __future__ import annotations

__version__ = "2.3.0"

import time
import uuid

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .base_agent import BaseAgent
from .base.utils.config_contract import assert_valid_config_contract
from .base.utils.main_config_loader import get_config_section, load_global_config
from .quality import (
    QualityMemory,
    SemanticQuality,
    StatisticalQuality,
    StructuralQuality,
    WorkflowControl,
    BaselineDecision,
    BaselineGovernor,
    EvidenceCalibrator,
    FitnessPolicyDecision,
    FitnessPolicyResolver,
    RelationshipQuality,
)
from .quality.utils.quality_error import *
from .quality.utils.quality_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Quality Agent")
printer = PrettyPrinter()


_SUBSYSTEMS = ("structural", "statistical", "semantic")
_BASELINE_DEPENDENT_CHECKS = {
    "drift",
    "distribution_drift",
    "distribution_shift",
    "relationship_drift",
    "numeric_correlation",
    "missingness_correlation",
}


@dataclass(slots=True)
class SubsystemExecution:
    subsystem: str
    enabled: bool
    verdict: str
    batch_score: float
    findings: List[Dict[str, Any]]
    flags: List[str]
    remediation_actions: List[str]
    quarantine_count: int
    shift_metrics: Dict[str, float]
    context: Dict[str, Any]
    duration_ms: float
    raw_result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QualityAgentDecision:
    decision_id: str
    dataset_id: str
    source_id: str
    batch_id: str
    verdict: str
    batch_score: float
    subsystem_scores: Dict[str, float]
    subsystem_verdicts: Dict[str, str]
    flags: List[str]
    remediation_actions: List[str]
    quarantine_count: int
    subsystem_results: Dict[str, Dict[str, Any]]
    workflow_decision: Dict[str, Any]
    route_records: List[Dict[str, Any]]
    quarantine_entries: List[Dict[str, Any]]
    shared_memory_keys: List[str]
    context: Dict[str, Any]
    created_at: float

    # New intelligence evidence. These are additive fields and preserve the
    # existing output contract above.
    fitness_policy: Dict[str, Any] = field(default_factory=dict)
    baseline_decisions: Dict[str, Any] = field(default_factory=dict)
    relationship_quality: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    policy_enforcement: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["created_at_iso"] = datetime.fromtimestamp(
            self.created_at,
            tz=timezone.utc,
        ).isoformat()
        return payload


class QualityAgent(BaseAgent):
    """Agent-level quality orchestration and fitness-for-use decision layer.

    Ownership boundaries
    --------------------
    QualityAgent owns:
    - agent configuration from agents_config.yaml;
    - context-sensitive fitness-for-use selection;
    - orchestration of existing quality subsystems;
    - baseline-governance coordination;
    - relationship-quality integration;
    - evidence calibration;
    - final agent-level policy enforcement;
    - stable publication of the final quality decision.

    QualityAgent does NOT own:
    - schema/missingness/drift/leakage detector implementation;
    - QualityMemory persistence internals;
    - WorkflowControl quarantine/routing internals;
    - general model/agent evaluation;
    - learning/adaptation or hyperparameter search.

    StructuralQuality, StatisticalQuality, SemanticQuality and WorkflowControl
    retain their subsystem responsibilities and subsystem configuration.
    """

    def __init__(
        self,
        shared_memory: Any,
        agent_factory: Any,
        config: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config,
        )

        self.shared_memory = shared_memory if shared_memory is not None else self.shared_memory
        self.agent_factory = agent_factory

        # Agent configuration source: agents_config.yaml only.
        self.config = load_global_config()
        self.quality_config = get_config_section(
            "quality_agent",
            config=self.config,
        )
        if config is not None:
            if not isinstance(config, Mapping):
                raise DataQualityError(
                    message="QualityAgent runtime config must be a mapping",
                    error_type=QualityErrorType.CONFIGURATION_INVALID,
                    severity=QualitySeverity.HIGH,
                    retryable=False,
                    stage=QualityStage.VALIDATION,
                    domain=QualityDomain.SYSTEM,
                    disposition=QualityDisposition.ESCALATE,
                    context={"config_type": type(config).__name__},
                    remediation=(
                        "Pass QualityAgent config overrides as a mapping. "
                        "Persistent agent configuration belongs in agents_config.yaml."
                    ),
                )
            # Runtime overrides are intentionally shallow, matching the existing
            # BaseAgent/agent convention. A nested section supplied at runtime
            # replaces that complete nested section.
            self.quality_config.update(dict(config))

        assert_valid_config_contract(
            global_config=self.config,
            agent_key="quality_agent",
            agent_config=self.quality_config,
            logger=logger,
            require_global_keys=False,
            require_agent_section=False,
            warn_unknown_global_keys=False,
        )

        self._load_agent_configuration()
        self._validate_runtime_configuration()

        self.handler_bridge = kwargs.get("handler_bridge")
        self.safety_bridge = kwargs.get("safety_bridge")
        self._results_by_batch: Dict[str, Dict[str, Any]] = {}
        self._results_by_source: Dict[str, List[Dict[str, Any]]] = {}

        # One coherent quality-memory instance per QualityAgent. Existing
        # subsystem constructors require the small optional memory= integration
        # patch documented with this file.
        self.quality_memory = QualityMemory()

        self.structural_quality = StructuralQuality()
        self.structural_quality.memory = self.quality_memory

        self.statistical_quality = StatisticalQuality()
        self.statistical_quality.memory = self.quality_memory

        self.semantic_quality = SemanticQuality()
        self.semantic_quality.memory = self.quality_memory

        self.workflow_control = WorkflowControl(
            shared_memory=self.shared_memory,
            handler_bridge=self.handler_bridge,
            safety_bridge=self.safety_bridge,
        )
        self.workflow_control.memory = self.quality_memory

        self.fitness_policy: Optional[FitnessPolicyResolver] = None
        self.baseline_governor: Optional[BaselineGovernor] = None
        self.relationship_quality: Optional[RelationshipQuality] = None
        self.evidence_calibrator: Optional[EvidenceCalibrator] = None
        self._initialize_intelligence_modules()

        self.attach_runtime(
            shared_memory=self.shared_memory,
            handler_bridge=self.handler_bridge,
            safety_bridge=self.safety_bridge,
        )

        logger.info(
            "Quality Agent initialized | enabled=%s | intelligence=%s | "
            "auto_route_via_workflow=%s | subsystem_order=%s",
            self.enabled,
            self.intelligence_enabled,
            self.auto_route_via_workflow,
            self.subsystem_order,
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def _load_agent_configuration(self) -> None:
        cfg = self.quality_config

        self.enabled = bool(cfg.get("enabled", True))
        self.default_window = str(
            cfg.get("default_window", "latest")
        ).strip() or "latest"
        self.stop_on_blocking_structural = bool(
            cfg.get("stop_on_blocking_structural", True)
        )
        self.fail_closed_on_subsystem_error = bool(
            cfg.get("fail_closed_on_subsystem_error", True)
        )
        self.auto_route_via_workflow = bool(
            cfg.get("auto_route_via_workflow", True)
        )
        self.prefer_workflow_verdict = bool(
            cfg.get("prefer_workflow_verdict", True)
        )
        self.allow_workflow_verdict_relaxation = bool(
            cfg.get("allow_workflow_verdict_relaxation", False)
        )
        self.include_workflow_findings_in_summary = bool(
            cfg.get("include_workflow_findings_in_summary", True)
        )
        self.include_record_previews_in_shared_memory = bool(
            cfg.get("include_record_previews_in_shared_memory", False)
        )
        self.max_shared_record_preview = positive_int(
            cfg.get("max_shared_record_preview", 5),
            "quality_agent.max_shared_record_preview",
        )

        self.pass_threshold = bounded_score(
            cfg.get("pass_threshold", 0.90),
            field_name="quality_agent.pass_threshold",
        )
        self.warn_threshold = bounded_score(
            cfg.get("warn_threshold", 0.75),
            field_name="quality_agent.warn_threshold",
        )
        self.subsystem_order = string_list(
            cfg.get(
                "subsystem_order",
                ["structural", "statistical", "semantic"],
            ),
            deduplicate=True,
        )
        self.subsystem_weights = normalize_weights(
            cfg.get(
                "subsystem_weights",
                {
                    "structural": 0.34,
                    "statistical": 0.33,
                    "semantic": 0.33,
                },
            ),
            expected_keys=_SUBSYSTEMS,
            field_name="quality_agent.subsystem_weights",
            reject_unknown=True,
        )

        self.shared_memory_config = self._mapping_section(
            cfg,
            "shared_memory",
            required=False,
        )
        self.publish_to_shared_memory = bool(
            self.shared_memory_config.get("enabled", True)
        )
        self.publish_notifications = bool(
            self.shared_memory_config.get("publish_notifications", True)
        )
        self.shared_ttl_seconds = optional_nonnegative_int(
            self.shared_memory_config.get("ttl_seconds", 86400),
            "quality_agent.shared_memory.ttl_seconds",
        )
        self.shared_result_key_prefix = (
            str(
                self.shared_memory_config.get(
                    "result_key_prefix",
                    "quality_agent.result",
                )
            ).strip()
            or "quality_agent.result"
        )
        self.shared_summary_key_prefix = (
            str(
                self.shared_memory_config.get(
                    "summary_key_prefix",
                    "quality_agent.summary",
                )
            ).strip()
            or "quality_agent.summary"
        )
        self.shared_error_key_prefix = (
            str(
                self.shared_memory_config.get(
                    "error_key_prefix",
                    "quality_agent.error",
                )
            ).strip()
            or "quality_agent.error"
        )
        self.shared_event_channel = (
            str(
                self.shared_memory_config.get(
                    "event_channel",
                    "quality.events",
                )
            ).strip()
            or "quality.events"
        )

        bridge_resolution = self._mapping_section(
            cfg,
            "bridge_resolution",
            required=False,
        )
        self.resolve_handler_from_factory = bool(
            bridge_resolution.get("resolve_handler_from_factory", False)
        )
        self.resolve_safety_from_factory = bool(
            bridge_resolution.get("resolve_safety_from_factory", False)
        )
        self.handler_factory_names = string_list(
            bridge_resolution.get(
                "handler_factory_names",
                ["handler_agent", "handler", "HandlerAgent"],
            ),
            deduplicate=True,
        )
        self.safety_factory_names = string_list(
            bridge_resolution.get(
                "safety_factory_names",
                ["safety_agent", "safety", "SafetyAgent"],
            ),
            deduplicate=True,
        )

        self.intelligence_config = self._mapping_section(
            cfg,
            "intelligence",
            required=False,
        )
        self.intelligence_enabled = bool(
            self.intelligence_config.get("enabled", True)
        )
        self.use_fitness_policy = bool(
            self.intelligence_config.get("use_fitness_policy", True)
        )
        self.relationship_integration = self._mapping_section(
            self.intelligence_config,
            "relationship_integration",
            required=False,
        )
        self.relationship_integration_enabled = bool(
            self.relationship_integration.get("enabled", True)
        )
        self.relationship_weight = bounded_score(
            self.relationship_integration.get("weight", 0.15),
            field_name="quality_agent.intelligence.relationship_integration.weight",
        )
        self.apply_relationship_score_without_baseline = bool(
            self.relationship_integration.get(
                "apply_score_when_baseline_unavailable",
                False,
            )
        )
        self.fail_closed_on_relationship_error = bool(
            self.relationship_integration.get(
                "fail_closed_on_error",
                False,
            )
        )
        self.policy_enforcement_config = self._mapping_section(
            self.intelligence_config,
            "policy_enforcement",
            required=False,
        )

    def _initialize_intelligence_modules(self) -> None:
        if not self.intelligence_enabled:
            return

        fitness_cfg = self._mapping_section(
            self.intelligence_config,
            "fitness_policy",
            required=True,
        )
        baseline_cfg = self._mapping_section(
            self.intelligence_config,
            "baseline_governance",
            required=True,
        )
        relationship_cfg = self._mapping_section(
            self.intelligence_config,
            "relationship_quality",
            required=True,
        )
        evidence_cfg = self._mapping_section(
            self.intelligence_config,
            "evidence_calibration",
            required=True,
        )

        self.fitness_policy = FitnessPolicyResolver(
            memory=self.quality_memory,
            config=fitness_cfg,
        )
        self.baseline_governor = BaselineGovernor(
            memory=self.quality_memory,
            config=baseline_cfg,
        )
        self.relationship_quality = RelationshipQuality(
            memory=self.quality_memory,
            config=relationship_cfg,
        )
        self.evidence_calibrator = EvidenceCalibrator(
            config=evidence_cfg,
        )

    @staticmethod
    def _mapping_section(
        source: Mapping[str, Any],
        key: str,
        *,
        required: bool,
    ) -> Dict[str, Any]:
        value = source.get(key)
        if value is None:
            if required:
                raise DataQualityError(
                    message=f"Missing required QualityAgent config section '{key}'",
                    error_type=QualityErrorType.CONFIGURATION_INVALID,
                    severity=QualitySeverity.HIGH,
                    retryable=False,
                    stage=QualityStage.VALIDATION,
                    domain=QualityDomain.SYSTEM,
                    disposition=QualityDisposition.ESCALATE,
                    context={"section": key},
                    remediation=(
                        f"Define quality_agent.intelligence.{key} "
                        "in agents_config.yaml."
                    ),
                )
            return {}
        if not isinstance(value, Mapping):
            raise DataQualityError(
                message=f"QualityAgent config section '{key}' must be a mapping",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.ESCALATE,
                context={
                    "section": key,
                    "value_type": type(value).__name__,
                },
                remediation=(
                    f"Define '{key}' as a YAML mapping in agents_config.yaml."
                ),
            )
        return dict(value)

    # ------------------------------------------------------------------
    # Runtime wiring
    # ------------------------------------------------------------------
    def attach_runtime(
        self,
        *,
        shared_memory: Any = None,
        handler_bridge: Any = None,
        safety_bridge: Any = None,
    ) -> None:
        if shared_memory is not None:
            self.shared_memory = shared_memory
        if handler_bridge is not None:
            self.handler_bridge = handler_bridge
        if safety_bridge is not None:
            self.safety_bridge = safety_bridge

        self.workflow_control.attach_runtime(
            shared_memory=self.shared_memory,
            handler_bridge=self.handler_bridge,
            safety_bridge=self.safety_bridge,
        )

    # ------------------------------------------------------------------
    # Public orchestration
    # ------------------------------------------------------------------
    def evaluate_batch(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        dataset_id: str,
        source_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        schema: Optional[Mapping[str, Any]] = None,
        baseline: Optional[Mapping[str, Any]] = None,
        relationship_baseline: Optional[Mapping[str, Any]] = None,
        label_field: Optional[str] = None,
        feature_fields: Optional[Sequence[str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        source_metadata: Optional[Mapping[str, Any]] = None,
        window: Optional[str] = None,
        use_case: Optional[str] = None,
        policy_id: Optional[str] = None,
        policy_overrides: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
        workflow_context: Optional[Mapping[str, Any]] = None,
        handler_bridge: Any = None,
        safety_bridge: Any = None,
    ) -> Dict[str, Any]:
        with quality_error_boundary(
            stage=QualityStage.VALIDATION,
            context={
                "operation": "evaluate_batch",
                "dataset_id": dataset_id,
                "source_id": source_id,
                "batch_id": batch_id,
            },
            error_type=QualityErrorType.SCORING_PIPELINE_FAILED,
            severity=QualitySeverity.HIGH,
            retryable=False,
            remediation=(
                "Review the incoming batch and QualityAgent configuration, "
                "then rerun the quality gate."
            ),
            disposition=QualityDisposition.BLOCK,
        ):
            normalized_records = normalize_records(
                records,
                field_name="records",
                require_nonempty=self.enabled,
            )
            dataset_key = nonempty_text(dataset_id, "dataset_id")
            source_key = nonempty_text(
                source_id or dataset_key,
                "source_id",
            )
            batch_key = nonempty_text(
                batch_id or generate_quality_id("batch"),
                "batch_id",
            )
            resolved_window = (
                str(window or self.default_window).strip()
                or self.default_window
            )
            normalized_context = normalized_mapping(context)
            normalized_workflow_context = normalized_mapping(workflow_context)

            if not self.enabled:
                return self._disabled_decision(
                    dataset_id=dataset_key,
                    source_id=source_key,
                    batch_id=batch_key,
                    context=normalized_context,
                )

            runtime_handler = (
                handler_bridge
                if handler_bridge is not None
                else self._resolve_bridge("handler")
            )
            runtime_safety = (
                safety_bridge
                if safety_bridge is not None
                else self._resolve_bridge("safety")
            )
            self.attach_runtime(
                shared_memory=self.shared_memory,
                handler_bridge=runtime_handler,
                safety_bridge=runtime_safety,
            )

            policy_decision = self._resolve_policy(
                use_case=use_case,
                source_id=source_key,
                policy_id=policy_id,
                context=normalized_context,
                overrides=policy_overrides,
            )
            active_policy = (
                policy_decision.policy
                if policy_decision is not None
                else None
            )
            active_weights = (
                dict(active_policy.subsystem_weights)
                if active_policy is not None and self.use_fitness_policy
                else dict(self.subsystem_weights)
            )
            active_pass_threshold = (
                active_policy.pass_threshold
                if active_policy is not None and self.use_fitness_policy
                else self.pass_threshold
            )
            active_warn_threshold = (
                active_policy.warn_threshold
                if active_policy is not None and self.use_fitness_policy
                else self.warn_threshold
            )

            subsystem_results: Dict[str, Dict[str, Any]] = {}
            subsystem_errors: List[DataQualityError] = []
            baseline_decisions: Dict[str, Any] = {}
            relationship_result: Dict[str, Any] = {}

            # Structural quality is always first because all later checks
            # assume minimally well-formed records.
            structural_result = self._execute_structural(
                records=normalized_records,
                dataset_id=dataset_key,
                source_id=source_key,
                batch_id=batch_key,
                schema=schema,
                window=resolved_window,
                context=normalized_context,
            )
            subsystem_results["structural"] = structural_result.to_dict()
            self._collect_subsystem_error(
                structural_result,
                subsystem_errors,
            )

            if (
                self.stop_on_blocking_structural
                and structural_result.verdict == "block"
            ):
                logger.warning(
                    "Structural quality blocked batch '%s'; semantic, "
                    "statistical and relationship checks were skipped.",
                    batch_key,
                )
                semantic_result = self._skipped_subsystem_result(
                    subsystem="semantic",
                    reason=(
                        "Skipped because StructuralQuality returned block "
                        "and stop_on_blocking_structural is enabled."
                    ),
                    source_id=source_key,
                    batch_id=batch_key,
                )
                statistical_result = self._skipped_subsystem_result(
                    subsystem="statistical",
                    reason=(
                        "Skipped because StructuralQuality returned block "
                        "and stop_on_blocking_structural is enabled."
                    ),
                    source_id=source_key,
                    batch_id=batch_key,
                )
            else:
                # Semantic precedes statistical only at the orchestration
                # layer because baseline promotion needs provenance/source
                # reliability evidence. This does not move semantic logic.
                semantic_result = self._execute_semantic(
                    records=normalized_records,
                    source_id=source_key,
                    batch_id=batch_key,
                    label_field=label_field,
                    feature_fields=feature_fields,
                    provenance=provenance,
                    source_metadata=source_metadata,
                    schema_version=schema_version(
                        schema,
                        structural_result.raw_result,
                    ),
                    window=resolved_window,
                    context=normalized_context,
                )
                self._collect_subsystem_error(
                    semantic_result,
                    subsystem_errors,
                )

                source_reliability = self._resolve_source_reliability(
                    semantic_result,
                    policy_decision,
                    normalized_context,
                )
                schema_ver = schema_version(
                    schema,
                    structural_result.raw_result,
                )
                required_fields = schema_required_fields(schema)

                statistical_baseline: Mapping[str, Any] = (
                    baseline if baseline is not None else {}
                )

                if self.intelligence_enabled:
                    statistical_profile = (
                        self.statistical_quality.build_profile(
                            normalized_records
                        )
                    )
                    statistical_baseline_decision = (
                        self.baseline_governor.resolve(
                            source_id=source_key,
                            current_profile=statistical_profile,
                            metric=self.baseline_governor.metric_name,
                            explicit_baseline=baseline,
                            structural_verdict=structural_result.verdict,
                            semantic_verdict=semantic_result.verdict,
                            source_reliability=source_reliability,
                            schema_version=schema_ver,
                            window=resolved_window,
                            context=normalized_context,
                        )
                        if self.baseline_governor is not None
                        else None
                    )
                    if statistical_baseline_decision is not None:
                        baseline_decisions["statistical"] = (
                            statistical_baseline_decision.to_dict()
                        )
                        statistical_baseline = (
                            statistical_baseline_decision.profile
                            if statistical_baseline_decision.usable
                            and statistical_baseline_decision.profile
                            else {}
                        )

                # Passing {} is deliberate when governance did not approve a
                # baseline: StatisticalQuality sees an explicit empty
                # baseline instead of auto-registering the current profile.
                statistical_result = self._execute_statistical(
                    records=normalized_records,
                    source_id=source_key,
                    batch_id=batch_key,
                    baseline=statistical_baseline,
                    required_fields=required_fields,
                    window=resolved_window,
                    schema_version=schema_ver,
                    context=normalized_context,
                )
                self._collect_subsystem_error(
                    statistical_result,
                    subsystem_errors,
                )

                if self.intelligence_enabled:
                    (
                        relationship_result,
                        relationship_baseline_decision,
                    ) = self._execute_relationship_intelligence(
                        records=normalized_records,
                        source_id=source_key,
                        batch_id=batch_key,
                        structural_result=structural_result,
                        semantic_result=semantic_result,
                        source_reliability=source_reliability,
                        schema_version_value=schema_ver,
                        window=resolved_window,
                        explicit_baseline=relationship_baseline,
                        context=normalized_context,
                    )
                    if relationship_baseline_decision is not None:
                        baseline_decisions["relationship"] = (
                            relationship_baseline_decision.to_dict()
                        )
                    statistical_result = self._integrate_relationship_result(
                        statistical_result,
                        relationship_result,
                    )

            subsystem_results["semantic"] = semantic_result.to_dict()
            subsystem_results["statistical"] = statistical_result.to_dict()

            if subsystem_errors and self.fail_closed_on_subsystem_error:
                raise DataQualityErrorGroup(
                    message=(
                        "One or more quality subsystems failed while "
                        f"evaluating batch '{batch_key}'"
                    ),
                    error_type=QualityErrorType.INTERNAL_QUALITY_AGENT_FAILURE,
                    severity=QualitySeverity.HIGH,
                    retryable=False,
                    stage=QualityStage.VALIDATION,
                    domain=QualityDomain.SYSTEM,
                    disposition=QualityDisposition.BLOCK,
                    dataset_id=dataset_key,
                    source_id=source_key,
                    batch_id=batch_key,
                    errors=subsystem_errors,
                    remediation=(
                        "Review subsystem-specific failures and rerun the "
                        "batch after correcting the failed stage."
                    ),
                )

            source_reliability = self._resolve_source_reliability(
                semantic_result,
                policy_decision,
                normalized_context,
            )
            detector_agreement = self._detector_agreement(
                subsystem_results
            )

            evidence_summary: Dict[str, Any] = {}
            if self.intelligence_enabled and self.evidence_calibrator is not None:
                subsystem_results = self._calibrate_subsystem_evidence(
                    subsystem_results,
                    population_count=len(normalized_records),
                    baseline_decisions=baseline_decisions,
                    source_reliability=source_reliability,
                    detector_agreement=detector_agreement,
                )
                all_calibrations = [
                    finding.get("evidence", {})
                    for result in subsystem_results.values()
                    for finding in result.get("findings", [])
                    if isinstance(finding, Mapping)
                    and isinstance(finding.get("evidence"), Mapping)
                ]
                evidence_summary = self.evidence_calibrator.aggregate(
                    all_calibrations
                )

            combined_findings = self._combine_findings(
                subsystem_results,
                pass_threshold=active_pass_threshold,
                warn_threshold=active_warn_threshold,
            )
            aggregate_score = aggregate_subsystem_score(
                subsystem_results,
                active_weights,
            )
            base_verdict = aggregate_subsystem_verdict(
                subsystem_results,
                batch_score=aggregate_score,
                pass_threshold=active_pass_threshold,
                warn_threshold=active_warn_threshold,
            )

            policy_enforcement = self._enforce_fitness_policy(
                policy_decision=policy_decision,
                findings=combined_findings,
                baseline_decisions=baseline_decisions,
                evidence=evidence_summary,
                source_reliability=source_reliability,
                base_verdict=base_verdict,
                relationship_executed=bool(relationship_result),
            )
            preliminary_verdict = policy_enforcement.get(
                "verdict",
                base_verdict,
            )

            policy_finding = policy_enforcement.get("finding")
            if isinstance(policy_finding, Mapping):
                combined_findings.append(dict(policy_finding))

            flags = merge_unique_strings(
                subsystem_results["structural"].get("flags"),
                subsystem_results["statistical"].get("flags"),
                subsystem_results["semantic"].get("flags"),
                relationship_result.get("flags"),
                policy_enforcement.get("flags"),
                [f"quality_agent:{preliminary_verdict}"],
            )
            remediation_actions = merge_unique_strings(
                subsystem_results["structural"].get(
                    "remediation_actions"
                ),
                subsystem_results["statistical"].get(
                    "remediation_actions"
                ),
                subsystem_results["semantic"].get(
                    "remediation_actions"
                ),
                relationship_result.get("remediation_actions"),
                policy_enforcement.get("remediation_actions"),
            )

            workflow_payload_context = {
                "dataset_id": dataset_key,
                "source_id": source_key,
                "batch_id": batch_key,
                "window": resolved_window,
                "use_case": (
                    active_policy.use_case
                    if active_policy is not None
                    else use_case
                ),
                "fitness_policy_id": (
                    active_policy.policy_id
                    if active_policy is not None
                    else None
                ),
                **normalized_context,
                **normalized_workflow_context,
            }

            workflow_decision = (
                self.workflow_control.coordinate_batch(
                    source_id=source_key,
                    batch_id=batch_key,
                    findings=combined_findings,
                    batch_score=aggregate_score,
                    records=normalized_records,
                    context=workflow_payload_context,
                    shared_memory=self.shared_memory,
                    handler_bridge=runtime_handler,
                    safety_bridge=runtime_safety,
                )
                if self.auto_route_via_workflow
                else {}
            )

            final_verdict = self._resolve_final_verdict(
                preliminary_verdict=preliminary_verdict,
                workflow_decision=workflow_decision,
            )
            final_score = self._resolve_final_score(
                aggregate_score=aggregate_score,
                workflow_decision=workflow_decision,
            )

            final_flags = merge_unique_strings(
                flags,
                (
                    workflow_decision.get("flags")
                    if (
                        self.include_workflow_findings_in_summary
                        and workflow_decision
                    )
                    else []
                ),
            )
            final_remediation = merge_unique_strings(
                remediation_actions,
                (
                    workflow_decision.get(
                        "remediation_plan",
                        {},
                    ).get("actions")
                    if workflow_decision
                    else []
                ),
            )
            quarantine_entries = (
                list(workflow_decision.get("quarantine_entries", []))
                if workflow_decision
                else []
            )
            route_records = (
                list(workflow_decision.get("route_records", []))
                if workflow_decision
                else []
            )

            confidence = (
                bounded_score(
                    evidence_summary.get("confidence", 0.0),
                    field_name="quality_agent.evidence.confidence",
                )
                if evidence_summary
                else decision_confidence(
                    subsystem_results,
                    pass_threshold=active_pass_threshold,
                    warn_threshold=active_warn_threshold,
                )
            )

            decision = QualityAgentDecision(
                decision_id=generate_quality_id("quality_decision"),
                dataset_id=dataset_key,
                source_id=source_key,
                batch_id=batch_key,
                verdict=normalize_verdict(final_verdict),
                batch_score=bounded_score(
                    final_score,
                    field_name="quality_agent.batch_score",
                ),
                subsystem_scores={
                    name: float(result.get("batch_score", 0.0))
                    for name, result in subsystem_results.items()
                },
                subsystem_verdicts={
                    name: normalize_verdict(
                        result.get("verdict", "warn"),
                        default="warn",
                    )
                    for name, result in subsystem_results.items()
                },
                flags=final_flags,
                remediation_actions=final_remediation,
                quarantine_count=len(quarantine_entries),
                subsystem_results=deepcopy(subsystem_results),
                workflow_decision=deepcopy(workflow_decision),
                route_records=deepcopy(route_records),
                quarantine_entries=deepcopy(quarantine_entries),
                shared_memory_keys=[],
                context={
                    **normalized_context,
                    "window": resolved_window,
                    "record_count": len(normalized_records),
                    "preliminary_verdict": preliminary_verdict,
                    "preliminary_batch_score": aggregate_score,
                    "active_pass_threshold": active_pass_threshold,
                    "active_warn_threshold": active_warn_threshold,
                },
                created_at=time.time(),
                fitness_policy=(
                    policy_decision.to_dict()
                    if policy_decision is not None
                    else {}
                ),
                baseline_decisions=deepcopy(baseline_decisions),
                relationship_quality=deepcopy(relationship_result),
                evidence=deepcopy(evidence_summary),
                policy_enforcement=deepcopy(policy_enforcement),
            )

            decision_dict = decision.to_dict()
            decision_dict["quality_verdict"] = decision_dict["verdict"]
            decision_dict["record_level_flags"] = record_level_flags(
                subsystem_results
            )
            decision_dict["confidence"] = confidence
            decision_dict["uncertainty"] = 1.0 - confidence
            decision_dict["evidence_strength"] = evidence_summary.get(
                "evidence_strength",
                "unavailable",
            )
            decision_dict["quarantine_queue_entries"] = deepcopy(
                quarantine_entries
            )
            decision_dict["subsystem_findings"] = deepcopy(
                combined_findings
            )

            decision_dict["shared_memory_keys"] = (
                self._publish_result_to_shared_memory(
                    decision=decision_dict,
                    records=normalized_records,
                )
            )
            self._persist_local_result(decision_dict)
            return decision_dict

    def assess_batch(self, records: Sequence[Mapping[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        return self.evaluate_batch(records, **kwargs)

    def latest_decision(self, batch_id: str) -> Optional[Dict[str, Any]]:
        batch_key = nonempty_text(batch_id, "batch_id")
        result = self._results_by_batch.get(batch_key)
        return None if result is None else deepcopy(result)

    def recent_source_results(self, source_id: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        source_key = nonempty_text(source_id, "source_id")
        return deepcopy(
            (self._results_by_source.get(source_key) or [])[
                -max(int(limit), 1) :
            ]
        )

    def summary(self) -> Dict[str, Any]:
        all_results = list(self._results_by_batch.values())
        verdict_counts = {"pass": 0, "warn": 0, "block": 0}
        evidence_strength_counts: Dict[str, int] = {}
        policy_counts: Dict[str, int] = {}

        for result in all_results:
            verdict = str(result.get("verdict", "warn"))
            if verdict in verdict_counts:
                verdict_counts[verdict] += 1

            strength = str(result.get("evidence_strength", "unavailable"))
            evidence_strength_counts[strength] = (evidence_strength_counts.get(strength, 0) + 1)

            policy = result.get("fitness_policy", {})
            if isinstance(policy, Mapping):
                policy_payload = policy.get("policy", {})
                if isinstance(policy_payload, Mapping):
                    policy_id = policy_payload.get("policy_id")
                    if policy_id:
                        key = str(policy_id)
                        policy_counts[key] = policy_counts.get(key, 0) + 1

        return {
            "enabled": self.enabled,
            "intelligence_enabled": self.intelligence_enabled,
            "results_tracked": len(all_results),
            "sources_tracked": len(self._results_by_source),
            "verdict_counts": verdict_counts,
            "evidence_strength_counts": evidence_strength_counts,
            "fitness_policy_counts": policy_counts,
            "subsystem_order": list(self.subsystem_order),
            "auto_route_via_workflow": self.auto_route_via_workflow,
            "publish_to_shared_memory": self.publish_to_shared_memory,
            "shared_quality_memory": self.quality_memory is not None,
            "intelligence_modules": {
                "fitness_policy": self.fitness_policy is not None,
                "baseline_governance": self.baseline_governor is not None,
                "relationship_quality": self.relationship_quality is not None,
                "evidence_calibration": self.evidence_calibrator is not None,
            },
        }

    def perform_task(self, task_data: Any) -> Dict[str, Any]:
        with quality_error_boundary(
            stage=QualityStage.ROUTING,
            context={"operation": "perform_task"},
            error_type=QualityErrorType.SCORING_PIPELINE_FAILED,
            severity=QualitySeverity.HIGH,
            retryable=False,
            remediation=(
                "Provide a valid QualityAgent task payload and rerun the agent."
            ),
            disposition=QualityDisposition.BLOCK,
        ):
            if not isinstance(task_data, Mapping):
                raise DataQualityError(
                    message=(
                        "QualityAgent.perform_task expects a mapping payload"
                    ),
                    error_type=QualityErrorType.CONFIGURATION_INVALID,
                    severity=QualitySeverity.MEDIUM,
                    retryable=False,
                    stage=QualityStage.ROUTING,
                    domain=QualityDomain.SYSTEM,
                    disposition=QualityDisposition.WARN,
                    context={"task_data_type": type(task_data).__name__},
                    remediation=(
                        "Wrap the payload in a mapping containing records "
                        "and quality-evaluation metadata."
                    ),
                )

            operation = str(task_data.get("operation", "evaluate_batch")).strip().lower()
            payload = task_data.get("task_data", task_data.get("input_data", task_data.get("payload", task_data)))
            payload_map = (
                payload
                if isinstance(payload, Mapping)
                else {"records": payload}
            )
            context = payload_map.get("context", task_data.get("context", {}))

            if operation in {"summary", "status"}:
                return self.summary()

            if operation in {
                "latest_decision",
                "get_latest_decision",
            }:
                resolved_batch = (
                    task_data.get("batch_id")
                    or payload_map.get("batch_id")
                )
                if not resolved_batch:
                    return {}
                return self.latest_decision(str(resolved_batch)) or {}

            if operation in {
                "resolve_policy",
                "fitness_policy",
            }:
                decision = self._resolve_policy(
                    use_case=payload_map.get("use_case"),
                    source_id=payload_map.get("source_id"),
                    policy_id=payload_map.get("policy_id"),
                    context=normalized_mapping(context),
                    overrides=payload_map.get("policy_overrides"),
                )
                return (
                    decision.to_dict()
                    if decision is not None
                    else {"intelligence_enabled": False}
                )

            if operation in {
                "workflow_only",
                "route_quality",
                "coordinate_batch",
            }:
                return self.workflow_control.coordinate_batch(
                    source_id=nonempty_text(payload_map.get("source_id"), "source_id"),
                    batch_id=nonempty_text(payload_map.get("batch_id"), "batch_id"),
                    findings=payload_map.get("findings"),
                    batch_score=payload_map.get("batch_score"),
                    records=payload_map.get("records"),
                    context=context,
                    shared_memory=self.shared_memory,
                    handler_bridge=self._resolve_bridge("handler"),
                    safety_bridge=self._resolve_bridge("safety"),
                )

            return self.evaluate_batch(
                records=normalize_records(
                    payload_map.get("records", []),
                    require_nonempty=True,
                ),
                dataset_id=nonempty_text(
                    payload_map.get("dataset_id")
                    or payload_map.get("source_id"),
                    "dataset_id",
                ),
                source_id=payload_map.get("source_id"),
                batch_id=payload_map.get("batch_id"),
                schema=payload_map.get("schema"),
                baseline=payload_map.get("baseline"),
                relationship_baseline=payload_map.get("relationship_baseline"),
                label_field=payload_map.get("label_field"),
                feature_fields=payload_map.get("feature_fields"),
                provenance=payload_map.get("provenance"),
                source_metadata=payload_map.get("source_metadata"),
                window=payload_map.get("window"),
                use_case=payload_map.get("use_case"),
                policy_id=payload_map.get("policy_id"),
                policy_overrides=payload_map.get("policy_overrides"),
                context=context,
                workflow_context=payload_map.get("workflow_context"),
                handler_bridge=payload_map.get("handler_bridge"),
                safety_bridge=payload_map.get("safety_bridge"),
            )

    # ------------------------------------------------------------------
    # Intelligence
    # ------------------------------------------------------------------
    def _resolve_policy(
        self,
        *,
        use_case: Optional[str],
        source_id: Optional[str],
        policy_id: Optional[str],
        context: Mapping[str, Any],
        overrides: Optional[Mapping[str, Any]],
    ) -> Optional[FitnessPolicyDecision]:
        if (
            not self.intelligence_enabled
            or not self.use_fitness_policy
            or self.fitness_policy is None
            or not self.fitness_policy.enabled
        ):
            return None

        effective_use_case = (
            use_case
            or context.get("intended_use")
            or context.get("use_case")
            or context.get("task_type")
            or "general"
        )
        return self.fitness_policy.resolve(
            use_case=str(effective_use_case),
            source_id=source_id,
            policy_id=policy_id,
            context=context,
            overrides=overrides,
        )

    def _execute_relationship_intelligence(
        self,
        *,
        records: Sequence[Mapping[str, Any]],
        source_id: str,
        batch_id: str,
        structural_result: SubsystemExecution,
        semantic_result: SubsystemExecution,
        source_reliability: Optional[float],
        schema_version_value: Optional[str],
        window: str,
        explicit_baseline: Optional[Mapping[str, Any]],
        context: Mapping[str, Any],
    ) -> tuple[Dict[str, Any], Optional[BaselineDecision]]:
        if (
            self.relationship_quality is None
            or self.baseline_governor is None
        ):
            return {}, None

        try:
            profile = self.relationship_quality.build_profile(records)
            baseline_decision = self.baseline_governor.resolve(
                source_id=source_id,
                current_profile=profile,
                metric=self.relationship_quality.profile_metric_name,
                explicit_baseline=explicit_baseline,
                structural_verdict=structural_result.verdict,
                semantic_verdict=semantic_result.verdict,
                source_reliability=source_reliability,
                schema_version=schema_version_value,
                window=window,
                context=context,
            )
            approved_baseline = (
                baseline_decision.profile
                if baseline_decision.usable
                and baseline_decision.profile
                else {}
            )
            result = self.relationship_quality.assess(
                records,
                source_id=source_id,
                batch_id=batch_id,
                baseline_profile=approved_baseline,
            )
            result["baseline_status"] = baseline_decision.status.value
            result["baseline_usable"] = baseline_decision.usable
            return result, baseline_decision
        except Exception as exc:
            normalized = normalize_quality_exception(
                exc,
                stage=QualityStage.SCORING,
                context={
                    "operation": "relationship_quality",
                    "source_id": source_id,
                    "batch_id": batch_id,
                },
                error_type=QualityErrorType.SCORING_PIPELINE_FAILED,
                severity=QualitySeverity.HIGH,
                retryable=False,
                remediation=(
                    "Review relationship-quality configuration or input "
                    "support before rerunning the batch."
                ),
                disposition=(
                    QualityDisposition.BLOCK
                    if self.fail_closed_on_relationship_error
                    else QualityDisposition.WARN
                ),
            )
            normalized.report()
            verdict = (
                "block"
                if self.fail_closed_on_relationship_error
                else "warn"
            )
            return {
                "source_id": source_id,
                "batch_id": batch_id,
                "verdict": verdict,
                "batch_score": 0.0 if verdict == "block" else 0.75,
                "confidence": 1.0,
                "baseline_available": False,
                "findings": [],
                "flags": ["relationship_quality_failure"],
                "remediation_actions": (
                    [normalized.remediation]
                    if normalized.remediation
                    else []
                ),
                "error": normalized.to_dict(),
            }, None

    def _integrate_relationship_result(
        self,
        statistical_result: SubsystemExecution,
        relationship_result: Mapping[str, Any],
    ) -> SubsystemExecution:
        if not relationship_result:
            return statistical_result

        relationship_findings = self._relationship_findings(relationship_result)
        baseline_available = bool(
            relationship_result.get("baseline_available", False)
            or relationship_result.get("baseline_usable", False)
        )
        apply_score = (
            baseline_available
            or self.apply_relationship_score_without_baseline
        )

        relationship_score = bounded_score(
            relationship_result.get("batch_score", statistical_result.batch_score),
            field_name="relationship_quality.batch_score",
        )
        relationship_verdict = normalize_verdict(
            relationship_result.get("verdict", "warn"),
            default="warn",
        )

        score = statistical_result.batch_score
        verdict = statistical_result.verdict
        if apply_score:
            score = (
                (1.0 - self.relationship_weight)
                * statistical_result.batch_score
                + self.relationship_weight * relationship_score
            )
            verdict = worst_verdict(
                statistical_result.verdict,
                relationship_verdict,
            )

        raw = deepcopy(statistical_result.raw_result)
        raw["relationship_quality"] = deepcopy(dict(relationship_result))
        raw["batch_score"] = bounded_score(score, field_name="statistical.relationship_adjusted_score")
        raw["verdict"] = verdict

        shift_metrics = dict(statistical_result.shift_metrics)
        shift_metrics["relationship_quality_score"] = relationship_score
        shift_metrics["relationship_baseline_available"] = (
            1.0 if baseline_available else 0.0
        )

        return replace(
            statistical_result,
            verdict=verdict,
            batch_score=bounded_score(
                score,
                field_name="statistical.relationship_adjusted_score",
            ),
            findings=[
                *statistical_result.findings,
                *relationship_findings,
            ],
            flags=merge_unique_strings(
                statistical_result.flags,
                relationship_result.get("flags"),
            ),
            remediation_actions=merge_unique_strings(
                statistical_result.remediation_actions,
                relationship_result.get("remediation_actions"),
            ),
            shift_metrics=shift_metrics,
            raw_result=raw,
        )

    @staticmethod
    def _relationship_findings(
        relationship_result: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for finding in relationship_result.get("findings", []) or []:
            if not isinstance(finding, Mapping):
                continue
            pair = finding.get("pair", [])
            relationship_type = str(
                finding.get("relationship_type", "relationship")
            )
            normalized.append(
                {
                    "checker": "relationship_drift",
                    "check": "relationship_drift",
                    "domain": "statistical",
                    "verdict": finding.get("verdict", "warn"),
                    "severity": finding.get("severity", "medium"),
                    "confidence": finding.get("confidence", 1.0),
                    "score": finding.get("score"),
                    "message": (
                        f"{relationship_type} changed for pair {pair}"
                    ),
                    "flags": finding.get("flags", []),
                    "affected_records": [],
                    "remediation_actions": finding.get(
                        "remediation_actions",
                        [],
                    ),
                    "metrics": {
                        "relationship_type": relationship_type,
                        "pair": pair,
                        "current_value": finding.get("current_value"),
                        "baseline_value": finding.get("baseline_value"),
                        "absolute_delta": finding.get("absolute_delta"),
                        "current_support": finding.get("current_support"),
                        "baseline_support": finding.get("baseline_support"),
                        "fisher_z": finding.get("fisher_z"),
                        "p_value": finding.get("p_value"),
                        "statistically_supported": finding.get(
                            "statistically_supported"
                        ),
                    },
                    "error_type": (
                        QualityErrorType.DISTRIBUTION_DRIFT_DETECTED.value
                        if normalize_verdict(
                            finding.get("verdict", "warn"),
                            default="warn",
                        )
                        == "block"
                        else ""
                    ),
                }
            )
        return normalize_findings(
            normalized,
            default_domain="statistical",
            default_checker="relationship_drift",
        )

    def _calibrate_subsystem_evidence(
        self,
        subsystem_results: Mapping[str, Mapping[str, Any]],
        *,
        population_count: int,
        baseline_decisions: Mapping[str, Any],
        source_reliability: Optional[float],
        detector_agreement: float,
    ) -> Dict[str, Dict[str, Any]]:
        if (
            self.evidence_calibrator is None
            or not self.evidence_calibrator.enabled
        ):
            return {
                str(k): deepcopy(dict(v))
                for k, v in subsystem_results.items()
            }

        calibrated: Dict[str, Dict[str, Any]] = {}
        statistical_baseline_status = self._baseline_status(
            baseline_decisions.get("statistical")
        )
        relationship_baseline_status = self._baseline_status(
            baseline_decisions.get("relationship")
        )

        for subsystem, result in subsystem_results.items():
            result_copy = deepcopy(dict(result))
            updated_findings: List[Dict[str, Any]] = []

            for raw_finding in result_copy.get("findings", []) or []:
                if not isinstance(raw_finding, Mapping):
                    continue
                finding = dict(raw_finding)
                check = self._finding_check_name(finding)

                if check == "relationship_drift":
                    baseline_status = relationship_baseline_status
                elif check in _BASELINE_DEPENDENT_CHECKS:
                    baseline_status = statistical_baseline_status
                else:
                    # Baseline is not a prerequisite for this measurement.
                    baseline_status = "not_applicable"

                updated_findings.append(
                    self.evidence_calibrator.calibrate_finding(
                        finding,
                        population_count=population_count,
                        baseline_status=baseline_status,
                        source_reliability=source_reliability,
                        detector_agreement=detector_agreement,
                    )
                )

            result_copy["findings"] = updated_findings
            calibrated[subsystem] = result_copy

        return calibrated

    def _enforce_fitness_policy(
        self,
        *,
        policy_decision: Optional[FitnessPolicyDecision],
        findings: Sequence[Mapping[str, Any]],
        baseline_decisions: Mapping[str, Any],
        evidence: Mapping[str, Any],
        source_reliability: Optional[float],
        base_verdict: str,
        relationship_executed: bool,
    ) -> Dict[str, Any]:
        base = normalize_verdict(base_verdict)
        if policy_decision is None:
            return {
                "verdict": base,
                "changed": False,
                "flags": [],
                "remediation_actions": [],
                "reasons": [],
            }

        policy = policy_decision.policy
        verdict = base
        flags: List[str] = []
        remediation: List[str] = []
        reasons: List[str] = []

        if (
            self.fitness_policy is not None
            and self.fitness_policy.should_hard_block(
                policy,
                findings,
            )
        ):
            verdict = "block"
            flags.append("fitness_policy_hard_block")
            remediation.append("review_hard_block_quality_findings")
            reasons.append(
                "A finding matched the active policy hard-block error set."
            )

        if (
            source_reliability is None
            or source_reliability < policy.min_source_reliability
        ):
            requested = self._configured_disposition(
                "source_reliability_failure",
                default="warn",
            )
            verdict = self._escalate_verdict(verdict, requested)
            flags.append("fitness_policy_source_reliability")
            remediation.append("revalidate_source_reliability")
            reasons.append(
                "Source reliability did not satisfy the active policy floor."
            )

        if policy.require_trusted_baseline:
            statistical = baseline_decisions.get("statistical", {})
            usable = (
                isinstance(statistical, Mapping)
                and bool(statistical.get("usable"))
            )
            if not usable:
                requested = self._configured_disposition(
                    "trusted_baseline_missing",
                    default="warn",
                )
                verdict = self._escalate_verdict(verdict, requested)
                flags.append("fitness_policy_trusted_baseline_missing")
                remediation.append("establish_trusted_baseline")
                reasons.append(
                    "The active policy requires a trusted statistical baseline."
                )

        observed_checks = self._observed_checks(
            findings,
            relationship_executed=relationship_executed,
        )
        missing_checks = sorted(
            set(policy.required_checks) - observed_checks
        )
        if missing_checks:
            requested = self._configured_disposition(
                "required_check_missing",
                default="warn",
            )
            verdict = self._escalate_verdict(verdict, requested)
            flags.append("fitness_policy_required_check_missing")
            remediation.append("run_required_quality_checks")
            reasons.append(
                "Required checks were not observed: "
                + ", ".join(missing_checks)
            )

        evidence_strength = str(
            evidence.get("evidence_strength", "insufficient")
        )
        if (
            self.fitness_policy is not None
            and evidence
            and not self.fitness_policy.evidence_requirement_satisfied(
                policy,
                evidence_strength,
            )
        ):
            requested = self._configured_disposition(
                "insufficient_evidence",
                default="warn",
            )
            verdict = self._escalate_verdict(verdict, requested)
            flags.append("fitness_policy_insufficient_evidence")
            remediation.append("collect_stronger_quality_evidence")
            reasons.append(
                "Aggregate evidence strength is below the active "
                f"policy requirement ({evidence_strength} < "
                f"{policy.minimum_evidence_strength})."
            )

        changed = verdict != base
        finding = None
        if changed or reasons:
            finding = {
                "checker": "fitness_policy",
                "domain": "system",
                "verdict": verdict,
                "severity": (
                    "critical"
                    if verdict == "block"
                    and "fitness_policy_hard_block" in flags
                    else "high"
                    if verdict == "block"
                    else "medium"
                    if verdict == "warn"
                    else "low"
                ),
                "confidence": (
                    float(evidence.get("confidence", 1.0))
                    if evidence
                    else 1.0
                ),
                "message": (
                    "Fitness-for-use policy enforcement: "
                    + "; ".join(reasons)
                ),
                "flags": flags,
                "remediation_actions": remediation,
                "affected_records": [],
                "error_type": (
                    QualityErrorType.POLICY_THRESHOLD_INVALID.value
                    if verdict == "block"
                    else ""
                ),
                "fitness_policy_id": policy.policy_id,
            }

        return {
            "verdict": verdict,
            "changed": changed,
            "flags": flags,
            "remediation_actions": remediation,
            "reasons": reasons,
            "observed_checks": sorted(observed_checks),
            "missing_required_checks": missing_checks,
            "source_reliability": source_reliability,
            "minimum_source_reliability": policy.min_source_reliability,
            "evidence_strength": evidence_strength,
            "minimum_evidence_strength": policy.minimum_evidence_strength,
            "finding": finding,
        }

    def _configured_disposition(self, key: str, *, default: str) -> str:
        value = str(self.policy_enforcement_config.get(key, default)).strip().lower()
        if value not in {"pass", "warn", "block"}:
            raise DataQualityError(
                message=(
                    "quality_agent.intelligence.policy_enforcement."
                    f"{key} must be pass, warn, or block"
                ),
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.ESCALATE,
                context={"key": key, "value": value},
                remediation=(
                    "Correct policy_enforcement in agents_config.yaml."
                ),
            )
        return value

    @staticmethod
    def _escalate_verdict(current: str, requested: str) -> str:
        # worst_verdict is monotonic: policy enforcement never turns a
        # stronger current verdict into a weaker one.
        return worst_verdict(current, requested)

    @staticmethod
    def _finding_check_name(finding: Mapping[str, Any]) -> str:
        return str(
            finding.get("check")
            or finding.get("check_name")
            or finding.get("checker")
            or ""
        ).strip().lower()

    def _observed_checks(
        self,
        findings: Sequence[Mapping[str, Any]],
        *,
        relationship_executed: bool,
    ) -> set[str]:
        observed: set[str] = set()
        aliases = {
            "schema_validation": "schema",
            "schema": "schema",
            "required_fields": "schema",
            "leakage": "leakage",
            "label_leakage": "leakage",
            "detect_label_leakage": "leakage",
            "provenance": "provenance",
            "source_provenance": "provenance",
            "consistency": "consistency",
            "cross_field_consistency": "consistency",
            "drift": "drift",
            "distribution_shift": "drift",
            "distribution_drift": "drift",
            "duplicates": "duplicates",
            "duplicate": "duplicates",
            "outliers": "outliers",
            "outlier": "outliers",
            "missingness": "missingness",
            "relationship_drift": "relationship_drift",
        }

        for finding in findings:
            name = self._finding_check_name(finding)
            if not name:
                continue
            observed.add(name)
            if name in aliases:
                observed.add(aliases[name])
            for token, canonical in aliases.items():
                if token in name:
                    observed.add(canonical)

        if relationship_executed:
            observed.add("relationship_drift")

        return observed

    @staticmethod
    def _baseline_status(value: Any) -> str:
        if isinstance(value, Mapping):
            return str(value.get("status", "unestablished"))
        return "unestablished"

    @staticmethod
    def _detector_agreement(subsystem_results: Mapping[str, Mapping[str, Any]]) -> float:
        verdicts = [
            normalize_verdict(
                result.get("verdict", "warn"),
                default="warn",
            )
            for result in subsystem_results.values()
            if isinstance(result, Mapping)
            and result.get("enabled", True)
        ]
        if not verdicts:
            return 0.0
        counts: Dict[str, int] = {}
        for verdict in verdicts:
            counts[verdict] = counts.get(verdict, 0) + 1
        return max(counts.values()) / len(verdicts)

    @staticmethod
    def _resolve_source_reliability(
        semantic_result: SubsystemExecution,
        policy_decision: Optional[FitnessPolicyDecision],
        context: Mapping[str, Any],
    ) -> Optional[float]:
        raw = semantic_result.raw_result
        value = raw.get("source_reliability")
        if value is None:
            value = context.get("source_reliability")
        if value is None and policy_decision is not None:
            value = policy_decision.source_reliability
        if value is None:
            return None
        return bounded_score(
            value,
            field_name="quality_agent.source_reliability",
        )

    # ------------------------------------------------------------------
    # Existing subsystem execution
    # ------------------------------------------------------------------
    def _execute_structural(
        self,
        *,
        records: Sequence[Mapping[str, Any]],
        dataset_id: str,
        source_id: str,
        batch_id: str,
        schema: Optional[Mapping[str, Any]],
        window: Optional[str],
        context: Optional[Mapping[str, Any]],
    ) -> SubsystemExecution:
        return self._execute_subsystem(
            subsystem="structural",
            fn=self.structural_quality.evaluate_batch,
            fn_kwargs={
                "records": records,
                "dataset_id": dataset_id,
                "source_id": source_id,
                "batch_id": batch_id,
                "schema": schema,
                "window": window,
                "context": context,
            },
            source_id=source_id,
            batch_id=batch_id,
        )

    def _execute_statistical(
        self,
        *,
        records: Sequence[Mapping[str, Any]],
        source_id: str,
        batch_id: str,
        baseline: Optional[Mapping[str, Any]],
        required_fields: Optional[Sequence[str]],
        window: Optional[str],
        schema_version: Optional[str],
        context: Optional[Mapping[str, Any]],
    ) -> SubsystemExecution:
        return self._execute_subsystem(
            subsystem="statistical",
            fn=self.statistical_quality.assess_batch,
            fn_kwargs={
                "records": records,
                "source_id": source_id,
                "batch_id": batch_id,
                "baseline": baseline,
                "required_fields": required_fields,
                "window": window,
                "schema_version": schema_version,
                "context": context,
            },
            source_id=source_id,
            batch_id=batch_id,
        )

    def _execute_semantic(
        self,
        *,
        records: Sequence[Mapping[str, Any]],
        source_id: str,
        batch_id: str,
        label_field: Optional[str],
        feature_fields: Optional[Sequence[str]],
        provenance: Optional[Mapping[str, Any]],
        source_metadata: Optional[Mapping[str, Any]],
        schema_version: Optional[str],
        window: Optional[str],
        context: Optional[Mapping[str, Any]],
    ) -> SubsystemExecution:
        return self._execute_subsystem(
            subsystem="semantic",
            fn=self.semantic_quality.evaluate_batch,
            fn_kwargs={
                "records": records,
                "source_id": source_id,
                "batch_id": batch_id,
                "label_field": label_field,
                "feature_fields": feature_fields,
                "provenance": provenance,
                "source_metadata": source_metadata,
                "schema_version": schema_version,
                "window": window,
                "context": context,
            },
            source_id=source_id,
            batch_id=batch_id,
        )

    def _execute_subsystem(
        self,
        *,
        subsystem: str,
        fn: Any,
        fn_kwargs: Dict[str, Any],
        source_id: str,
        batch_id: str,
    ) -> SubsystemExecution:
        started = time.perf_counter()
        try:
            result = fn(**fn_kwargs)
            payload = (
                normalized_mapping(result, field_name=f"{subsystem}.result")
                if isinstance(result, Mapping)
                else {"result": result}
            )
            duration_ms = max(
                (time.perf_counter() - started) * 1000.0,
                0.0,
            )
            verdict = normalize_verdict(
                payload.get("verdict", "warn"),
                default="warn",
            )
            return SubsystemExecution(
                subsystem=subsystem,
                enabled=True,
                verdict=verdict,
                batch_score=bounded_score(
                    payload.get(
                        "batch_score",
                        score_from_verdict(verdict),
                    ),
                    field_name=f"{subsystem}.batch_score",
                ),
                findings=normalize_findings(
                    payload.get("findings", []),
                    default_domain=subsystem,
                    default_checker=subsystem,
                ),
                flags=string_list(
                    payload.get("flags"),
                    deduplicate=True,
                ),
                remediation_actions=string_list(
                    payload.get("remediation_actions"),
                    deduplicate=True,
                ),
                quarantine_count=(
                    optional_nonnegative_int(
                        payload.get("quarantine_count", 0),
                        f"{subsystem}.quarantine_count",
                    )
                    or 0
                ),
                shift_metrics=float_mapping(
                    payload.get("shift_metrics", {}),
                    field_name=f"{subsystem}.shift_metrics",
                ),
                context={
                    "source_id": source_id,
                    "batch_id": batch_id,
                    "record_count": payload.get("record_count")
                    or payload.get("reviewed_record_count"),
                },
                duration_ms=duration_ms,
                raw_result=payload,
                error=None,
            )
        except Exception as exc:
            normalized = normalize_quality_exception(
                exc,
                stage=QualityStage.VALIDATION,
                context={
                    "operation": f"{subsystem}_quality",
                    "source_id": source_id,
                    "batch_id": batch_id,
                    "subsystem": subsystem,
                },
                error_type=QualityErrorType.INTERNAL_QUALITY_AGENT_FAILURE,
                severity=QualitySeverity.HIGH,
                retryable=False,
                remediation=(
                    f"Repair the {subsystem} subsystem failure and rerun "
                    "the batch evaluation."
                ),
                disposition=QualityDisposition.BLOCK,
            )
            normalized.report()
            duration_ms = max(
                (time.perf_counter() - started) * 1000.0,
                0.0,
            )
            verdict = (
                "block"
                if self.fail_closed_on_subsystem_error
                else "warn"
            )
            return SubsystemExecution(
                subsystem=subsystem,
                enabled=True,
                verdict=verdict,
                batch_score=0.0,
                findings=[
                    self._error_finding(
                        subsystem=subsystem,
                        error=normalized,
                    )
                ],
                flags=[f"{subsystem}_subsystem_failure"],
                remediation_actions=(
                    [normalized.remediation]
                    if normalized.remediation
                    else []
                ),
                quarantine_count=0,
                shift_metrics={},
                context={
                    "source_id": source_id,
                    "batch_id": batch_id,
                },
                duration_ms=duration_ms,
                raw_result={},
                error=normalized.to_dict(),
            )

    def _skipped_subsystem_result(
        self,
        *,
        subsystem: str,
        reason: str,
        source_id: str,
        batch_id: str,
    ) -> SubsystemExecution:
        return SubsystemExecution(
            subsystem=subsystem,
            enabled=False,
            verdict="pass",
            batch_score=1.0,
            findings=[
                {
                    "checker": subsystem,
                    "domain": subsystem,
                    "verdict": "pass",
                    "severity": "low",
                    "confidence": 1.0,
                    "message": reason,
                    "flags": [f"{subsystem}_skipped"],
                    "affected_records": [],
                    "remediation_actions": [],
                    "error_type": "",
                }
            ],
            flags=[f"{subsystem}_skipped"],
            remediation_actions=[],
            quarantine_count=0,
            shift_metrics={},
            context={
                "source_id": source_id,
                "batch_id": batch_id,
            },
            duration_ms=0.0,
            raw_result={
                "skipped": True,
                "reason": reason,
            },
            error=None,
        )

    @staticmethod
    def _collect_subsystem_error(result: SubsystemExecution, errors: List[DataQualityError]) -> None:
        if result.error is not None:
            errors.append(quality_error_from_payload(result.error))

    # ------------------------------------------------------------------
    # Aggregation / policy / workflow
    # ------------------------------------------------------------------
    def _combine_findings(
        self,
        subsystem_results: Mapping[str, Mapping[str, Any]],
        *,
        pass_threshold: float,
        warn_threshold: float,
    ) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []

        for subsystem in self.subsystem_order:
            result = subsystem_results.get(subsystem) or {}
            findings.extend(
                normalize_findings(
                    result.get("findings", []),
                    default_domain=subsystem,
                    default_checker=subsystem,
                )
            )

            result_verdict = normalize_verdict(
                result.get("verdict", "warn"),
                default="warn",
            )
            score = bounded_score(
                result.get("batch_score", score_from_verdict(result_verdict)),
                field_name=f"{subsystem}.batch_score",
            )
            findings.append(
                {
                    "checker": subsystem,
                    "domain": subsystem,
                    "verdict": result_verdict,
                    "severity": self._summary_severity_for_verdict(
                        result_verdict
                    ),
                    "confidence": (
                        0.95
                        if score >= pass_threshold
                        else 0.80
                        if score >= warn_threshold
                        else 0.65
                    ),
                    "message": (
                        f"{subsystem} subsystem summary "
                        f"verdict={result_verdict} score={score:.6f}"
                    ),
                    "flags": string_list(
                        result.get("flags"),
                        deduplicate=True,
                    ),
                    "affected_records": [],
                    "remediation_actions": string_list(
                        result.get("remediation_actions"),
                        deduplicate=True,
                    ),
                    "error_type": self._summary_error_type(
                        subsystem,
                        result_verdict,
                    ),
                    "subsystem_summary": True,
                }
            )

        return normalize_findings(findings)

    def _resolve_final_verdict(
        self,
        *,
        preliminary_verdict: str,
        workflow_decision: Mapping[str, Any],
    ) -> str:
        preliminary = normalize_verdict(preliminary_verdict)

        if not workflow_decision or not self.prefer_workflow_verdict:
            return preliminary

        workflow = normalize_verdict(
            workflow_decision.get("verdict", preliminary),
            default=preliminary,
        )
        if self.allow_workflow_verdict_relaxation:
            return workflow

        # Default production rule: operational workflow may escalate a
        # QualityAgent verdict but may not silently relax it.
        return worst_verdict(preliminary, workflow)

    def _resolve_final_score(
        self,
        *,
        aggregate_score: float,
        workflow_decision: Mapping[str, Any],
    ) -> float:
        base = bounded_score(aggregate_score, field_name="quality_agent.aggregate_score")
        if not workflow_decision:
            return base
        workflow_score = bounded_score(
            workflow_decision.get("batch_score", base),
            field_name="workflow.batch_score",
        )
        if self.allow_workflow_verdict_relaxation:
            return workflow_score
        return min(base, workflow_score)

    # ------------------------------------------------------------------
    # Shared memory / local history
    # ------------------------------------------------------------------
    def _publish_result_to_shared_memory(
        self,
        *,
        decision: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> List[str]:
        if (
            not self.publish_to_shared_memory
            or self.shared_memory is None
        ):
            return []

        shared_keys: List[str] = []
        try:
            result_key = (f"{self.shared_result_key_prefix}:{decision['batch_id']}")
            summary_key = (f"{self.shared_summary_key_prefix}:{decision['source_id']}")
            summary_payload = {
                "dataset_id": decision.get("dataset_id"),
                "source_id": decision.get("source_id"),
                "batch_id": decision.get("batch_id"),
                "verdict": decision.get("verdict"),
                "batch_score": decision.get("batch_score"),
                "confidence": decision.get("confidence"),
                "evidence_strength": decision.get("evidence_strength"),
                "fitness_policy": (
                    decision.get("fitness_policy", {})
                    .get("policy", {})
                    .get("policy_id")
                    if isinstance(
                        decision.get("fitness_policy"),
                        Mapping,
                    )
                    else None
                ),
                "flags": deepcopy(decision.get("flags", [])),
                "quarantine_count": decision.get("quarantine_count", 0),
                "route_count": len(decision.get("route_records", [])),
                "updated_at": time.time(),
            }

            if self.include_record_previews_in_shared_memory:
                summary_payload["record_preview"] = deepcopy(
                    list(records[: self.max_shared_record_preview])
                )

            shared_memory_set(
                self.shared_memory,
                result_key,
                deepcopy(dict(decision)),
                ttl=self.shared_ttl_seconds,
            )
            shared_memory_set(
                self.shared_memory,
                summary_key,
                summary_payload,
                ttl=self.shared_ttl_seconds,
            )
            shared_keys.extend([result_key, summary_key])

            if self.publish_notifications:
                shared_memory_publish(
                    self.shared_memory,
                    self.shared_event_channel,
                    {
                        "event_type": "quality_agent_decision",
                        "decision": deepcopy(dict(decision)),
                    },
                )

            return shared_keys

        except Exception as exc:
            normalized = normalize_quality_exception(
                exc,
                stage=QualityStage.PERSISTENCE,
                context={
                    "operation": "publish_result_to_shared_memory",
                    "batch_id": decision.get("batch_id"),
                },
                error_type=QualityErrorType.QUALITY_MEMORY_UNAVAILABLE,
                severity=QualitySeverity.MEDIUM,
                retryable=True,
                remediation=(
                    "Retry shared-memory publication or inspect the "
                    "shared-memory backend."
                ),
                disposition=QualityDisposition.WARN,
            )
            normalized.report()
            self._store_error_marker(normalized.to_dict())
            return shared_keys

    def _persist_local_result(
        self,
        decision: Mapping[str, Any],
    ) -> None:
        batch_key = str(decision["batch_id"])
        source_key = str(decision["source_id"])
        self._results_by_batch[batch_key] = deepcopy(dict(decision))
        history = self._results_by_source.setdefault(source_key, [])
        history.append(deepcopy(dict(decision)))
        max_history = max(self.max_error_log_size, 10)
        if len(history) > max_history:
            del history[: len(history) - max_history]

    def _store_error_marker(
        self,
        error_payload: Mapping[str, Any],
    ) -> None:
        if (
            self.shared_memory is None
            or not self.publish_to_shared_memory
        ):
            return
        try:
            key = (
                f"{self.shared_error_key_prefix}:"
                f"{uuid.uuid4().hex}"
            )
            shared_memory_set(
                self.shared_memory,
                key,
                deepcopy(dict(error_payload)),
                ttl=self.shared_ttl_seconds,
            )
        except Exception:
            # Error-marker publication must never mask the originating
            # quality failure.
            return

    # ------------------------------------------------------------------
    # Bridge resolution
    # ------------------------------------------------------------------
    def _resolve_bridge(self, kind: str) -> Any:
        if kind == "handler" and self.handler_bridge is not None:
            return self.handler_bridge
        if kind == "safety" and self.safety_bridge is not None:
            return self.safety_bridge

        if kind not in {"handler", "safety"}:
            raise ValueError(f"Unsupported quality bridge kind '{kind}'")

        should_resolve = (
            self.resolve_handler_from_factory
            if kind == "handler"
            else self.resolve_safety_from_factory
        )
        if not should_resolve or self.agent_factory is None:
            return None

        candidates = (
            self.handler_factory_names
            if kind == "handler"
            else self.safety_factory_names
        )
        for candidate in candidates:
            try:
                bridge = resolve_factory_candidate(
                    self.agent_factory,
                    candidate,
                )
            except Exception as exc:
                normalized = normalize_quality_exception(
                    exc,
                    stage=QualityStage.ROUTING,
                    context={
                        "operation": "resolve_bridge",
                        "candidate": candidate,
                        "kind": kind,
                    },
                    error_type=QualityErrorType.ROUTING_FAILED,
                    severity=QualitySeverity.MEDIUM,
                    retryable=True,
                    remediation=(
                        "Review AgentFactory resolution or attach the "
                        "runtime bridge explicitly."
                    ),
                    disposition=QualityDisposition.WARN,
                )
                normalized.report()
                self._store_error_marker(normalized.to_dict())
                continue

            if bridge is not None:
                if kind == "handler":
                    self.handler_bridge = bridge
                else:
                    self.safety_bridge = bridge
                return bridge

        return None

    # ------------------------------------------------------------------
    # Validation / small helpers
    # ------------------------------------------------------------------
    def _validate_runtime_configuration(self) -> None:
        if self.warn_threshold > self.pass_threshold:
            raise DataQualityError(
                message=(
                    "quality_agent.warn_threshold must be <= "
                    "quality_agent.pass_threshold"
                ),
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.ESCALATE,
                context={
                    "warn_threshold": self.warn_threshold,
                    "pass_threshold": self.pass_threshold,
                },
                remediation=(
                    "Correct QualityAgent thresholds in agents_config.yaml."
                ),
            )

        if set(self.subsystem_order) != set(_SUBSYSTEMS):
            raise DataQualityError(
                message=(
                    "quality_agent.subsystem_order must contain each of "
                    "structural, statistical, and semantic exactly once"
                ),
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.ESCALATE,
                context={"subsystem_order": self.subsystem_order},
                remediation=(
                    "Correct quality_agent.subsystem_order in "
                    "agents_config.yaml."
                ),
            )

        if self.intelligence_enabled:
            for required in (
                "fitness_policy",
                "baseline_governance",
                "relationship_quality",
                "evidence_calibration",
            ):
                self._mapping_section(
                    self.intelligence_config,
                    required,
                    required=True,
                )

    def _disabled_decision(
        self,
        *,
        dataset_id: str,
        source_id: str,
        batch_id: str,
        context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        decision = QualityAgentDecision(
            decision_id=generate_quality_id("quality_decision"),
            dataset_id=dataset_id,
            source_id=source_id,
            batch_id=batch_id,
            verdict="pass",
            batch_score=1.0,
            subsystem_scores={},
            subsystem_verdicts={},
            flags=["quality_agent_disabled"],
            remediation_actions=[],
            quarantine_count=0,
            subsystem_results={},
            workflow_decision={},
            route_records=[],
            quarantine_entries=[],
            shared_memory_keys=[],
            context=dict(context),
            created_at=time.time(),
        )
        payload = decision.to_dict()
        payload["quality_verdict"] = "pass"
        payload["record_level_flags"] = {}
        payload["confidence"] = 1.0
        payload["uncertainty"] = 0.0
        payload["evidence_strength"] = "not_applicable"
        return payload

    def _error_finding(
        self,
        *,
        subsystem: str,
        error: DataQualityError,
    ) -> Dict[str, Any]:
        payload = error.to_dict()
        failure_verdict = (
            "block" if self.fail_closed_on_subsystem_error else "warn"
        )
        return {
            "checker": subsystem,
            "domain": subsystem,
            "verdict": failure_verdict,
            "severity": payload.get("severity", "high"),
            "confidence": 1.0,
            "message": payload.get(
                "message",
                f"{subsystem} subsystem failure",
            ),
            "flags": [f"{subsystem}_subsystem_failure"],
            "affected_records": [],
            "remediation_actions": (
                [payload.get("remediation")]
                if payload.get("remediation")
                else []
            ),
            "error_type": payload.get(
                "error_type",
                QualityErrorType.INTERNAL_QUALITY_AGENT_FAILURE.value,
            ),
            "error": payload,
        }

    @staticmethod
    def _summary_error_type(
        subsystem: str,
        verdict: str,
    ) -> str:
        normalized = normalize_verdict(verdict)
        if normalized != "block":
            return ""
        if subsystem == "structural":
            return QualityErrorType.SCHEMA_VALIDATION_FAILED.value
        if subsystem == "statistical":
            return QualityErrorType.DISTRIBUTION_DRIFT_DETECTED.value
        if subsystem == "semantic":
            return QualityErrorType.CROSS_FIELD_CONFLICT.value
        return ""

    @staticmethod
    def _summary_severity_for_verdict(verdict: str) -> str:
        normalized = normalize_verdict(verdict)
        if normalized == "block":
            return "high"
        if normalized == "warn":
            return "medium"
        return "low"


__all__ = [
    "QualityAgent",
    "QualityAgentDecision",
    "SubsystemExecution",
]


if __name__ == "__main__":
    print("\n=== Running  Data quality agent ===\n")
    printer.status("TEST", " Data quality agent initialized", "info")
    from .collaborative.shared_memory import SharedMemory
    from .agent_factory import AgentFactory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()
    handler_bridge = agent_factory.create("handler", shared_memory)
    safety_bridge = agent_factory.create("safety", shared_memory)

    agent = QualityAgent(
        shared_memory=shared_memory,
        agent_factory=agent_factory,
        handler_bridge=handler_bridge,
        safety_bridge=safety_bridge,
        config={
            "publish_to_shared_memory": True,
            "auto_route_via_workflow": True,
        },
    )

    printer.status("CONFIG", f"Loaded quality_agent config from {agent.config.get('__config_path__', 'unknown')}", "success")

    schema = {
        "schema_version": "v2.3.0",
        "required_fields": ["id", "text", "label", "score", "status", "source_id", "source_type", "collected_at"],
        "fields": {
            "id": {"type": "str", "required": True},
            "text": {"type": "str", "required": True},
            "label": {"type": "str", "required": True},
            "score": {"type": "float", "required": True, "minimum": 0.0, "maximum": 1.0},
            "status": {"type": "str", "allowed_values": ["open", "resolved"]},
            "source_id": {"type": "str", "required": True},
            "source_type": {"type": "str", "required": True},
            "collected_at": {"type": "str", "required": True},
        },
    }

    records = [
        {
            "id": "rec_001",
            "text": "customer asked for refund",
            "label": "refund",
            "score": 0.91,
            "status": "resolved",
            "resolved_at": "2026-04-09T10:00:00",
            "source_id": "support_portal",
            "source_type": "ticketing",
            "collected_at": "2026-04-09T10:01:00",
        },
        {
            "id": "rec_002",
            "text": "refund request waiting for review",
            "label": "refund",
            "score": 0.88,
            "status": "open",
            "source_id": "support_portal",
            "source_type": "ticketing",
            "collected_at": "2026-04-09T10:01:30",
        },
        {
            "id": "rec_003",
            "text": "chargeback dispute escalated",
            "label": "chargeback",
            "score": 0.77,
            "status": "open",
            "source_id": "support_portal",
            "source_type": "ticketing",
            "collected_at": "2026-04-09T10:02:00",
        },
    ]

    provenance = {
        "source_id": "support_portal",
        "source_type": "ticketing",
        "collected_at": "2026-04-09T10:05:00",
        "checksum": "abc123",
        "lineage_id": "lineage_demo_001",
        "collector": "reader_agent",
        "schema_version": "v2.0.0",
    }

    result = agent.evaluate_batch(
        records,
        dataset_id="customer_support_quality",
        source_id="support_portal",
        batch_id="batch_quality_agent_demo_001",
        schema=schema,
        label_field="label",
        feature_fields=["text", "status"],
        provenance=provenance,
        source_metadata={"source_id": "support_portal", "source_type": "ticketing"},
        context={"route": "reader->quality->knowledge_ingestion", "task_type": "knowledge_ingestion"},
    )
    printer.pretty("QUALITY_AGENT_RESULT", result, "success")
    printer.pretty("QUALITY_AGENT_SUMMARY", agent.summary(), "success")
    printer.pretty("SHARED_MEMORY_KEYS", shared_memory.get_all_keys(), "info")
    printer.pretty("SHARED_SUBSCRIBERS", shared_memory.subscribers, "info") 
    printer.pretty("HANDLER_CALLS", handler_bridge.calls, "info")
    printer.pretty("SAFETY_CALLS", safety_bridge.calls, "info")

    print("\n=== Test ran successfully ===\n")

