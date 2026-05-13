"""
The Data Quality Agent is the quality gatekeeper between raw inputs and all downstream components (knowledge ingestion, training loops, inference context, and memory updates).
It continuously measures and enforces data integrity, consistency, and fitness-for-use.

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

__version__ = "2.1.0"

import time
import uuid

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .base_agent import BaseAgent
from .base.utils.config_contract import assert_valid_config_contract
from .base.utils.main_config_loader import load_global_config, get_config_section
from .quality import SemanticQuality, StatisticalQuality, StructuralQuality, WorkflowControl
from .quality.utils.quality_error import (DataQualityError, DataQualityErrorGroup, QualityStage,
                                          QualityDisposition, QualityDomain, QualityErrorType,
                                          QualitySeverity, normalize_quality_exception, quality_error_boundary)
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("Quality Agent")
printer = PrettyPrinter()


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

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["created_at_iso"] = datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat()
        return payload


class QualityAgent(BaseAgent):
    """Owns integrity checks and quarantine policy prior to downstream use.

    The agent is intentionally orchestration-heavy. The structural, statistical,
    semantic, and workflow subsystems already own their local quality logic. The
    role of ``QualityAgent`` is to:
    - normalize incoming tasks into a stable batch-evaluation contract,
    - execute subsystem checks in a policy-aware order,
    - aggregate subsystem signals into a single quality posture,
    - delegate operational decisions to WorkflowControl,
    - publish stable decision artifacts to shared memory for sibling agents.

    Architectural note:
    ``WorkflowControl`` remains dependency-light and bridge-driven. The parent
    ``QualityAgent`` owns the runtime wiring for shared memory, handler bridges,
    and safety bridges. This prevents circular imports, repeated initialization,
    and the kind of hidden cross-agent coupling that makes production debugging
    difficult.
    """

    def __init__(self, shared_memory, agent_factory, config=None, **kwargs):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
        self.shared_memory = shared_memory or self.shared_memory
        self.agent_factory = agent_factory

        self.config = load_global_config()
        self.quality_config = get_config_section("quality_agent")
        if config:
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

        self.enabled = bool(self.quality_config.get("enabled", True))
        self.default_window = str(self.quality_config.get("default_window", "latest")).strip() or "latest"
        self.stop_on_blocking_structural = bool(self.quality_config.get("stop_on_blocking_structural", True))
        self.fail_closed_on_subsystem_error = bool(self.quality_config.get("fail_closed_on_subsystem_error", True))
        self.auto_route_via_workflow = bool(self.quality_config.get("auto_route_via_workflow", True))
        self.prefer_workflow_verdict = bool(self.quality_config.get("prefer_workflow_verdict", True))
        self.include_workflow_findings_in_summary = bool(
            self.quality_config.get("include_workflow_findings_in_summary", True)
        )
        self.include_record_previews_in_shared_memory = bool(
            self.quality_config.get("include_record_previews_in_shared_memory", False)
        )
        self.max_shared_record_preview = self._positive_int(
            self.quality_config.get("max_shared_record_preview", 5),
            "quality_agent.max_shared_record_preview",
        )

        self.pass_threshold = self._bounded_score(
            self.quality_config.get("pass_threshold", 0.90),
            field_name="quality_agent.pass_threshold",
        )
        self.warn_threshold = self._bounded_score(
            self.quality_config.get("warn_threshold", 0.75),
            field_name="quality_agent.warn_threshold",
        )
        self.subsystem_order = self._string_list(
            self.quality_config.get("subsystem_order", ["structural", "statistical", "semantic"])
        )
        self.subsystem_weights = self._normalized_weights(
            self.quality_config.get(
                "subsystem_weights",
                {"structural": 0.34, "statistical": 0.33, "semantic": 0.33},
            )
        )

        self.shared_memory_config = dict(self.quality_config.get("shared_memory", {}))
        self.publish_to_shared_memory = bool(self.shared_memory_config.get("enabled", True))
        self.publish_notifications = bool(self.shared_memory_config.get("publish_notifications", True))
        self.shared_ttl_seconds = self._optional_nonnegative_int(
            self.shared_memory_config.get("ttl_seconds", 86400),
            "quality_agent.shared_memory.ttl_seconds",
        )
        self.shared_result_key_prefix = str(
            self.shared_memory_config.get("result_key_prefix", "quality_agent.result")
        ).strip() or "quality_agent.result"
        self.shared_summary_key_prefix = str(
            self.shared_memory_config.get("summary_key_prefix", "quality_agent.summary")
        ).strip() or "quality_agent.summary"
        self.shared_error_key_prefix = str(
            self.shared_memory_config.get("error_key_prefix", "quality_agent.error")
        ).strip() or "quality_agent.error"
        self.shared_event_channel = str(
            self.shared_memory_config.get("event_channel", "quality.events")
        ).strip() or "quality.events"

        bridge_resolution = dict(self.quality_config.get("bridge_resolution", {}))
        self.resolve_handler_from_factory = bool(bridge_resolution.get("resolve_handler_from_factory", False))
        self.resolve_safety_from_factory = bool(bridge_resolution.get("resolve_safety_from_factory", False))
        self.handler_factory_names = self._string_list(
            bridge_resolution.get("handler_factory_names", ["handler_agent", "handler", "HandlerAgent"])
        )
        self.safety_factory_names = self._string_list(
            bridge_resolution.get("safety_factory_names", ["safety_agent", "safety", "SafetyAgent"])
        )

        self.handler_bridge = kwargs.get("handler_bridge")
        self.safety_bridge = kwargs.get("safety_bridge")
        self._results_by_batch: Dict[str, Dict[str, Any]] = {}
        self._results_by_source: Dict[str, List[Dict[str, Any]]] = {}

        self._validate_runtime_configuration()

        # subsystem initialization
        self.structural_quality = StructuralQuality()
        self.statistical_quality = StatisticalQuality()
        self.semantic_quality = SemanticQuality()
        self.workflow_control = WorkflowControl(
            shared_memory=self.shared_memory,
            handler_bridge=self.handler_bridge,
            safety_bridge=self.safety_bridge,
        )
        self.attach_runtime(
            shared_memory=self.shared_memory,
            handler_bridge=self.handler_bridge,
            safety_bridge=self.safety_bridge,
        )

        logger.info(
            "Quality Agent initialized | enabled=%s | auto_route_via_workflow=%s | subsystem_order=%s",
            self.enabled,
            self.auto_route_via_workflow,
            self.subsystem_order,
        )

    # ------------------------------------------------------------------
    # Runtime wiring
    # ------------------------------------------------------------------
    def attach_runtime(self, *,
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
    # Public orchestration APIs
    # ------------------------------------------------------------------
    def evaluate_batch(self, records: Sequence[Mapping[str, Any]], *,
        dataset_id: str,
        source_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        schema: Optional[Mapping[str, Any]] = None,
        baseline: Optional[Mapping[str, Any]] = None,
        label_field: Optional[str] = None,
        feature_fields: Optional[Sequence[str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        source_metadata: Optional[Mapping[str, Any]] = None,
        window: Optional[str] = None,
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
            remediation="Review upstream batch payloads and rerun the quality gate with normalized inputs.",
            disposition=QualityDisposition.BLOCK,
        ):
            if not self.enabled:
                dataset_key = self._nonempty(dataset_id, "dataset_id")
                source_key = self._nonempty(source_id or dataset_key, "source_id")
                batch_key = self._nonempty(batch_id or self._new_id("batch"), "batch_id")
                disabled = QualityAgentDecision(
                    decision_id=self._new_id("quality_decision"),
                    dataset_id=dataset_key,
                    source_id=source_key,
                    batch_id=batch_key,
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
                    context=self._normalized_mapping(context),
                    created_at=time.time(),
                )
                return disabled.to_dict()

            normalized_records = self._normalize_records(records)
            if not normalized_records:
                raise DataQualityError(
                    message="QualityAgent requires at least one record for batch evaluation",
                    error_type=QualityErrorType.SCHEMA_VALIDATION_FAILED,
                    severity=QualitySeverity.HIGH,
                    retryable=False,
                    stage=QualityStage.VALIDATION,
                    domain=QualityDomain.SYSTEM,
                    disposition=QualityDisposition.BLOCK,
                    dataset_id=dataset_id,
                    source_id=source_id,
                    batch_id=batch_id,
                    remediation="Provide a non-empty normalized batch before invoking the quality gate.",
                )

            dataset_key = self._nonempty(dataset_id, "dataset_id")
            source_key = self._nonempty(source_id or dataset_key, "source_id")
            batch_key = self._nonempty(batch_id or self._new_id("batch"), "batch_id")
            resolved_window = str(window or self.default_window)
            normalized_context = self._normalized_mapping(context)
            normalized_workflow_context = self._normalized_mapping(workflow_context)
            runtime_handler = handler_bridge if handler_bridge is not None else self._resolve_bridge("handler")
            runtime_safety = safety_bridge if safety_bridge is not None else self._resolve_bridge("safety")
            self.attach_runtime(
                shared_memory=self.shared_memory,
                handler_bridge=runtime_handler,
                safety_bridge=runtime_safety,
            )

            subsystem_results: Dict[str, Dict[str, Any]] = {}
            subsystem_errors: List[DataQualityError] = []

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
            if structural_result.error is not None:
                subsystem_errors.append(self._error_from_payload(structural_result.error))

            if self.stop_on_blocking_structural and structural_result.verdict == "block":
                logger.warning(
                    "Structural quality blocked batch '%s'; statistical and semantic checks were skipped by policy.",
                    batch_key,
                )
                statistical_result = self._skipped_subsystem_result(
                    subsystem="statistical",
                    reason="Skipped because structural quality returned block and stop_on_blocking_structural is enabled.",
                    source_id=source_key,
                    batch_id=batch_key,
                )
                semantic_result = self._skipped_subsystem_result(
                    subsystem="semantic",
                    reason="Skipped because structural quality returned block and stop_on_blocking_structural is enabled.",
                    source_id=source_key,
                    batch_id=batch_key,
                )
            else:
                required_fields = self._schema_required_fields(schema)
                schema_version = self._schema_version(schema, structural_result.raw_result)
                statistical_result = self._execute_statistical(
                    records=normalized_records,
                    source_id=source_key,
                    batch_id=batch_key,
                    baseline=baseline,
                    required_fields=required_fields,
                    window=resolved_window,
                    schema_version=schema_version,
                    context=normalized_context,
                )
                semantic_result = self._execute_semantic(
                    records=normalized_records,
                    source_id=source_key,
                    batch_id=batch_key,
                    label_field=label_field,
                    feature_fields=feature_fields,
                    provenance=provenance,
                    source_metadata=source_metadata,
                    schema_version=schema_version,
                    window=resolved_window,
                    context=normalized_context,
                )
                if statistical_result.error is not None:
                    subsystem_errors.append(self._error_from_payload(statistical_result.error))
                if semantic_result.error is not None:
                    subsystem_errors.append(self._error_from_payload(semantic_result.error))

            subsystem_results["statistical"] = statistical_result.to_dict()
            subsystem_results["semantic"] = semantic_result.to_dict()

            if subsystem_errors and self.fail_closed_on_subsystem_error:
                raise DataQualityErrorGroup(
                    message=f"One or more quality subsystems failed while evaluating batch '{batch_key}'",
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
                    remediation="Review subsystem-specific failures and rerun the batch after correcting the broken stage.",
                )

            combined_findings = self._combine_findings(subsystem_results)
            aggregate_score = self._aggregate_score(subsystem_results)
            preliminary_verdict = self._aggregate_verdict(subsystem_results, batch_score=aggregate_score)
            flags = self._merge_unique_strings(
                subsystem_results["structural"].get("flags"),
                subsystem_results["statistical"].get("flags"),
                subsystem_results["semantic"].get("flags"),
                [f"quality_agent:{preliminary_verdict}"],
            )
            remediation_actions = self._merge_unique_strings(
                subsystem_results["structural"].get("remediation_actions"),
                subsystem_results["statistical"].get("remediation_actions"),
                subsystem_results["semantic"].get("remediation_actions"),
            )

            workflow_payload_context = {
                "dataset_id": dataset_key,
                "source_id": source_key,
                "batch_id": batch_key,
                "window": resolved_window,
                **normalized_context,
                **normalized_workflow_context,
            }
            workflow_decision = self.workflow_control.coordinate_batch(
                source_id=source_key,
                batch_id=batch_key,
                findings=combined_findings,
                batch_score=aggregate_score,
                records=normalized_records,
                context=workflow_payload_context,
                shared_memory=self.shared_memory,
                handler_bridge=runtime_handler,
                safety_bridge=runtime_safety,
            ) if self.auto_route_via_workflow else {}

            final_verdict = str(
                workflow_decision.get("verdict") if self.prefer_workflow_verdict and workflow_decision else preliminary_verdict
            )
            final_score = float(workflow_decision.get("batch_score", aggregate_score)) if workflow_decision else aggregate_score
            final_flags = self._merge_unique_strings(
                flags,
                workflow_decision.get("flags") if self.include_workflow_findings_in_summary else [],
            )
            final_remediation = self._merge_unique_strings(
                remediation_actions,
                workflow_decision.get("remediation_plan", {}).get("actions") if workflow_decision else [],
            )
            quarantine_entries = list(workflow_decision.get("quarantine_entries", [])) if workflow_decision else []
            route_records = list(workflow_decision.get("route_records", [])) if workflow_decision else []

            decision = QualityAgentDecision(
                decision_id=self._new_id("quality_decision"),
                dataset_id=dataset_key,
                source_id=source_key,
                batch_id=batch_key,
                verdict=self._normalize_verdict(final_verdict),
                batch_score=self._bounded_score(final_score, field_name="quality_agent.batch_score"),
                subsystem_scores={
                    name: float(result.get("batch_score", 0.0))
                    for name, result in subsystem_results.items()
                },
                subsystem_verdicts={
                    name: self._normalize_verdict(result.get("verdict", "warn"))
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
                },
                created_at=time.time(),
            )

            decision_dict = decision.to_dict()
            decision_dict["quality_verdict"] = decision_dict["verdict"]
            decision_dict["record_level_flags"] = self._record_level_flags(subsystem_results)
            decision_dict["confidence"] = self._decision_confidence(subsystem_results)
            decision_dict["quarantine_queue_entries"] = deepcopy(quarantine_entries)
            decision_dict["subsystem_findings"] = deepcopy(combined_findings)

            decision_dict["shared_memory_keys"] = self._publish_result_to_shared_memory(
                decision=decision_dict,
                records=normalized_records,
            )
            self._persist_local_result(decision_dict)
            return decision_dict

    def assess_batch(self, records: Sequence[Mapping[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        return self.evaluate_batch(records, **kwargs)

    def latest_decision(self, batch_id: str) -> Optional[Dict[str, Any]]:
        batch_key = self._nonempty(batch_id, "batch_id")
        result = self._results_by_batch.get(batch_key)
        return None if result is None else deepcopy(result)

    def recent_source_results(self, source_id: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        source_key = self._nonempty(source_id, "source_id")
        return deepcopy((self._results_by_source.get(source_key) or [])[-max(int(limit), 1):])

    def summary(self) -> Dict[str, Any]:
        all_results = list(self._results_by_batch.values())
        verdict_counts = {"pass": 0, "warn": 0, "block": 0}
        for result in all_results:
            verdict = str(result.get("verdict", "warn"))
            if verdict in verdict_counts:
                verdict_counts[verdict] += 1
        return {
            "enabled": self.enabled,
            "results_tracked": len(all_results),
            "sources_tracked": len(self._results_by_source),
            "verdict_counts": verdict_counts,
            "subsystem_order": list(self.subsystem_order),
            "auto_route_via_workflow": self.auto_route_via_workflow,
            "publish_to_shared_memory": self.publish_to_shared_memory,
        }

    def perform_task(self, task_data: Any) -> Dict[str, Any]:
        with quality_error_boundary(
            stage=QualityStage.ROUTING,
            context={"operation": "perform_task"},
            error_type=QualityErrorType.SCORING_PIPELINE_FAILED,
            severity=QualitySeverity.HIGH,
            retryable=False,
            remediation="Provide a valid quality-agent task payload and rerun the agent.",
            disposition=QualityDisposition.BLOCK,
        ):
            if not isinstance(task_data, Mapping):
                raise DataQualityError(
                    message="QualityAgent.perform_task expects a mapping payload",
                    error_type=QualityErrorType.CONFIGURATION_INVALID,
                    severity=QualitySeverity.MEDIUM,
                    retryable=False,
                    stage=QualityStage.ROUTING,
                    domain=QualityDomain.SYSTEM,
                    disposition=QualityDisposition.WARN,
                    remediation="Wrap the batch payload in a mapping with records, identifiers, and optional context.",
                    context={"task_data_type": type(task_data).__name__},
                )

            operation = str(task_data.get("operation", "evaluate_batch")).strip().lower()
            payload = task_data.get("task_data", task_data.get("input_data", task_data.get("payload", task_data)))
            payload_map = payload if isinstance(payload, Mapping) else {"records": payload}
            context = payload_map.get("context", task_data.get("context", {}))

            if operation in {"summary", "status"}:
                return self.summary()
            if operation in {"latest_decision", "get_latest_decision"}:
                batch_id = task_data.get("batch_id") or payload_map.get("batch_id")
                if not batch_id:
                    return {}
                return self.latest_decision(str(batch_id)) or {}
            if operation in {"workflow_only", "route_quality", "coordinate_batch"}:
                return self.workflow_control.coordinate_batch(
                    source_id=self._nonempty(payload_map.get("source_id"), "source_id"),
                    batch_id=self._nonempty(payload_map.get("batch_id"), "batch_id"),
                    findings=payload_map.get("findings"),
                    batch_score=payload_map.get("batch_score"),
                    records=payload_map.get("records"),
                    context=context,
                    shared_memory=self.shared_memory,
                    handler_bridge=self._resolve_bridge("handler"),
                    safety_bridge=self._resolve_bridge("safety"),
                )

            return self.evaluate_batch(
                records=self._normalize_records(payload_map.get("records", [])),
                dataset_id=self._nonempty(payload_map.get("dataset_id") or payload_map.get("source_id"), "dataset_id"),
                source_id=payload_map.get("source_id"),
                batch_id=payload_map.get("batch_id"),
                schema=payload_map.get("schema"),
                baseline=payload_map.get("baseline"),
                label_field=payload_map.get("label_field"),
                feature_fields=payload_map.get("feature_fields"),
                provenance=payload_map.get("provenance"),
                source_metadata=payload_map.get("source_metadata"),
                window=payload_map.get("window"),
                context=context,
                workflow_context=payload_map.get("workflow_context"),
                handler_bridge=payload_map.get("handler_bridge"),
                safety_bridge=payload_map.get("safety_bridge"),
            )

    # ------------------------------------------------------------------
    # Subsystem execution
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
        started = time.time()
        try:
            result = fn(**fn_kwargs)
            normalized = self._normalized_mapping(result if isinstance(result, Mapping) else {"result": result})
            duration_ms = max((time.time() - started) * 1000.0, 0.0)
            return SubsystemExecution(
                subsystem=subsystem,
                enabled=True,
                verdict=self._normalize_verdict(normalized.get("verdict", "warn")),
                batch_score=self._bounded_score(
                    normalized.get("batch_score", self._score_from_verdict(normalized.get("verdict", "warn"))),
                    field_name=f"{subsystem}.batch_score",
                ),
                findings=self._normalize_subsystem_findings(normalized.get("findings", []), subsystem=subsystem),
                flags=self._string_list(normalized.get("flags")),
                remediation_actions=self._string_list(normalized.get("remediation_actions")),
                quarantine_count=self._optional_nonnegative_int(normalized.get("quarantine_count", 0), f"{subsystem}.quarantine_count") or 0,
                shift_metrics=self._float_mapping(normalized.get("shift_metrics", {})),
                context={
                    "source_id": source_id,
                    "batch_id": batch_id,
                    "record_count": normalized.get("record_count") or normalized.get("reviewed_record_count"),
                },
                duration_ms=duration_ms,
                raw_result=normalized,
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
                remediation=f"Repair the {subsystem} subsystem failure and rerun the batch evaluation.",
                disposition=QualityDisposition.BLOCK,
            )
            normalized.report()
            duration_ms = max((time.time() - started) * 1000.0, 0.0)
            return SubsystemExecution(
                subsystem=subsystem,
                enabled=True,
                verdict="block" if self.fail_closed_on_subsystem_error else "warn",
                batch_score=0.0,
                findings=[self._error_finding(subsystem=subsystem, error=normalized)],
                flags=[f"{subsystem}_subsystem_failure"],
                remediation_actions=[normalized.remediation] if normalized.remediation else [],
                quarantine_count=0,
                shift_metrics={},
                context={"source_id": source_id, "batch_id": batch_id},
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
                    "remediation_actions": [],
                    "error_type": "",
                }
            ],
            flags=[f"{subsystem}_skipped"],
            remediation_actions=[],
            quarantine_count=0,
            shift_metrics={},
            context={"source_id": source_id, "batch_id": batch_id},
            duration_ms=0.0,
            raw_result={"skipped": True, "reason": reason},
            error=None,
        )

    # ------------------------------------------------------------------
    # Aggregation and publication helpers
    # ------------------------------------------------------------------
    def _combine_findings(self, subsystem_results: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        for subsystem in self.subsystem_order:
            result = subsystem_results.get(subsystem) or {}
            findings.extend(self._normalize_subsystem_findings(result.get("findings", []), subsystem=subsystem))
            findings.append(
                {
                    "checker": subsystem,
                    "domain": subsystem,
                    "verdict": self._normalize_verdict(result.get("verdict", "warn")),
                    "severity": self._summary_severity_for_verdict(result.get("verdict", "warn")),
                    "confidence": self._decision_confidence_for_score(float(result.get("batch_score", 0.0))),
                    "message": f"{subsystem} subsystem summary verdict={result.get('verdict', 'warn')} score={result.get('batch_score', 0.0)}",
                    "flags": self._string_list(result.get("flags")),
                    "remediation_actions": self._string_list(result.get("remediation_actions")),
                    "error_type": self._summary_error_type(subsystem, result.get("verdict", "warn")),
                    "subsystem_summary": True,
                }
            )
        return findings

    def _aggregate_score(self, subsystem_results: Mapping[str, Mapping[str, Any]]) -> float:
        weighted_sum = 0.0
        total_weight = 0.0
        for subsystem in self.subsystem_order:
            result = subsystem_results.get(subsystem) or {}
            if not result:
                continue
            weight = float(self.subsystem_weights.get(subsystem, 0.0))
            if weight <= 0:
                continue
            weighted_sum += weight * self._bounded_score(
                result.get("batch_score", self._score_from_verdict(result.get("verdict", "warn"))),
                field_name=f"quality_agent.subsystem_score.{subsystem}",
            )
            total_weight += weight
        if total_weight <= 0:
            return 0.0
        return self._bounded_score(weighted_sum / total_weight, field_name="quality_agent.aggregate_score")

    def _aggregate_verdict(self, subsystem_results: Mapping[str, Mapping[str, Any]], *, batch_score: float) -> str:
        verdicts = [self._normalize_verdict((subsystem_results.get(name) or {}).get("verdict", "warn")) for name in self.subsystem_order]
        if "block" in verdicts:
            return "block"
        if batch_score >= self.pass_threshold and all(verdict == "pass" for verdict in verdicts):
            return "pass"
        if batch_score >= self.warn_threshold:
            return "warn"
        return "block"

    def _publish_result_to_shared_memory(
        self,
        *,
        decision: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> List[str]:
        if not self.publish_to_shared_memory or self.shared_memory is None:
            return []

        shared_keys: List[str] = []
        try:
            result_key = f"{self.shared_result_key_prefix}:{decision['batch_id']}"
            summary_key = f"{self.shared_summary_key_prefix}:{decision['source_id']}"
            summary_payload = {
                "dataset_id": decision.get("dataset_id"),
                "source_id": decision.get("source_id"),
                "batch_id": decision.get("batch_id"),
                "verdict": decision.get("verdict"),
                "batch_score": decision.get("batch_score"),
                "flags": deepcopy(decision.get("flags", [])),
                "quarantine_count": decision.get("quarantine_count", 0),
                "route_count": len(decision.get("route_records", [])),
                "updated_at": time.time(),
            }
            if self.include_record_previews_in_shared_memory:
                summary_payload["record_preview"] = deepcopy(list(records[: self.max_shared_record_preview]))

            self._shared_set(result_key, deepcopy(dict(decision)), ttl=self.shared_ttl_seconds)
            self._shared_set(summary_key, summary_payload, ttl=self.shared_ttl_seconds)
            shared_keys.extend([result_key, summary_key])

            if self.publish_notifications:
                self._shared_publish(
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
                context={"operation": "publish_result_to_shared_memory", "batch_id": decision.get("batch_id")},
                error_type=QualityErrorType.QUALITY_MEMORY_UNAVAILABLE,
                severity=QualitySeverity.MEDIUM,
                retryable=True,
                remediation="Retry shared-memory publication or inspect the shared-memory backend health.",
                disposition=QualityDisposition.WARN,
            )
            normalized.report()
            self._store_error_marker(normalized.to_dict())
            return shared_keys

    def _persist_local_result(self, decision: Mapping[str, Any]) -> None:
        batch_key = str(decision["batch_id"])
        source_key = str(decision["source_id"])
        self._results_by_batch[batch_key] = deepcopy(dict(decision))
        history = self._results_by_source.setdefault(source_key, [])
        history.append(deepcopy(dict(decision)))
        max_history = max(self.max_error_log_size, 10)
        if len(history) > max_history:
            del history[: len(history) - max_history]

    def _resolve_bridge(self, kind: str) -> Any:
        if kind == "handler" and self.handler_bridge is not None:
            return self.handler_bridge
        if kind == "safety" and self.safety_bridge is not None:
            return self.safety_bridge

        should_resolve = self.resolve_handler_from_factory if kind == "handler" else self.resolve_safety_from_factory
        if not should_resolve or self.agent_factory is None:
            return None

        candidates = self.handler_factory_names if kind == "handler" else self.safety_factory_names
        for candidate in candidates:
            bridge = self._resolve_from_factory(candidate)
            if bridge is not None:
                if kind == "handler":
                    self.handler_bridge = bridge
                else:
                    self.safety_bridge = bridge
                return bridge
        return None

    def _resolve_from_factory(self, candidate: str) -> Any:
        factory = self.agent_factory
        try:
            if callable(factory):
                return factory(candidate)
            for method_name in ("get", "create", "create_agent", "build", "resolve"):
                method = getattr(factory, method_name, None)
                if callable(method):
                    try:
                        return method(candidate)
                    except TypeError:
                        continue
        except Exception as exc:
            normalized = normalize_quality_exception(
                exc,
                stage=QualityStage.ROUTING,
                context={"operation": "resolve_bridge", "candidate": candidate},
                error_type=QualityErrorType.ROUTING_FAILED,
                severity=QualitySeverity.MEDIUM,
                retryable=True,
                remediation="Review agent_factory resolution rules or attach the runtime bridges explicitly.",
                disposition=QualityDisposition.WARN,
            )
            normalized.report()
            self._store_error_marker(normalized.to_dict())
        return None

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------
    def _normalize_subsystem_findings(
        self,
        findings: Iterable[Mapping[str, Any]],
        *,
        subsystem: str,
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in findings or []:
            finding = self._normalized_mapping(item)
            finding["checker"] = str(finding.get("checker") or finding.get("check_name") or finding.get("check") or subsystem)
            finding["domain"] = str(finding.get("domain") or subsystem)
            finding["verdict"] = self._normalize_verdict(finding.get("verdict", "warn"))
            finding["severity"] = self._normalize_severity(finding.get("severity", "medium"))
            finding["confidence"] = self._clamp(float(finding.get("confidence", 1.0)), 0.0, 1.0)
            finding["flags"] = self._string_list(finding.get("flags"))
            finding["remediation_actions"] = self._string_list(finding.get("remediation_actions"))
            if "error_type" in finding and hasattr(finding["error_type"], "value"):
                finding["error_type"] = finding["error_type"].value
            elif finding.get("error_type") is None:
                finding["error_type"] = ""
            normalized.append(finding)
        return normalized

    def _record_level_flags(self, subsystem_results: Mapping[str, Mapping[str, Any]]) -> Dict[str, List[str]]:
        record_flags: Dict[str, List[str]] = {}
        for subsystem, result in subsystem_results.items():
            for finding in result.get("findings", []):
                flags = self._string_list(finding.get("flags"))
                affected = self._string_list(
                    finding.get("affected_records") or finding.get("affected_record_ids") or []
                )
                for record_id in affected:
                    bucket = record_flags.setdefault(record_id, [])
                    for flag in flags:
                        scoped_flag = f"{subsystem}:{flag}"
                        if scoped_flag not in bucket:
                            bucket.append(scoped_flag)
        return record_flags

    def _decision_confidence(self, subsystem_results: Mapping[str, Mapping[str, Any]]) -> float:
        confidences: List[float] = []
        for result in subsystem_results.values():
            findings = result.get("findings", [])
            if findings:
                confidences.extend(self._clamp(float(item.get("confidence", 1.0)), 0.0, 1.0) for item in findings)
            else:
                confidences.append(self._decision_confidence_for_score(float(result.get("batch_score", 0.0))))
        if not confidences:
            return 1.0
        return self._clamp(sum(confidences) / len(confidences), 0.0, 1.0)

    def _decision_confidence_for_score(self, score: float) -> float:
        if score >= self.pass_threshold:
            return 0.95
        if score >= self.warn_threshold:
            return 0.80
        return 0.65

    def _error_finding(self, *, subsystem: str, error: DataQualityError) -> Dict[str, Any]:
        payload = error.to_dict()
        return {
            "checker": subsystem,
            "domain": subsystem,
            "verdict": "block" if self.fail_closed_on_subsystem_error else "warn",
            "severity": payload.get("severity", "high"),
            "confidence": 1.0,
            "message": payload.get("message", f"{subsystem} subsystem failure"),
            "flags": [f"{subsystem}_subsystem_failure"],
            "remediation_actions": [payload.get("remediation")] if payload.get("remediation") else [],
            "error_type": payload.get("error_type", QualityErrorType.INTERNAL_QUALITY_AGENT_FAILURE.value),
            "error": payload,
        }

    def _error_from_payload(self, payload: Mapping[str, Any]) -> DataQualityError:
        return DataQualityError(
            message=str(payload.get("message") or "subsystem failure"),
            error_type=QualityErrorType(str(payload.get("error_type") or QualityErrorType.INTERNAL_QUALITY_AGENT_FAILURE.value)),
            severity=QualitySeverity(str(payload.get("severity") or QualitySeverity.HIGH.value)),
            retryable=bool(payload.get("retryable", False)),
            context=self._normalized_mapping(payload.get("context", {})),
            remediation=payload.get("remediation"),
            stage=QualityStage(str(payload.get("stage") or QualityStage.UNKNOWN.value)),
            domain=QualityDomain(str(payload.get("domain") or QualityDomain.SYSTEM.value)),
            disposition=QualityDisposition(str(payload.get("disposition") or QualityDisposition.BLOCK.value)),
            dataset_id=payload.get("dataset_id"),
            source_id=payload.get("source_id"),
            batch_id=payload.get("batch_id"),
            record_id=payload.get("record_id"),
            rule_id=payload.get("rule_id"),
        )

    def _schema_required_fields(self, schema: Optional[Mapping[str, Any]]) -> Optional[List[str]]:
        if not isinstance(schema, Mapping):
            return None
        required = schema.get("required_fields") or schema.get("required") or []
        if isinstance(required, Mapping):
            required = [key for key, value in required.items() if value]
        return self._string_list(required)

    def _schema_version(self, schema: Optional[Mapping[str, Any]], structural_result: Mapping[str, Any]) -> Optional[str]:
        if isinstance(schema, Mapping) and schema.get("schema_version"):
            return str(schema.get("schema_version"))
        raw = structural_result if isinstance(structural_result, Mapping) else {}
        return raw.get("schema_version")

    def _summary_error_type(self, subsystem: str, verdict: str) -> str:
        normalized = self._normalize_verdict(verdict)
        if subsystem == "structural":
            return QualityErrorType.SCHEMA_VALIDATION_FAILED.value if normalized == "block" else ""
        if subsystem == "statistical":
            return QualityErrorType.DISTRIBUTION_DRIFT_DETECTED.value if normalized == "block" else ""
        if subsystem == "semantic":
            return QualityErrorType.CROSS_FIELD_CONFLICT.value if normalized == "block" else ""
        return ""

    def _summary_severity_for_verdict(self, verdict: str) -> str:
        normalized = self._normalize_verdict(verdict)
        if normalized == "block":
            return "high"
        if normalized == "warn":
            return "medium"
        return "low"

    def _store_error_marker(self, error_payload: Mapping[str, Any]) -> None:
        if self.shared_memory is None or not self.publish_to_shared_memory:
            return
        try:
            key = f"{self.shared_error_key_prefix}:{uuid.uuid4().hex}"
            self._shared_set(key, deepcopy(dict(error_payload)), ttl=self.shared_ttl_seconds)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Shared-memory helpers
    # ------------------------------------------------------------------
    def _shared_set(self, key: str, value: Any, *, ttl: Optional[int] = None) -> None:
        if self.shared_memory is None:
            return
        setter = getattr(self.shared_memory, "set", None) or getattr(self.shared_memory, "put", None)
        if setter is None:
            raise TypeError("shared_memory does not expose a set/put method")
        try:
            if ttl is not None:
                setter(key, value, ttl=ttl)
            else:
                setter(key, value)
        except TypeError:
            setter(key, value)

    def _shared_publish(self, channel: str, payload: Any) -> None:
        if self.shared_memory is None:
            return
        publisher = getattr(self.shared_memory, "publish", None)
        if publisher is None:
            return
        publisher(channel, payload)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_runtime_configuration(self) -> None:
        if self.warn_threshold > self.pass_threshold:
            raise DataQualityError(
                message="quality_agent.warn_threshold must be less than or equal to pass_threshold",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.ESCALATE,
                remediation="Correct the quality_agent thresholds so warn_threshold <= pass_threshold.",
                context={"warn_threshold": self.warn_threshold, "pass_threshold": self.pass_threshold},
            )

        valid_subsystems = {"structural", "statistical", "semantic"}
        order = [item for item in self.subsystem_order if item]
        if not order or any(item not in valid_subsystems for item in order):
            raise DataQualityError(
                message="quality_agent.subsystem_order contains unsupported subsystem names",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.ESCALATE,
                remediation="Use a non-empty subsystem_order composed of structural, statistical, and semantic.",
                context={"subsystem_order": self.subsystem_order},
            )

        missing_weights = [item for item in valid_subsystems if item not in self.subsystem_weights]
        if missing_weights:
            raise DataQualityError(
                message="quality_agent.subsystem_weights is missing one or more subsystem weights",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                stage=QualityStage.SCORING,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.ESCALATE,
                remediation="Provide weights for structural, statistical, and semantic subsystems.",
                context={"missing_weights": missing_weights},
            )

    def _normalized_weights(self, weights: Mapping[str, Any]) -> Dict[str, float]:
        normalized: Dict[str, float] = {}
        total = 0.0
        for key, value in dict(weights or {}).items():
            numeric = float(value)
            if numeric < 0:
                raise DataQualityError(
                    message=f"Subsystem weight '{key}' must be non-negative",
                    error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
                    severity=QualitySeverity.HIGH,
                    retryable=False,
                    stage=QualityStage.SCORING,
                    domain=QualityDomain.SYSTEM,
                    disposition=QualityDisposition.ESCALATE,
                    remediation="Use non-negative subsystem weights.",
                    context={"weight": key, "value": value},
                )
            normalized[str(key)] = numeric
            total += numeric
        if total <= 0:
            raise DataQualityError(
                message="At least one positive subsystem weight is required",
                error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                stage=QualityStage.SCORING,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.ESCALATE,
                remediation="Provide positive subsystem weights that sum to a non-zero value.",
            )
        return {key: value / total for key, value in normalized.items()}

    def _normalize_records(self, records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for record in records or []:
            if not isinstance(record, Mapping):
                raise DataQualityError(
                    message="Each record supplied to QualityAgent must be a mapping",
                    error_type=QualityErrorType.SCHEMA_VALIDATION_FAILED,
                    severity=QualitySeverity.HIGH,
                    retryable=False,
                    stage=QualityStage.VALIDATION,
                    domain=QualityDomain.SYSTEM,
                    disposition=QualityDisposition.BLOCK,
                    remediation="Normalize records into dictionaries before invoking the quality gate.",
                    context={"record_type": type(record).__name__},
                )
            normalized.append({str(key): self._safe_value(value) for key, value in dict(record).items()})
        return normalized

    def _normalize_verdict(self, verdict: Any) -> str:
        value = str(verdict).strip().lower()
        if value not in {"pass", "warn", "block"}:
            raise DataQualityError(
                message=f"Unsupported verdict '{verdict}'",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.MEDIUM,
                retryable=False,
                stage=QualityStage.ROUTING,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.WARN,
                remediation="Use one of the supported verdicts: pass, warn, or block.",
                context={"verdict": verdict},
            )
        return value

    def _normalize_severity(self, severity: Any) -> str:
        value = str(severity).strip().lower()
        if value not in {"low", "medium", "high", "critical"}:
            return "medium"
        return value

    def _score_from_verdict(self, verdict: Any) -> float:
        normalized = self._normalize_verdict(verdict)
        return {"pass": 1.0, "warn": 0.75, "block": 0.0}[normalized]

    def _positive_int(self, value: Any, field_name: str) -> int:
        try:
            resolved = int(value)
        except Exception as exc:
            raise DataQualityError(
                message=f"{field_name} must be a positive integer",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.ESCALATE,
                remediation="Provide a positive integer configuration value.",
                context={"field_name": field_name, "value": value},
            ) from exc
        if resolved <= 0:
            raise DataQualityError(
                message=f"{field_name} must be a positive integer",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.HIGH,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.ESCALATE,
                remediation="Provide a positive integer configuration value.",
                context={"field_name": field_name, "value": value},
            )
        return resolved

    def _optional_nonnegative_int(self, value: Any, field_name: str) -> Optional[int]:
        if value is None:
            return None
        try:
            resolved = int(value)
        except Exception as exc:
            raise DataQualityError(
                message=f"{field_name} must be a non-negative integer or null",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.MEDIUM,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.WARN,
                remediation="Provide a non-negative integer or null value.",
                context={"field_name": field_name, "value": value},
            ) from exc
        if resolved < 0:
            raise DataQualityError(
                message=f"{field_name} must be a non-negative integer or null",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.MEDIUM,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.WARN,
                remediation="Provide a non-negative integer or null value.",
                context={"field_name": field_name, "value": value},
            )
        return resolved

    def _bounded_score(self, value: Any, *, field_name: str) -> float:
        try:
            numeric = float(value)
        except Exception as exc:
            raise DataQualityError(
                message=f"{field_name} must be numeric",
                error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
                severity=QualitySeverity.MEDIUM,
                retryable=False,
                stage=QualityStage.SCORING,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.WARN,
                remediation="Provide a numeric score within [0.0, 1.0].",
                context={"field_name": field_name, "value": value},
            ) from exc
        if numeric < 0.0 or numeric > 1.0:
            raise DataQualityError(
                message=f"{field_name} must be within [0.0, 1.0]",
                error_type=QualityErrorType.POLICY_THRESHOLD_INVALID,
                severity=QualitySeverity.MEDIUM,
                retryable=False,
                stage=QualityStage.SCORING,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.WARN,
                remediation="Adjust the score or threshold so it remains within [0.0, 1.0].",
                context={"field_name": field_name, "value": numeric},
            )
        return numeric

    def _nonempty(self, value: Any, field_name: str) -> str:
        text = "" if value is None else str(value).strip()
        if not text:
            raise DataQualityError(
                message=f"{field_name} must not be empty",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.MEDIUM,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.WARN,
                remediation="Provide a non-empty identifier value.",
                context={"field_name": field_name},
            )
        return text

    def _float_mapping(self, value: Any) -> Dict[str, float]:
        if not isinstance(value, Mapping):
            return {}
        return {str(key): float(item) for key, item in dict(value).items()}

    def _string_list(self, values: Any) -> List[str]:
        if values is None:
            return []
        return [str(item) for item in values]

    def _merge_unique_strings(self, *groups: Any) -> List[str]:
        merged: List[str] = []
        for group in groups:
            for item in self._string_list(group):
                if item not in merged:
                    merged.append(item)
        return merged

    def _normalized_mapping(self, value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise DataQualityError(
                message="Expected a mapping while normalizing task context",
                error_type=QualityErrorType.CONFIGURATION_INVALID,
                severity=QualitySeverity.MEDIUM,
                retryable=False,
                stage=QualityStage.VALIDATION,
                domain=QualityDomain.SYSTEM,
                disposition=QualityDisposition.WARN,
                remediation="Provide context and configuration payloads as dictionaries.",
                context={"value_type": type(value).__name__},
            )
        return {str(key): self._safe_value(item) for key, item in dict(value).items()}

    def _safe_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Mapping):
            return {str(key): self._safe_value(item) for key, item in dict(value).items()}
        if isinstance(value, (list, tuple, set)):
            return [self._safe_value(item) for item in value]
        return str(value)

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, float(value)))

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


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
        "schema_version": "v2.0.0",
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
