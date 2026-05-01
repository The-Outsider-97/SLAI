"""
Production-grade Secure STPA / STPA-Sec analysis for the Safety Agent subsystem.

This module implements a security-aware System-Theoretic Process Analysis layer
for safety, cyber-security, AI/user-safety, and operational-governance controls.
It keeps the original STPA context from Nancy Leveson and John Thomas while
extending the workflow with STPA-Sec style control-action analysis, audit-safe
artifact storage, deterministic hazard linking, fault-tree style context
reasoning, and system-of-systems consistency checks.

Design goals:
- keep the STPA workflow explicit, inspectable, and auditable;
- preserve the existing public methods used by the Safety Agent subsystem;
- use secure_config.yaml as the single configuration source;
- use shared safety helpers for identifiers, timestamps, redaction, hashing,
  scoring, serialization, and config access instead of duplicating helpers;
- use structured security errors for malformed analysis inputs, configuration
  tampering, unsafe state, audit failure, and report integrity issues;
- avoid leaking raw sensitive system details through logs, memory, or reports.
"""

from __future__ import annotations

import itertools
import re

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, DefaultDict, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.security_error import *
from .utils.safety_helpers import *
from .secure_memory import SecureMemory
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI System-Theoretic Process Analysis")
printer = PrettyPrinter()

MODULE_VERSION = "2.1.0"
SCOPE_SCHEMA_VERSION = "secure_stpa.scope.v2"
UCA_SCHEMA_VERSION = "secure_stpa.uca.v2"
CONTEXT_SCHEMA_VERSION = "secure_stpa.context.v2"
SCENARIO_SCHEMA_VERSION = "secure_stpa.loss_scenario.v2"
REPORT_SCHEMA_VERSION = "secure_stpa.report.v2"


@dataclass(frozen=True)
class STPAScope:
    """Structured scope definition for one STPA/STPA-Sec run."""

    scope_id: str
    losses: List[str]
    hazards: List[str]
    constraints: List[str]
    system_boundary: str
    assumptions: List[str] = field(default_factory=list)
    assets: List[str] = field(default_factory=list)
    stakeholders: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_iso)
    schema_version: str = SCOPE_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return redact_value(asdict(self))


@dataclass(frozen=True)
class UnsafeControlAction:
    """Audit-safe representation of an unsafe control action."""

    id: str
    controller: str
    control_action: str
    guideword: str
    hazard_link: str
    state_constraints: List[str]
    severity: float
    likelihood: float
    risk_score: float
    risk_level: str
    decision: str
    rationale: str
    indicators: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=utc_iso)
    schema_version: str = UCA_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return redact_value(asdict(self))


@dataclass(frozen=True)
class ContextTableEntry:
    """Context table row derived from an unsafe control action."""

    context_id: str
    uca_id: str
    controller: str
    control_action: str
    guideword: str
    process_variables: List[str]
    hazard_conditions: List[Dict[str, Any]]
    state_constraints: List[str]
    security_context: Dict[str, Any]
    risk_score: float
    risk_level: str
    decision: str
    timestamp: str = field(default_factory=utc_iso)
    schema_version: str = CONTEXT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return redact_value(asdict(self))


@dataclass(frozen=True)
class LossScenario:
    """Loss scenario generated from a context table entry."""

    scenario_id: str
    context_id: str
    loss: str
    severity: float
    probability: float
    risk_level: float
    normalized_risk: float
    risk_band: str
    decision: str
    mitigation: List[str]
    causal_factors: List[str]
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_iso)
    schema_version: str = SCENARIO_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return redact_value(asdict(self))


class SecureSTPA:
    """
    Security-aware STPA/STPA-Sec workflow for Safety Agent control analysis.

    Backward-compatible public methods retained:
    - reset_analysis
    - define_analysis_scope
    - model_control_structure
    - identify_unsafe_control_actions
    - build_context_tables
    - identify_loss_scenarios
    - perform_sos_analysis
    - export_analysis_report
    """

    def __init__(self):
        """
        Implementing functions from:
        STPA Handbook - Nancy G. Leveson & John P. Thomas
        Nuclear Engineering and Technology - Sejin Jung, Yoona Heo, Junbeom Yoo
        1.2 Identifying hazardous system behaviour - Stephan Baumgart & Sasikumar Punnekkat
        """
        self.config = load_global_config()
        self.stpa_config = get_config_section("secure_stpa")
        self.memory = SecureMemory()
        self._validate_configuration()
        self.reset_analysis()
        logger.info(
            "Secure STPA initialized: %s",
            stable_json(safe_log_payload("secure_stpa.initialized", {
                "schema_version": self._cfg("schema_version"),
                "enabled": self._cfg("enabled", True),
                "store_artifacts": self._cfg("memory.store_artifacts", True),
            })),
        )

    # ------------------------------------------------------------------
    # Configuration and internal utilities
    # ------------------------------------------------------------------

    def _cfg(self, path: Union[str, Sequence[str]], default: Any = None) -> Any:
        return get_nested(self.stpa_config or {}, path, default)

    def _validate_configuration(self) -> None:
        if not isinstance(self.stpa_config, Mapping):
            raise ConfigurationTamperingError(
                "secure_config.yaml:secure_stpa",
                "secure_stpa section must be a mapping",
                component="secure_stpa",
                severity=SecuritySeverity.HIGH,
            )
        if not coerce_bool(self._cfg("enabled", True), True):
            logger.warning("Secure STPA is disabled by configuration.")
            return
        if not coerce_bool(self._cfg("strict_config_validation", True), True):
            return

        required = ["guidewords", "thresholds", "memory", "probability", "sos", "report"]
        missing = [key for key in required if key not in self.stpa_config]
        if missing:
            raise ConfigurationTamperingError(
                "secure_config.yaml:secure_stpa",
                f"Missing required secure_stpa config keys: {missing}",
                component="secure_stpa",
                severity=SecuritySeverity.HIGH,
            )
        if not isinstance(self._cfg("guidewords"), Sequence):
            raise ConfigurationTamperingError(
                "secure_config.yaml:secure_stpa.guidewords",
                "guidewords must be a sequence",
                component="secure_stpa",
                severity=SecuritySeverity.HIGH,
            )

    def _max_items(self, key: str, default: int) -> int:
        return coerce_int(self._cfg(f"limits.{key}", default), default, minimum=1)

    def _normalize_items(self, values: Optional[Iterable[Any]], *, field_name: str, max_items_key: str, max_length: int = 512) -> List[str]:
        if values is None:
            return []
        if isinstance(values, (str, bytes)):
            raw_values = [values]
        else:
            raw_values = list(values)
        max_items = self._max_items(max_items_key, len(raw_values) or 1)
        cleaned = [normalize_text(value, max_length=max_length, preserve_newlines=False) for value in raw_values[:max_items]]
        cleaned = [item for item in cleaned if item]
        if not cleaned:
            raise SecurityError(
                SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                f"Secure STPA field '{field_name}' must contain at least one valid item.",
                severity=SecuritySeverity.HIGH,
                component="secure_stpa",
                context={"field_name": field_name, "max_items_key": max_items_key},
                response_action=SecurityResponseAction.BLOCK,
            )
        return dedupe_preserve_order(cleaned)

    def _memory_tags(self, *tags: str) -> List[str]:
        base_tags = self._cfg("memory.base_tags", ["secure_stpa", "safety_analysis"])
        return dedupe_preserve_order([normalize_identifier(tag, max_length=96) for tag in list(base_tags or []) + list(tags) if str(tag).strip()])

    def _internal_context(self, action: str) -> Dict[str, Any]:
        return {
            "auth_token": "internal_secure_stpa_context",
            "access_level": coerce_int(self._cfg("memory.internal_access_level", 10), 10),
            "purpose": normalize_identifier(action, max_length=96),
            "principal": "secure_stpa",
            "component": "secure_stpa",
            "request_id": generate_request_id(),
        }

    def _store_artifact(
        self,
        artifact: Mapping[str, Any],
        *,
        tags: Sequence[str],
        sensitivity: Optional[float] = None,
        purpose: str = "secure_stpa_analysis",
        classification: Optional[str] = None,
    ) -> Optional[str]:
        if not coerce_bool(self._cfg("memory.store_artifacts", True), True):
            return None
        try:
            safe_artifact = redact_value(dict(artifact))
            return self.memory.add(
                safe_artifact,
                tags=self._memory_tags(*tags),
                sensitivity=coerce_float(sensitivity if sensitivity is not None else self._cfg("memory.default_sensitivity", 0.7), 0.7, minimum=0.0, maximum=1.0),
                ttl_seconds=coerce_int(self._cfg("memory.ttl_seconds", 604800), 604800, minimum=0),
                purpose=purpose,
                owner="secure_stpa",
                classification=classification or str(self._cfg("memory.default_classification", "confidential")),
                source="secure_stpa",
                metadata={"module_version": MODULE_VERSION, "artifact_fingerprint": fingerprint(safe_artifact)},
            )
        except TypeError:
            # Compatibility with older SecureMemory.add signatures in downstream branches.
            try:
                return self.memory.add(
                    redact_value(dict(artifact)),
                    tags=self._memory_tags(*tags),
                    sensitivity=coerce_float(sensitivity if sensitivity is not None else self._cfg("memory.default_sensitivity", 0.7), 0.7, minimum=0.0, maximum=1.0),
                )
            except Exception as exc:
                return self._handle_store_error(exc, artifact, tags)
        except Exception as exc:
            return self._handle_store_error(exc, artifact, tags)

    def _handle_store_error(self, exc: BaseException, artifact: Mapping[str, Any], tags: Sequence[str]) -> Optional[str]:
        if coerce_bool(self._cfg("memory.fail_closed_on_store_error", False), False):
            raise AuditLogFailureError(
                "secure_stpa.memory",
                f"Failed to store STPA artifact: {type(exc).__name__}",
                component="secure_stpa",
                cause=exc,
                context={"artifact_fingerprint": fingerprint(artifact), "tags": list(tags)},
            )
        logger.warning("Secure STPA artifact storage failed: %s", stable_json(safe_log_payload(
            "secure_stpa.store_failed",
            {"error": type(exc).__name__, "artifact_fingerprint": fingerprint(artifact), "tags": list(tags)},
        )))
        return None

    def _risk_decision(self, risk: float) -> str:
        return threshold_decision(
            risk,
            block_threshold=coerce_float(self._cfg("thresholds.block", 0.72), 0.72, minimum=0.0, maximum=1.0),
            review_threshold=coerce_float(self._cfg("thresholds.review", 0.40), 0.40, minimum=0.0, maximum=1.0),
        )

    # ------------------------------------------------------------------
    # Lifecycle and public workflow
    # ------------------------------------------------------------------

    def reset_analysis(self) -> None:
        """Reset all analysis components."""
        self.scope: Optional[STPAScope] = None
        self.scope_id = generate_identifier("stpa_scope")
        self.losses: List[str] = []
        self.hazards: List[str] = []
        self.safety_constraints: List[str] = []
        self.system_boundary = "System Boundary Not Defined"
        self.control_structure: Dict[str, Dict[str, Any]] = {}
        self.context_tables: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.uca_table: List[Dict[str, Any]] = []
        self.loss_scenarios: List[Dict[str, Any]] = []
        self.process_models: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)
        self.component_states: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)
        self.transition_history: DefaultDict[str, Deque[Dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=coerce_int(self._cfg("sos.max_transition_history", 100), 100, minimum=1))
        )
        self.analysis_stats: Dict[str, Any] = {
            "scope_defined": False,
            "components": 0,
            "unsafe_control_actions": 0,
            "contexts": 0,
            "loss_scenarios": 0,
            "reports_generated": 0,
            "last_updated": utc_iso(),
        }

    def define_analysis_scope(
        self,
        losses: List[str],
        hazards: List[str],
        constraints: List[str],
        system_boundary: Optional[str] = None,
        *,
        assumptions: Optional[List[str]] = None,
        assets: Optional[List[str]] = None,
        stakeholders: Optional[List[str]] = None,
    ) -> None:
        """Define the scope of the STPA/STPA-Sec analysis."""
        self.losses = self._normalize_items(losses, field_name="losses", max_items_key="max_losses")
        self.hazards = self._normalize_items(hazards, field_name="hazards", max_items_key="max_hazards")
        self.safety_constraints = self._normalize_items(constraints, field_name="constraints", max_items_key="max_constraints")
        self.system_boundary = normalize_text(system_boundary or self._cfg("default_system_boundary", "System Boundary Not Defined"), max_length=1024)
        self.scope = STPAScope(
            scope_id=self.scope_id,
            losses=self.losses,
            hazards=self.hazards,
            constraints=self.safety_constraints,
            system_boundary=self.system_boundary,
            assumptions=dedupe_preserve_order(normalize_text(item, max_length=512) for item in (assumptions or [])),
            assets=dedupe_preserve_order(normalize_text(item, max_length=256) for item in (assets or [])),
            stakeholders=dedupe_preserve_order(normalize_text(item, max_length=256) for item in (stakeholders or [])),
        )
        self.analysis_stats["scope_defined"] = True
        self.analysis_stats["last_updated"] = utc_iso()
        self._store_artifact(self.scope.to_dict(), tags=["stpa_scope"], sensitivity=0.75)
        printer.status("STPA", "Scope defined with hazards, losses, constraints.", "info")

    def model_control_structure(
        self,
        structure: Dict[str, Dict[str, List[str]]],
        process_models: Optional[Dict[str, Dict[str, List[str]]]] = None,
    ) -> None:
        """
        Model the control structure.

        Expected shape:
        {
            "Controller": {
                "inputs": ["sensor1", "feedback"],
                "outputs": ["control_action1"],
                "process_vars": ["state_var1", "threshold"],
                "safe_states": ["SAFE"],
                "trust_boundary": "internal"
            }
        }
        """
        if not isinstance(structure, Mapping) or not structure:
            raise SecurityError(
                SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                "Control structure must be a non-empty mapping.",
                severity=SecuritySeverity.HIGH,
                component="secure_stpa",
                response_action=SecurityResponseAction.BLOCK,
            )

        max_components = self._max_items("max_components", 100)
        normalized_structure: Dict[str, Dict[str, Any]] = {}
        for component_index, (raw_component, raw_data) in enumerate(structure.items()):
            if component_index >= max_components:
                break
            component = normalize_identifier(raw_component, max_length=128)
            if not isinstance(raw_data, Mapping):
                raise SecurityError(
                    SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                    "Control component configuration must be a mapping.",
                    severity=SecuritySeverity.HIGH,
                    component="secure_stpa",
                    context={"component": component},
                    response_action=SecurityResponseAction.BLOCK,
                )
            outputs = self._normalize_optional_list(raw_data.get("outputs", []), max_items_key="max_control_actions", max_length=256)
            if not outputs:
                raise SecurityError(
                    SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                    "Control component must define at least one output/control action.",
                    severity=SecuritySeverity.HIGH,
                    component="secure_stpa",
                    context={"component": component},
                    response_action=SecurityResponseAction.BLOCK,
                )
            normalized_structure[component] = {
                "inputs": self._normalize_optional_list(raw_data.get("inputs", []), max_items_key="max_interfaces", max_length=256),
                "outputs": outputs,
                "process_vars": self._normalize_optional_list(raw_data.get("process_vars", raw_data.get("process_variables", [])), max_items_key="max_process_variables", max_length=256),
                "feedback": self._normalize_optional_list(raw_data.get("feedback", []), max_items_key="max_interfaces", max_length=256),
                "trust_boundary": normalize_text(raw_data.get("trust_boundary", "internal"), max_length=128, lowercase=True),
                "authority_level": normalize_text(raw_data.get("authority_level", "nominal"), max_length=128, lowercase=True),
                "safe_states": self._normalize_optional_list(raw_data.get("safe_states", self._cfg("sos.default_safe_states", ["SAFE", "NORMAL"])), max_items_key="max_states", max_length=128),
                "unsafe_states": self._normalize_optional_list(raw_data.get("unsafe_states", []), max_items_key="max_states", max_length=128),
                "metadata": redact_value(dict(raw_data.get("metadata", {}))) if isinstance(raw_data.get("metadata", {}), Mapping) else {},
            }

        normalized_process_models: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)
        for component, data in dict(process_models or {}).items():
            normalized_component = normalize_identifier(component, max_length=128)
            normalized_process_models[normalized_component] = self._normalize_process_model(data)
        for component, data in normalized_structure.items():
            if component not in normalized_process_models:
                normalized_process_models[component] = {
                    "variables": list(data.get("process_vars", [])),
                    "constraints": [f"State-dependent constraint for {component}"],
                    "states": list(data.get("safe_states", [])) + list(data.get("unsafe_states", [])) + ["INIT"],
                    "safe_states": list(data.get("safe_states", [])),
                    "transitions": [],
                }

        self.control_structure = normalized_structure
        self.process_models = normalized_process_models
        self.component_states.clear()
        for component in self.control_structure:
            initial_state = normalize_text(self.process_models.get(component, {}).get("initial_state", self._cfg("sos.initial_state", "INIT")), max_length=128)
            self.component_states[component] = {
                "current": initial_state,
                "transitions": defaultdict(list),
                "last_updated": utc_iso(),
            }

        self.analysis_stats["components"] = len(self.control_structure)
        self.analysis_stats["last_updated"] = utc_iso()
        self._store_artifact(
            {"control_structure": self.control_structure, "process_models": self.process_models},
            tags=["stpa_model"],
            sensitivity=0.75,
        )
        printer.status("STPA", f"Control structure modeled with {len(self.control_structure)} components.", "info")

    def identify_unsafe_control_actions(self, custom_guidewords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Identify unsafe control actions using STPA and STPA-Sec guidewords."""
        self._require_scope_and_model()
        guidewords = self._guidewords(custom_guidewords)
        self.uca_table = []
        ucas: List[Dict[str, Any]] = []

        for controller, comp_data in self.control_structure.items():
            for action in comp_data.get("outputs", []):
                for guideword in guidewords:
                    hazard_link, rationale, indicators = self._determine_hazard_link(controller, action, guideword, include_details=True)
                    severity = self._estimate_uca_severity(hazard_link, guideword, controller, action)
                    likelihood = self._estimate_uca_likelihood(controller, action, guideword)
                    risk = clamp_score(severity * likelihood)
                    uca = UnsafeControlAction(
                        id=f"uca_{len(self.uca_table) + 1:04d}",
                        controller=controller,
                        control_action=action,
                        guideword=guideword,
                        hazard_link=hazard_link,
                        state_constraints=self._get_state_constraints(controller),
                        severity=severity,
                        likelihood=likelihood,
                        risk_score=risk,
                        risk_level=categorize_risk(risk),
                        decision=self._risk_decision(risk),
                        rationale=rationale,
                        indicators=indicators,
                    )
                    uca_dict = uca.to_dict()
                    self.uca_table.append(uca_dict)
                    ucas.append(uca_dict)
                    self._store_artifact(uca_dict, tags=["unsafe_control_action", f"controller:{controller}"], sensitivity=0.8)

        self.analysis_stats["unsafe_control_actions"] = len(ucas)
        self.analysis_stats["last_updated"] = utc_iso()
        printer.status("STPA", f"{len(ucas)} UCAs identified.", "warn")
        return ucas

    def build_context_tables(
        self,
        formal_spec: Optional[Dict[str, Any]] = None,
        fta_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Build context tables with formal specification and FTA-style integration."""
        if not self.uca_table:
            self.identify_unsafe_control_actions()
        self.context_tables = defaultdict(list)

        for uca in self.uca_table:
            context_entry = self._generate_context_entry(uca, formal_spec or {}, fta_config or {})
            self.context_tables[context_entry["controller"]].append(context_entry)
            self._store_artifact(context_entry, tags=["stpa_context", f"controller:{context_entry['controller']}"], sensitivity=0.8)

        self.analysis_stats["contexts"] = sum(len(items) for items in self.context_tables.values())
        self.analysis_stats["last_updated"] = utc_iso()
        printer.status("STPA", "Context tables constructed.", "info")
        return dict(self.context_tables)

    def identify_loss_scenarios(self, probability_model: str = "heuristic") -> List[Dict[str, Any]]:
        """Generate loss scenarios from context tables."""
        if not self.context_tables:
            self.build_context_tables()
        if probability_model not in set(self._cfg("probability.supported_models", ["heuristic", "conservative"])):
            raise ConfigurationTamperingError(
                "secure_config.yaml:secure_stpa.probability.supported_models",
                f"Unsupported probability model requested: {probability_model}",
                component="secure_stpa",
                severity=SecuritySeverity.HIGH,
            )

        scenarios: List[Dict[str, Any]] = []
        for _, contexts in self.context_tables.items():
            for context in contexts:
                scenario = self._generate_loss_scenario(context, probability_model)
                scenarios.append(scenario)
                self._store_artifact(scenario, tags=["loss_scenario", f"risk:{scenario.get('risk_band', 'unknown')}"], sensitivity=0.85)

        scenarios.sort(key=lambda item: coerce_float(item.get("normalized_risk", 0.0), 0.0), reverse=True)
        self.loss_scenarios = scenarios
        self.analysis_stats["loss_scenarios"] = len(scenarios)
        self.analysis_stats["last_updated"] = utc_iso()
        printer.status("STPA", f"{len(scenarios)} loss scenarios identified.", "error")
        return scenarios

    def perform_sos_analysis(
        self,
        consistency_checks: bool = True,
        deadlock_detection: bool = True,
        safe_state_reachability: bool = True,
    ) -> Dict[str, Any]:
        """Perform System-of-Systems consistency, deadlock, and reachability analysis."""
        self._require_control_structure()
        results: Dict[str, Any] = {
            "schema_version": "secure_stpa.sos.v2",
            "timestamp": utc_iso(),
            "analysis_id": generate_identifier("sos"),
        }
        if consistency_checks:
            results["state_inconsistencies"] = self._check_state_consistency()
        if deadlock_detection:
            results["deadlock_risks"] = self._detect_communication_deadlocks()
        if safe_state_reachability:
            results["safe_state_analysis"] = self._analyze_safe_state_reachability()
        results["risk_score"] = self._score_sos_results(results)
        results["risk_level"] = categorize_risk(results["risk_score"])
        results["decision"] = self._risk_decision(results["risk_score"])
        self._store_artifact(results, tags=["sos_analysis"], sensitivity=0.75)
        return redact_value(results)

    def export_analysis_report(self, format: str = "json", include_sos: bool = False) -> Dict[str, Any]:
        """Export a structured STPA analysis report."""
        normalized_format = normalize_identifier(format, max_length=16).lower()
        if normalized_format not in {"json", "yaml", "dict"}:
            raise ConfigurationTamperingError(
                "secure_stpa.report.allowed_formats",
                f"Unsupported report format: {format}",
                component="secure_stpa",
                severity=SecuritySeverity.MEDIUM,
            )
        if not self.loss_scenarios:
            self.identify_loss_scenarios()

        report: Dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "module_version": MODULE_VERSION,
            "metadata": {
                "report_id": generate_identifier("stpa_report"),
                "created": utc_iso(),
                "system_boundary": self.system_boundary,
                "scope_id": self.scope_id,
                "config_path": self.config.get("__config_path__"),
                "format": normalized_format,
            },
            "summary": self.get_analysis_summary(),
            "losses": list(self.losses),
            "hazards": list(self.hazards),
            "safety_constraints": list(self.safety_constraints),
            "control_structure": redact_value(self.control_structure),
            "unsafe_control_actions": redact_value(self.uca_table),
            "context_tables": redact_value(dict(self.context_tables)),
            "loss_scenarios": redact_value(self.loss_scenarios),
            "mitigation_backlog": self.generate_mitigation_backlog(),
        }
        if include_sos:
            report["sos_analysis"] = self.perform_sos_analysis()
        if coerce_bool(self._cfg("report.include_integrity_hash", True), True):
            report["integrity"] = {
                "report_fingerprint": fingerprint(report),
                "artifact_count": len(self.uca_table) + sum(len(v) for v in self.context_tables.values()) + len(self.loss_scenarios),
            }

        self.analysis_stats["reports_generated"] += 1
        self.analysis_stats["last_updated"] = utc_iso()
        self._store_artifact(report, tags=["stpa_report"], sensitivity=0.85)
        return redact_value(report)

    # ------------------------------------------------------------------
    # Additional production methods
    # ------------------------------------------------------------------

    def record_component_transition(
        self,
        component: str,
        to_state: str,
        *,
        trigger: str = "unspecified",
        from_state: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a component state transition for SoS consistency analysis."""
        normalized_component = normalize_identifier(component, max_length=128)
        if normalized_component not in self.component_states:
            raise SystemIntegrityError(
                "secure_stpa.component_state",
                "Transition references unknown component.",
                expected_state="known component in control structure",
                actual_state=normalized_component,
                component="secure_stpa",
            )
        current = normalize_text(self.component_states[normalized_component].get("current", "INIT"), max_length=128)
        source_state = normalize_text(from_state or current, max_length=128)
        target_state = normalize_text(to_state, max_length=128)
        transition = {
            "transition_id": generate_identifier("stpa_transition"),
            "component": normalized_component,
            "from_state": source_state,
            "to_state": target_state,
            "trigger": normalize_text(trigger, max_length=256),
            "timestamp": utc_iso(),
            "metadata": redact_value(dict(metadata or {})),
        }
        self.component_states[normalized_component]["current"] = target_state
        self.component_states[normalized_component]["last_updated"] = transition["timestamp"]
        self.component_states[normalized_component]["transitions"][source_state].append(target_state)
        self.transition_history[normalized_component].append(transition)
        self._store_artifact(transition, tags=["component_transition", f"component:{normalized_component}"], sensitivity=0.65)
        return transition

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Return a compact audit-safe summary of current analysis state."""
        scenario_risks = [coerce_float(item.get("normalized_risk", 0.0), 0.0) for item in self.loss_scenarios]
        uca_risks = [coerce_float(item.get("risk_score", 0.0), 0.0) for item in self.uca_table]
        aggregate_risk = max(scenario_risks or uca_risks or [0.0])
        return {
            "scope_defined": bool(self.scope),
            "component_count": len(self.control_structure),
            "loss_count": len(self.losses),
            "hazard_count": len(self.hazards),
            "constraint_count": len(self.safety_constraints),
            "unsafe_control_action_count": len(self.uca_table),
            "context_count": sum(len(values) for values in self.context_tables.values()),
            "loss_scenario_count": len(self.loss_scenarios),
            "max_risk": clamp_score(aggregate_risk),
            "risk_level": categorize_risk(aggregate_risk),
            "decision": self._risk_decision(aggregate_risk),
            "stats": dict(self.analysis_stats),
            "timestamp": utc_iso(),
        }

    def generate_mitigation_backlog(self) -> List[Dict[str, Any]]:
        """Create prioritized mitigation items from loss scenarios and UCAs."""
        source_items = self.loss_scenarios or []
        backlog: List[Dict[str, Any]] = []
        for scenario in source_items:
            risk = coerce_float(scenario.get("normalized_risk", 0.0), 0.0)
            for mitigation in scenario.get("mitigation", []) or []:
                backlog.append({
                    "id": generate_identifier("mit"),
                    "source_scenario": scenario.get("scenario_id"),
                    "context_id": scenario.get("context_id"),
                    "mitigation": normalize_text(mitigation, max_length=512),
                    "priority": categorize_risk(risk),
                    "risk_score": clamp_score(risk),
                    "decision": self._risk_decision(risk),
                })
        backlog.sort(key=lambda item: item["risk_score"], reverse=True)
        return redact_value(backlog[: self._max_items("max_mitigations", 250)])

    def validate_analysis_integrity(self) -> Dict[str, Any]:
        """Validate internal consistency of the current analysis artifacts."""
        issues: List[Dict[str, Any]] = []
        if not self.scope:
            issues.append({"type": "missing_scope", "severity": "high"})
        if not self.control_structure:
            issues.append({"type": "missing_control_structure", "severity": "high"})
        known_uca_ids = {uca.get("id") for uca in self.uca_table}
        for controller, contexts in self.context_tables.items():
            for context in contexts:
                if context.get("uca_id") not in known_uca_ids:
                    issues.append({"type": "orphan_context", "controller": controller, "uca_id": context.get("uca_id"), "severity": "medium"})
        known_context_ids = {context.get("context_id") for contexts in self.context_tables.values() for context in contexts}
        for scenario in self.loss_scenarios:
            if scenario.get("context_id") not in known_context_ids:
                issues.append({"type": "orphan_loss_scenario", "scenario_id": scenario.get("scenario_id"), "severity": "medium"})
        result = {
            "valid": not issues,
            "issue_count": len(issues),
            "issues": redact_value(issues),
            "fingerprint": fingerprint({"summary": self.get_analysis_summary(), "issues": issues}),
            "timestamp": utc_iso(),
        }
        if issues and coerce_bool(self._cfg("integrity.fail_closed", False), False):
            raise SystemIntegrityError(
                "secure_stpa.analysis_integrity",
                "Secure STPA analysis contains consistency issues.",
                expected_state="no orphaned or missing analysis artifacts",
                actual_state=stable_json(result),
                component="secure_stpa",
            )
        return result

    # ------------------------------------------------------------------
    # Core analysis methods
    # ------------------------------------------------------------------

    def _normalize_optional_list(self, values: Any, *, max_items_key: str, max_length: int) -> List[str]:
        if values is None:
            return []
        if isinstance(values, (str, bytes)):
            raw_values = [values]
        elif isinstance(values, Iterable):
            raw_values = list(values)
        else:
            raw_values = [values]
        max_items = self._max_items(max_items_key, len(raw_values) or 1)
        return dedupe_preserve_order(
            item for item in (normalize_text(value, max_length=max_length) for value in raw_values[:max_items]) if item
        )

    def _normalize_process_model(self, data: Any) -> Dict[str, Any]:
        if not isinstance(data, Mapping):
            return {}
        return {
            "variables": self._normalize_optional_list(data.get("variables", data.get("process_variables", [])), max_items_key="max_process_variables", max_length=256),
            "constraints": self._normalize_optional_list(data.get("constraints", []), max_items_key="max_constraints", max_length=512),
            "states": self._normalize_optional_list(data.get("states", []), max_items_key="max_states", max_length=128),
            "safe_states": self._normalize_optional_list(data.get("safe_states", self._cfg("sos.default_safe_states", ["SAFE", "NORMAL"])), max_items_key="max_states", max_length=128),
            "unsafe_states": self._normalize_optional_list(data.get("unsafe_states", []), max_items_key="max_states", max_length=128),
            "initial_state": normalize_text(data.get("initial_state", self._cfg("sos.initial_state", "INIT")), max_length=128),
            "transitions": self._normalize_transitions(data.get("transitions", [])),
            "metadata": redact_value(dict(data.get("metadata", {}))) if isinstance(data.get("metadata", {}), Mapping) else {},
        }

    def _normalize_transitions(self, transitions: Any) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        if isinstance(transitions, Mapping):
            for source, targets in transitions.items():
                for target in self._normalize_optional_list(targets, max_items_key="max_states", max_length=128):
                    normalized.append({"from": normalize_text(source, max_length=128), "to": target})
        elif isinstance(transitions, Iterable) and not isinstance(transitions, (str, bytes)):
            for item in transitions:
                if isinstance(item, Mapping):
                    source = normalize_text(item.get("from", item.get("source", "")), max_length=128)
                    target = normalize_text(item.get("to", item.get("target", "")), max_length=128)
                    if source and target:
                        normalized.append({"from": source, "to": target})
        return normalized[: self._max_items("max_transitions", 500)]

    def _guidewords(self, custom_guidewords: Optional[List[str]]) -> List[str]:
        configured = list(self._cfg("guidewords", [])) + list(self._cfg("security_guidewords", []))
        guidewords = custom_guidewords or configured
        return self._normalize_items(guidewords, field_name="guidewords", max_items_key="max_guidewords", max_length=256)

    def _require_scope_and_model(self) -> None:
        if not self.scope:
            raise SystemIntegrityError(
                "secure_stpa.scope",
                "STPA scope must be defined before identifying unsafe control actions.",
                expected_state="scope defined",
                actual_state="scope missing",
                component="secure_stpa",
            )
        self._require_control_structure()

    def _require_control_structure(self) -> None:
        if not self.control_structure:
            raise SystemIntegrityError(
                "secure_stpa.control_structure",
                "Control structure must be modeled before this STPA operation.",
                expected_state="control structure modeled",
                actual_state="control structure missing",
                component="secure_stpa",
            )

    def _determine_hazard_link(self, controller: str, action: str, guideword: str, *, include_details: bool = False) -> Union[str, Tuple[str, str, List[str]]]:
        """Enhanced hazard linking with deterministic lexical and state-aware analysis."""
        if not self.hazards:
            raise SystemIntegrityError(
                "secure_stpa.hazards",
                "No hazards are available for hazard linking.",
                expected_state="non-empty hazards list",
                actual_state="empty hazards list",
                component="secure_stpa",
            )

        query = f"{controller} {action} {guideword}"
        query_tokens = self._token_set(query)
        current_state = normalize_text(self.component_states[controller].get("current", "INIT"), lowercase=True)
        ranked: List[Tuple[str, float, List[str]]] = []

        for hazard in self.hazards:
            hazard_tokens = self._token_set(hazard)
            overlap = query_tokens & hazard_tokens
            union = query_tokens | hazard_tokens
            lexical_score = len(overlap) / max(len(union), 1)
            indicators = sorted(overlap)
            lowered_hazard = hazard.lower()
            if controller.lower() in lowered_hazard:
                lexical_score += 0.15
                indicators.append("controller_match")
            if any(token in lowered_hazard for token in self._token_set(action)):
                lexical_score += 0.15
                indicators.append("action_match")
            if "emergency" in current_state and any(token in lowered_hazard for token in {"shutdown", "fail", "loss", "injury"}):
                lexical_score += 0.20
                indicators.append("emergency_state_bias")
            if any(token in lowered_hazard for token in self._cfg("hazard_priority_terms", ["injury", "loss", "breach", "shutdown", "exposure", "damage"])):
                lexical_score += 0.05
                indicators.append("priority_hazard_term")
            ranked.append((hazard, lexical_score, dedupe_preserve_order(indicators)))

        ranked.sort(key=lambda item: item[1], reverse=True)
        top_hazard, score, indicators = ranked[0]
        rationale = f"Hazard selected by lexical/state matching score {score:.3f}."
        if include_details:
            return top_hazard, rationale, indicators[:10]
        return top_hazard

    def _token_set(self, value: Any) -> Set[str]:
        text = normalize_text(value, max_length=2048, lowercase=True)
        stop_words = set(self._cfg("analysis.stop_words", ["the", "and", "or", "of", "to", "a", "an", "in", "for", "with", "by", "on"]))
        return {token for token in re.findall(r"[a-z0-9_'-]+", text) if token not in stop_words and len(token) > 1}

    def _estimate_uca_severity(self, hazard_link: str, guideword: str, controller: str, action: str) -> float:
        text = f"{hazard_link} {guideword} {controller} {action}".lower()
        score = coerce_float(self._cfg("severity.default", 0.55), 0.55, minimum=0.0, maximum=1.0)
        for band, terms in dict(self._cfg("severity.keyword_bands", {})).items():
            if any(str(term).lower() in text for term in terms or []):
                score = max(score, coerce_float(self._cfg(f"severity.band_scores.{band}", score), score, minimum=0.0, maximum=1.0))
        if any(term in guideword.lower() for term in ["unauthorized", "tampered", "spoofed", "disclosure", "exfiltration"]):
            score = max(score, coerce_float(self._cfg("severity.security_guideword_floor", 0.72), 0.72, minimum=0.0, maximum=1.0))
        return clamp_score(score)

    def _estimate_uca_likelihood(self, controller: str, action: str, guideword: str) -> float:
        component = self.control_structure.get(controller, {})
        base = coerce_float(self._cfg("probability.base_uca_likelihood", 0.35), 0.35, minimum=0.0, maximum=1.0)
        complexity = min(
            1.0,
            (len(component.get("inputs", [])) + len(component.get("outputs", [])) + len(component.get("process_vars", []))) /
            max(coerce_float(self._cfg("probability.complexity_normalizer", 12.0), 12.0, minimum=1.0), 1.0),
        )
        timing_bonus = coerce_float(self._cfg("probability.timing_guideword_bonus", 0.12), 0.12, minimum=0.0, maximum=1.0) if any(term in guideword.lower() for term in ["early", "late", "order", "timing"]) else 0.0
        trust_bonus = coerce_float(self._cfg("probability.external_trust_boundary_bonus", 0.10), 0.10, minimum=0.0, maximum=1.0) if component.get("trust_boundary") in {"external", "third_party", "untrusted"} else 0.0
        return clamp_score(base + 0.35 * complexity + timing_bonus + trust_bonus)

    def _generate_context_entry(self, uca: Dict[str, Any], formal_spec: Dict[str, Any], fta_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate context table entry with formal-method style supporting data."""
        controller = str(uca["controller"])
        action = str(uca["control_action"])
        process_vars = list(self.process_models.get(controller, {}).get("variables", []))
        process_vars.extend(self.control_structure.get(controller, {}).get("process_vars", []))

        if formal_spec:
            controller_spec = formal_spec.get(controller, {}) if isinstance(formal_spec, Mapping) else {}
            if isinstance(controller_spec, Mapping):
                process_vars.extend(self._normalize_optional_list(controller_spec.get(action, []), max_items_key="max_process_variables", max_length=256))

        conditions: List[Dict[str, Any]] = []
        if fta_config:
            conditions = self._run_fta_analysis(action, fta_config)

        base_risk = coerce_float(uca.get("risk_score", 0.0), 0.0)
        fta_risk = max([coerce_float(item.get("probability", 0.0), 0.0) for item in conditions] or [0.0])
        context_risk = combine_risk_scores(base_risk, fta_risk, method=str(self._cfg("risk_aggregation", "noisy_or")))
        security_context = self._derive_security_context(controller, action, uca, conditions)
        entry = ContextTableEntry(
            context_id=generate_identifier("ctx"),
            uca_id=str(uca["id"]),
            controller=controller,
            control_action=action,
            guideword=str(uca["guideword"]),
            process_variables=dedupe_preserve_order(process_vars)[: self._max_items("max_process_variables", 100)],
            hazard_conditions=conditions,
            state_constraints=list(uca.get("state_constraints", [])),
            security_context=security_context,
            risk_score=context_risk,
            risk_level=categorize_risk(context_risk),
            decision=self._risk_decision(context_risk),
        )
        return entry.to_dict()

    def _derive_security_context(self, controller: str, action: str, uca: Mapping[str, Any], conditions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        component = self.control_structure.get(controller, {})
        guideword = str(uca.get("guideword", "")).lower()
        tags: List[str] = []
        if any(term in guideword for term in ["unauthorized", "spoof", "tamper", "exfiltrat", "disclosure", "privilege"]):
            tags.append("security_relevant")
        if component.get("trust_boundary") in {"external", "third_party", "untrusted"}:
            tags.append("trust_boundary_crossing")
        if conditions:
            tags.append("fault_tree_conditions")
        return {
            "controller_trust_boundary": component.get("trust_boundary", "internal"),
            "authority_level": component.get("authority_level", "nominal"),
            "security_tags": dedupe_preserve_order(tags),
            "action_fingerprint": fingerprint(action),
            "condition_count": len(conditions),
        }

    def _generate_loss_scenario(self, context: Dict[str, Any], probability_model: str) -> Dict[str, Any]:
        """Generate loss scenario with severity/probability estimation and mitigations."""
        severity = self._scenario_severity(context)
        probability = self._heuristic_probability_estimation(context) if probability_model == "heuristic" else self._conservative_probability_estimation(context)
        normalized_risk = clamp_score(severity * probability)
        scenario = LossScenario(
            scenario_id=generate_identifier("scenario"),
            context_id=str(context["context_id"]),
            loss=self._map_to_loss(context),
            severity=severity,
            probability=probability,
            risk_level=round(severity * probability, 6),
            normalized_risk=normalized_risk,
            risk_band=categorize_risk(normalized_risk),
            decision=self._risk_decision(normalized_risk),
            mitigation=self._generate_mitigation_strategy(context),
            causal_factors=self._identify_causal_factors(context),
            evidence={
                "uca_id": context.get("uca_id"),
                "controller": context.get("controller"),
                "control_action": context.get("control_action"),
                "hazard_conditions": len(context.get("hazard_conditions", [])),
            },
        )
        return scenario.to_dict()

    def _scenario_severity(self, context: Mapping[str, Any]) -> float:
        guideword = str(context.get("guideword", "")).lower()
        variables = " ".join(str(item).lower() for item in context.get("process_variables", []))
        severity = coerce_float(self._cfg("severity.default", 0.55), 0.55, minimum=0.0, maximum=1.0)
        if any(term in f"{guideword} {variables}" for term in self._cfg("severity.critical_terms", ["critical", "injury", "fatal", "breach", "shutdown"])):
            severity = max(severity, coerce_float(self._cfg("severity.band_scores.critical", 0.92), 0.92, minimum=0.0, maximum=1.0))
        elif any(term in guideword for term in ["too early", "too late", "out of order"]):
            severity = max(severity, coerce_float(self._cfg("severity.band_scores.high", 0.75), 0.75, minimum=0.0, maximum=1.0))
        return clamp_score(severity)

    # ------------------------------------------------------------------
    # Fault tree style analysis
    # ------------------------------------------------------------------

    def _run_fta_analysis(
        self,
        top_event: str,
        fta_config: Optional[Mapping[str, Any]] = None,
        basic_event_probs: Optional[Mapping[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run deterministic FTA-style analysis without optional external imports.

        Accepts both a full config mapping with `gate_structure` and
        `basic_event_probs`, or the older direct gate/probability parameters.
        """
        config = dict(fta_config or {})
        if "gate_structure" in config or "basic_event_probs" in config:
            gate_structure = dict(config.get("gate_structure", {}))
            probabilities = dict(config.get("basic_event_probs", basic_event_probs or {}))
            configured_top_event = str(config.get("top_event", top_event))
        else:
            gate_structure = dict(config)
            probabilities = dict(basic_event_probs or {})
            configured_top_event = top_event

        if not gate_structure:
            return self._fta_stub(configured_top_event)

        max_cut_size = self._max_items("max_cut_set_size", coerce_int(self._cfg("fta.max_cut_set_size", 5), 5, minimum=1))
        cut_sets = self._minimal_cut_sets(configured_top_event, gate_structure, max_cut_size=max_cut_size)
        results = []
        for cut_set in cut_sets:
            probability = 1.0
            for event in cut_set:
                probability *= coerce_float(probabilities.get(event, self._cfg("fta.default_basic_event_probability", 0.05)), 0.05, minimum=0.0, maximum=1.0)
            results.append({
                "top_event": configured_top_event,
                "cut_set": sorted(cut_set),
                "probability": clamp_score(probability),
                "risk_level": categorize_risk(probability),
                "method": "deterministic_minimal_cut_sets",
            })
        results.sort(key=lambda item: item["probability"], reverse=True)
        if results:
            self._store_artifact({"top_event": configured_top_event, "fta_results": results}, tags=["fta"], sensitivity=0.75)
        return redact_value(results[: self._max_items("max_fta_results", 100)])

    def _minimal_cut_sets(self, event: str, gate_structure: Mapping[str, Mapping[str, Any]], *, max_cut_size: int, seen: Optional[Set[str]] = None) -> List[Set[str]]:
        seen = set(seen or set())
        if event in seen:
            return [{event}]
        seen.add(event)
        gate = gate_structure.get(event)
        if not isinstance(gate, Mapping):
            return [{event}]
        gate_type = str(gate.get("type", "OR")).upper()
        inputs = [str(item) for item in gate.get("inputs", [])]
        if not inputs:
            return [{event}]

        child_sets = [self._minimal_cut_sets(child, gate_structure, max_cut_size=max_cut_size, seen=seen) for child in inputs]
        if gate_type == "AND":
            combined: List[Set[str]] = [set()]
            for options in child_sets:
                combined = [left | right for left in combined for right in options if len(left | right) <= max_cut_size]
            return self._minimize_cut_sets(combined)
        # OR and unknown gates default to OR semantics because it is fail-conservative.
        flattened = [item for options in child_sets for item in options if len(item) <= max_cut_size]
        return self._minimize_cut_sets(flattened)

    @staticmethod
    def _minimize_cut_sets(cut_sets: Sequence[Set[str]]) -> List[Set[str]]:
        unique: List[Set[str]] = []
        for candidate in cut_sets:
            if not any(existing <= candidate for existing in unique):
                unique = [existing for existing in unique if not candidate < existing]
                unique.append(set(candidate))
        return unique

    def _fta_stub(self, top_event: str) -> List[Dict[str, Any]]:
        probability = coerce_float(self._cfg("fta.default_basic_event_probability", 0.05), 0.05, minimum=0.0, maximum=1.0)
        return [{
            "top_event": normalize_text(top_event, max_length=256),
            "cut_set": [normalize_text(top_event, max_length=256)],
            "probability": probability,
            "risk_level": categorize_risk(probability),
            "method": "single_event_fallback",
        }]

    def _get_state_constraints(self, controller: str) -> List[str]:
        """Extract state-related constraints from process model."""
        constraints = self.process_models.get(controller, {}).get("constraints", [])
        if constraints:
            return list(constraints)
        configured = self._cfg("default_state_constraints", []) or []
        if configured:
            return [str(item).format(controller=controller) for item in configured]
        return [f"State-dependent constraint for {controller}"]

    # ------------------------------------------------------------------
    # System-of-Systems methods
    # ------------------------------------------------------------------

    def _check_state_consistency(self) -> List[Dict[str, Any]]:
        """Check for state inconsistencies across components."""
        inconsistencies: List[Dict[str, Any]] = []
        incompatible = dict(self._cfg("sos.incompatible_state_pairs", {}))
        component_states = {component: states.get("current", "INIT") for component, states in self.component_states.items()}

        for component, state in component_states.items():
            unsafe_states = set(self.process_models.get(component, {}).get("unsafe_states", []))
            if state in unsafe_states:
                inconsistencies.append({
                    "type": "unsafe_state_active",
                    "component": component,
                    "state": state,
                    "risk_score": coerce_float(self._cfg("sos.unsafe_state_risk", 0.85), 0.85, minimum=0.0, maximum=1.0),
                })
            for blocked_state, related_states in incompatible.items():
                if state == blocked_state:
                    for other_component, other_state in component_states.items():
                        if other_component != component and other_state in set(related_states or []):
                            inconsistencies.append({
                                "type": "incompatible_state_pair",
                                "component": component,
                                "state": state,
                                "conflicting_component": other_component,
                                "conflicting_state": other_state,
                                "risk_score": coerce_float(self._cfg("sos.incompatible_state_risk", 0.75), 0.75, minimum=0.0, maximum=1.0),
                            })
        return redact_value(inconsistencies)

    def _detect_communication_deadlocks(self) -> List[Dict[str, Any]]:
        """Detect circular or asymmetric dependencies in the control structure."""
        wait_for_graph: DefaultDict[str, Set[str]] = defaultdict(set)
        deadlocks: List[Dict[str, Any]] = []

        for controller, structure in self.control_structure.items():
            outputs = set(structure.get("outputs", []))
            for out in outputs:
                for other, other_struct in self.control_structure.items():
                    if other == controller:
                        continue
                    if out in set(other_struct.get("inputs", [])) or out in set(other_struct.get("feedback", [])):
                        wait_for_graph[controller].add(other)

        visited: Set[str] = set()

        def dfs(node: str, trace: List[str]) -> None:
            visited.add(node)
            trace.append(node)
            for neighbor in wait_for_graph[node]:
                if neighbor in trace:
                    cycle = trace[trace.index(neighbor):] + [neighbor]
                    deadlocks.append({
                        "type": "cyclic_dependency",
                        "components_involved": cycle,
                        "cause": "Communication wait cycle detected",
                        "cycle_length": len(cycle),
                        "risk_score": coerce_float(self._cfg("sos.deadlock_cycle_risk", 0.8), 0.8, minimum=0.0, maximum=1.0),
                    })
                elif neighbor not in visited:
                    dfs(neighbor, trace[:])

        for node in list(wait_for_graph):
            if node not in visited:
                dfs(node, [])

        if coerce_bool(self._cfg("sos.flag_asymmetric_dependencies", True), True):
            for src, neighbors in wait_for_graph.items():
                for dst in neighbors:
                    if src not in wait_for_graph[dst]:
                        deadlocks.append({
                            "type": "asymmetric_dependency",
                            "from": src,
                            "to": dst,
                            "issue": "Missing reciprocal output-input mapping",
                            "risk_score": coerce_float(self._cfg("sos.asymmetric_dependency_risk", 0.45), 0.45, minimum=0.0, maximum=1.0),
                        })

        if deadlocks:
            printer.status("STPA", f"{len(deadlocks)} communication deadlock risks detected.", "error")
            for item in deadlocks:
                self._store_artifact(item, tags=["communication_deadlock"], sensitivity=0.75)
        else:
            printer.status("STPA", "No communication deadlocks detected.", "success")
        return redact_value(deadlocks)

    def _analyze_safe_state_reachability(self) -> Dict[str, Any]:
        """Analyze reachability of safe states from current states."""
        unreachable: List[Dict[str, Any]] = []
        transition_issues: List[str] = []
        for component, model in self.process_models.items():
            current = normalize_text(self.component_states.get(component, {}).get("current", model.get("initial_state", "INIT")), max_length=128)
            safe_states = set(model.get("safe_states", [])) or set(self._cfg("sos.default_safe_states", ["SAFE", "NORMAL"]))
            transitions = model.get("transitions", [])
            reachable = self._reachable_states(current, transitions)
            if not safe_states & reachable and current not in safe_states:
                unreachable.append({
                    "component": component,
                    "current_state": current,
                    "safe_states": sorted(safe_states),
                    "reachable_states": sorted(reachable),
                    "risk_score": coerce_float(self._cfg("sos.unreachable_safe_state_risk", 0.8), 0.8, minimum=0.0, maximum=1.0),
                })
            if not transitions:
                transition_issues.append(f"No explicit state transitions modeled for {component}")
        return redact_value({
            "unreachable_safe_states": unreachable,
            "transition_issues": transition_issues,
            "risk_score": max([coerce_float(item.get("risk_score", 0.0), 0.0) for item in unreachable] or [0.0]),
        })

    @staticmethod
    def _reachable_states(start: str, transitions: Sequence[Mapping[str, str]]) -> Set[str]:
        graph: DefaultDict[str, Set[str]] = defaultdict(set)
        for item in transitions:
            source = str(item.get("from", ""))
            target = str(item.get("to", ""))
            if source and target:
                graph[source].add(target)
        visited = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    def _score_sos_results(self, results: Mapping[str, Any]) -> float:
        scores: List[float] = []
        for item in results.get("state_inconsistencies", []) or []:
            scores.append(coerce_float(item.get("risk_score", 0.0), 0.0))
        for item in results.get("deadlock_risks", []) or []:
            scores.append(coerce_float(item.get("risk_score", 0.0), 0.0))
        safe_state = results.get("safe_state_analysis", {}) or {}
        scores.append(coerce_float(safe_state.get("risk_score", 0.0), 0.0))
        return combine_risk_scores(*scores, method=str(self._cfg("risk_aggregation", "noisy_or"))) if scores else 0.0

    # ------------------------------------------------------------------
    # Helper methods retained from original API
    # ------------------------------------------------------------------

    def _nlp_based_hazard_prediction(self, action: str, guideword: str) -> str:
        """
        Deterministic lexical hazard prediction replacement for the old external NLP dependency.
        """
        hazard, _, _ = self._determine_hazard_link("", action, guideword, include_details=True)
        self._store_artifact(
            {
                "method": "lexical_hazard_prediction",
                "input_action_fingerprint": fingerprint(action),
                "input_guideword": guideword,
                "predicted_hazard": hazard,
            },
            tags=["hazard_prediction"],
            sensitivity=0.65,
        )
        return hazard

    def _heuristic_probability_estimation(self, context: Dict[str, Any]) -> float:
        """Estimate probability using configured heuristic factors."""
        process_variables = context.get("process_variables", []) or []
        state_constraints = context.get("state_constraints", []) or []
        guideword = str(context.get("guideword", ""))
        hazard_conditions = context.get("hazard_conditions", []) or []
        factors = {
            "complexity": min(1.0, len(process_variables) / max(coerce_float(self._cfg("probability.process_variable_normalizer", 10.0), 10.0, minimum=1.0), 1.0)),
            "state_dependency": coerce_float(self._cfg("probability.state_dependency_with_constraints", 0.7), 0.7) if state_constraints else coerce_float(self._cfg("probability.state_dependency_without_constraints", 0.3), 0.3),
            "timing_constraint": coerce_float(self._cfg("probability.timing_constraint", 0.8), 0.8) if any(term in guideword.lower() for term in ["timing", "early", "late", "order"]) else coerce_float(self._cfg("probability.non_timing_constraint", 0.4), 0.4),
            "fault_tree": max([coerce_float(item.get("probability", 0.0), 0.0) for item in hazard_conditions] or [0.0]),
        }
        weights = dict(self._cfg("probability.factor_weights", {}))
        return clamp_score(weighted_average(factors, weights, default=coerce_float(self._cfg("probability.default", 0.3), 0.3)))

    def _conservative_probability_estimation(self, context: Dict[str, Any]) -> float:
        heuristic = self._heuristic_probability_estimation(context)
        floor = coerce_float(self._cfg("probability.conservative_floor", 0.45), 0.45, minimum=0.0, maximum=1.0)
        return clamp_score(max(heuristic, floor))

    def _map_to_loss(self, context: Dict[str, Any]) -> str:
        """Map context to system-level loss using lexical overlap."""
        if not self.losses:
            return "Loss not defined"
        context_tokens = self._token_set(f"{context.get('control_action', '')} {context.get('guideword', '')} {' '.join(context.get('process_variables', []))}")
        ranked: List[Tuple[str, float]] = []
        for loss in self.losses:
            loss_tokens = self._token_set(loss)
            score = len(context_tokens & loss_tokens) / max(len(context_tokens | loss_tokens), 1)
            if any(keyword in loss.lower() for keyword in self._cfg("loss_priority_terms", ["life", "injury", "privacy", "security", "equipment", "damage"])):
                score += 0.05
            ranked.append((loss, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[0][0]

    def _generate_mitigation_strategy(self, context: Dict[str, Any]) -> List[str]:
        """Generate context-aware mitigation strategies."""
        strategies: List[str] = []
        guideword = str(context.get("guideword", "")).lower()
        security_tags = set((context.get("security_context", {}) or {}).get("security_tags", []))
        if any(term in guideword for term in ["timing", "early", "late", "order"]):
            strategies.extend(["Implement timing watchdog", "Add sequence verification"])
        if "not providing" in guideword:
            strategies.extend(["Add missing-action detection", "Add independent safety fallback"])
        if "providing" in guideword:
            strategies.extend(["Add pre-action validation checks", "Implement permission system"])
        if "trust_boundary_crossing" in security_tags:
            strategies.extend(["Add trust-boundary authentication", "Require signed control messages"])
        if "security_relevant" in security_tags:
            strategies.extend(["Add tamper-evident audit logging", "Add least-privilege authorization checks"])
        configured = self._cfg("mitigation.default_strategies", []) or []
        strategies.extend(str(item) for item in configured)
        return dedupe_preserve_order(strategies)[: self._max_items("max_mitigations_per_scenario", 12)]

    def _identify_causal_factors(self, context: Dict[str, Any]) -> List[str]:
        """Identify potential causal factors."""
        factors = ["Sensor failure", "Communication delay"]
        for variable in context.get("process_variables", []) or []:
            factors.append(f"{variable} out of range")
        if context.get("hazard_conditions"):
            factors.append("Fault-tree condition present")
        if (context.get("security_context", {}) or {}).get("controller_trust_boundary") in {"external", "third_party", "untrusted"}:
            factors.append("Trust boundary validation failure")
        return dedupe_preserve_order(factors)[: self._max_items("max_causal_factors", 50)]


if __name__ == "__main__":
    print("\n=== Running Secure STPA ===\n")
    printer.status("TEST", "Secure STPA initialized", "info")

    stpa = SecureSTPA()
    stpa.define_analysis_scope(
        losses=[
            "Loss of user safety or privacy",
            "Loss of service integrity",
            "Equipment or model-governance damage",
        ],
        hazards=[
            "Controller sends unsafe shutdown command causing service integrity loss",
            "Unauthorized control action exposes confidential user data",
            "Delayed intervention allows harmful content escalation",
        ],
        constraints=[
            "Unsafe shutdown commands require independent validation",
            "Confidential user data must not cross an untrusted boundary without authorization",
            "Safety interventions must be issued within the configured response window",
        ],
        system_boundary="Safety Agent cyber-safety control loop",
        assets=["user_data", "safety_controls", "model_outputs"],
        stakeholders=["users", "safety_oncall", "security_oncall"],
    )
    stpa.model_control_structure(
        {
            "SafetyController": {
                "inputs": ["risk_signal", "operator_feedback"],
                "outputs": ["issue_intervention", "block_output"],
                "process_vars": ["risk_score", "policy_state", "response_deadline"],
                "trust_boundary": "internal",
                "safe_states": ["SAFE", "MONITORING"],
            },
            "PolicyGateway": {
                "inputs": ["issue_intervention", "block_output"],
                "outputs": ["release_decision"],
                "process_vars": ["authorization_state", "policy_version"],
                "trust_boundary": "external",
                "safe_states": ["SAFE", "LOCKED_DOWN"],
                "unsafe_states": ["BYPASSED"],
            },
        },
        process_models={
            "SafetyController": {
                "variables": ["risk_score", "policy_state", "response_deadline"],
                "constraints": ["risk_score must be evaluated before release_decision"],
                "initial_state": "MONITORING",
                "safe_states": ["SAFE", "MONITORING"],
                "transitions": [
                    {"from": "MONITORING", "to": "SAFE"},
                    {"from": "MONITORING", "to": "LOCKED_DOWN"},
                ],
            },
            "PolicyGateway": {
                "variables": ["authorization_state", "policy_version"],
                "constraints": ["release_decision requires valid policy_version"],
                "initial_state": "SAFE",
                "safe_states": ["SAFE", "LOCKED_DOWN"],
                "unsafe_states": ["BYPASSED"],
                "transitions": [{"from": "SAFE", "to": "LOCKED_DOWN"}, {"from": "LOCKED_DOWN", "to": "SAFE"}],
            },
        },
    )
    ucas = stpa.identify_unsafe_control_actions()
    assert ucas, "Expected UCAs to be identified"
    contexts = stpa.build_context_tables(
        formal_spec={"SafetyController": {"issue_intervention": ["response_deadline", "operator_feedback"]}},
        fta_config={
            "top_event": "issue_intervention",
            "gate_structure": {
                "issue_intervention": {"type": "OR", "inputs": ["sensor_failure", "policy_stale"]},
            },
            "basic_event_probs": {"sensor_failure": 0.04, "policy_stale": 0.08},
        },
    )
    assert contexts, "Expected context tables"
    scenarios = stpa.identify_loss_scenarios()
    assert scenarios, "Expected loss scenarios"
    transition = stpa.record_component_transition("PolicyGateway", "LOCKED_DOWN", trigger="test_transition")
    assert transition["to_state"] == "LOCKED_DOWN"
    sos = stpa.perform_sos_analysis()
    assert "risk_score" in sos
    integrity = stpa.validate_analysis_integrity()
    assert integrity["valid"], integrity
    report = stpa.export_analysis_report(include_sos=True)
    assert report["summary"]["unsafe_control_action_count"] > 0
    serialized = stable_json(report)
    assert "api_key" not in serialized.lower()

    print("\n=== Test ran successfully ===\n")
