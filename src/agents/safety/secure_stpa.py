"""Security-aware STPA/STPA-Sec analysis for the SLAI Safety Agent subsystem."""
from __future__ import annotations

import contextvars
import re

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, DefaultDict, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast

from .utils.config_loader import get_config_section, load_global_config
from .utils.safety_helpers import *
from .utils.security_error import *
from .secure_memory import SecureMemory
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI System-Theoretic Process Analysis")
printer = PrettyPrinter()

MODULE_VERSION = "2.3.0"
SCOPE_SCHEMA_VERSION = "secure_stpa.scope.v3"
UCA_SCHEMA_VERSION = "secure_stpa.uca.v3"
CONTEXT_SCHEMA_VERSION = "secure_stpa.context.v3"
SCENARIO_SCHEMA_VERSION = "secure_stpa.loss_scenario.v3"
REPORT_SCHEMA_VERSION = "secure_stpa.report.v3"


@dataclass(frozen=True)
class STPAScope:
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
    id: str
    controller: str
    control_action: str
    guideword: str
    hazard_link: str
    state_constraints: List[str]
    severity: float
    likelihood: float  # compatibility: heuristic exposure score, not empirical probability
    risk_score: float
    risk_level: str
    decision: str
    rationale: str
    indicators: List[str] = field(default_factory=list)
    link_method: str = "heuristic"
    requires_review: bool = True
    likelihood_basis: str = "heuristic_exposure_not_probability"
    timestamp: str = field(default_factory=utc_iso)
    schema_version: str = UCA_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return redact_value(asdict(self))


@dataclass(frozen=True)
class ContextTableEntry:
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
    scenario_id: str
    context_id: str
    loss: str
    severity: float
    probability: float  # compatibility: heuristic exposure/prioritization factor
    risk_level: float
    normalized_risk: float
    risk_band: str
    decision: str
    mitigation: List[str]
    causal_factors: List[str]
    evidence: Dict[str, Any] = field(default_factory=dict)
    probability_basis: str = "heuristic_prioritization_not_probability"
    timestamp: str = field(default_factory=utc_iso)
    schema_version: str = SCENARIO_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return redact_value(asdict(self))


@dataclass
class _AnalysisState:
    scope_id: str = field(default_factory=lambda: generate_identifier("stpa_scope"))
    scope: Optional[STPAScope] = None
    losses: List[str] = field(default_factory=list)
    hazards: List[str] = field(default_factory=list)
    safety_constraints: List[str] = field(default_factory=list)
    system_boundary: str = "System Boundary Not Defined"
    control_structure: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    context_tables: DefaultDict[str, List[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    uca_table: List[Dict[str, Any]] = field(default_factory=list)
    loss_scenarios: List[Dict[str, Any]] = field(default_factory=list)
    process_models: DefaultDict[str, Dict[str, Any]] = field(default_factory=lambda: defaultdict(dict))
    component_states: DefaultDict[str, Dict[str, Any]] = field(default_factory=lambda: defaultdict(dict))
    transition_history: DefaultDict[str, Deque[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=100)))
    analysis_stats: Dict[str, Any] = field(default_factory=lambda: {"scope_defined": False, "components": 0, "unsafe_control_actions": 0, "contexts": 0, "loss_scenarios": 0, "reports_generated": 0, "last_updated": utc_iso()})


class SecureSTPA:
    """Context-local STPA/STPA-Sec engine with auditable heuristic prioritization."""

    def __init__(self, *, memory: Optional[SecureMemory] = None) -> None:
        self.config = load_global_config()
        self.stpa_config = get_config_section("secure_stpa")
        self.memory = memory or SecureMemory.shared()
        self._validate_configuration()
        self._state_var: contextvars.ContextVar[Optional[_AnalysisState]] = contextvars.ContextVar(f"secure_stpa_state_{id(self)}", default=None)
        self.reset_analysis()

    def _cfg(self, path: Union[str, Sequence[str]], default: Any = None) -> Any:
        return get_nested(self.stpa_config or {}, path, default)

    def _validate_configuration(self) -> None:
        if not isinstance(self.stpa_config, Mapping):
            raise ConfigurationTamperingError("secure_stpa", "secure_stpa config must be a mapping", component="secure_stpa")
        if not coerce_bool(self._cfg("enabled", True), True):
            logger.warning("Secure STPA is disabled by configuration.")
        guidewords = self._cfg("guidewords", [])
        if guidewords and (not isinstance(guidewords, Sequence) or isinstance(guidewords, (str, bytes))):
            raise ConfigurationTamperingError("secure_stpa.guidewords", "guidewords must be a sequence", component="secure_stpa")

    def _state(self) -> _AnalysisState:
        state = self._state_var.get()
        if state is None:
            state = _AnalysisState()
            state.transition_history = defaultdict(lambda: deque(maxlen=coerce_int(self._cfg("sos.max_transition_history", 100), 100, minimum=1)))
            self._state_var.set(state)
        return state

    # Compatibility properties read by SafetyAgent and external callers.
    @property
    def scope(self) -> Optional[STPAScope]: return self._state().scope
    @property
    def scope_id(self) -> str: return self._state().scope_id
    @property
    def losses(self) -> List[str]: return self._state().losses
    @property
    def hazards(self) -> List[str]: return self._state().hazards
    @property
    def safety_constraints(self) -> List[str]: return self._state().safety_constraints
    @property
    def system_boundary(self) -> str: return self._state().system_boundary
    @property
    def control_structure(self) -> Dict[str, Dict[str, Any]]: return self._state().control_structure
    @property
    def context_tables(self) -> DefaultDict[str, List[Dict[str, Any]]]: return self._state().context_tables
    @property
    def uca_table(self) -> List[Dict[str, Any]]: return self._state().uca_table
    @property
    def loss_scenarios(self) -> List[Dict[str, Any]]: return self._state().loss_scenarios
    @property
    def process_models(self) -> DefaultDict[str, Dict[str, Any]]: return self._state().process_models
    @property
    def component_states(self) -> DefaultDict[str, Dict[str, Any]]: return self._state().component_states
    @property
    def analysis_stats(self) -> Dict[str, Any]: return self._state().analysis_stats

    def reset_analysis(self) -> None:
        state = _AnalysisState()
        state.transition_history = defaultdict(lambda: deque(maxlen=coerce_int(self._cfg("sos.max_transition_history", 100), 100, minimum=1)))
        self._state_var.set(state)

    # ------------------------------------------------------------------
    # Scope and control structure
    # ------------------------------------------------------------------
    def _max_items(self, key: str, default: int) -> int:
        return coerce_int(self._cfg(f"limits.{key}", default), default, minimum=1)

    def _normalize_items(self, values: Optional[Iterable[Any]], *, field_name: str, max_items_key: str, max_length: int = 512) -> List[str]:
        if values is None: return []
        raw = [values] if isinstance(values, (str, bytes)) else list(values)
        cleaned = [normalize_text(v, max_length=max_length, preserve_newlines=False) for v in raw[:self._max_items(max_items_key, len(raw) or 1)]]
        cleaned = dedupe_preserve_order([v for v in cleaned if v])
        if not cleaned:
            raise SecurityError(SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT, f"STPA field '{field_name}' requires at least one valid item.", component="secure_stpa")
        return cleaned

    def _normalize_optional_list(self, values: Any, *, max_items_key: str, max_length: int) -> List[str]:
        if values is None: return []
        raw = [values] if isinstance(values, (str, bytes)) else list(values) if isinstance(values, Iterable) else [values]
        return dedupe_preserve_order([normalize_text(v, max_length=max_length) for v in raw[:self._max_items(max_items_key, len(raw) or 1)] if normalize_text(v, max_length=max_length)])

    def define_analysis_scope(self, losses: List[str], hazards: List[str], constraints: List[str], system_boundary: Optional[str] = None, *, assumptions: Optional[List[str]] = None, assets: Optional[List[str]] = None, stakeholders: Optional[List[str]] = None) -> None:
        state = self._state()
        state.losses = self._normalize_items(losses, field_name="losses", max_items_key="max_losses")
        state.hazards = self._normalize_items(hazards, field_name="hazards", max_items_key="max_hazards")
        state.safety_constraints = self._normalize_items(constraints, field_name="constraints", max_items_key="max_constraints")
        state.system_boundary = normalize_text(system_boundary or self._cfg("default_system_boundary", "System Boundary Not Defined"), max_length=1024)
        state.scope = STPAScope(state.scope_id, list(state.losses), list(state.hazards), list(state.safety_constraints), state.system_boundary, self._normalize_optional_list(assumptions or [], max_items_key="max_assumptions", max_length=512), self._normalize_optional_list(assets or [], max_items_key="max_assets", max_length=256), self._normalize_optional_list(stakeholders or [], max_items_key="max_stakeholders", max_length=256))
        state.analysis_stats["scope_defined"] = True; state.analysis_stats["last_updated"] = utc_iso()
        self._store_artifact(state.scope.to_dict(), tags=["stpa_scope"], sensitivity=0.75)

    def model_control_structure(self, structure: Dict[str, Dict[str, List[str]]], process_models: Optional[Dict[str, Dict[str, List[str]]]] = None) -> None:
        if not isinstance(structure, Mapping) or not structure:
            raise SecurityError(SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT, "Control structure must be a non-empty mapping.", component="secure_stpa")
        state = self._state(); normalized: Dict[str, Dict[str, Any]] = {}
        for index, (raw_component, raw_data) in enumerate(structure.items()):
            if index >= self._max_items("max_components", 100): break
            component = normalize_identifier(raw_component, max_length=128)
            if not isinstance(raw_data, Mapping): raise SecurityError(SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT, "Control component must be a mapping.", component="secure_stpa", context={"component": component})
            outputs = self._normalize_optional_list(raw_data.get("outputs", []), max_items_key="max_control_actions", max_length=256)
            if not outputs: raise SecurityError(SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT, "Control component must define at least one output/control action.", component="secure_stpa", context={"component": component})
            normalized[component] = {
                "inputs": self._normalize_optional_list(raw_data.get("inputs", []), max_items_key="max_interfaces", max_length=256),
                "outputs": outputs,
                "process_vars": self._normalize_optional_list(raw_data.get("process_vars", raw_data.get("process_variables", [])), max_items_key="max_process_variables", max_length=256),
                "feedback": self._normalize_optional_list(raw_data.get("feedback", []), max_items_key="max_interfaces", max_length=256),
                "trust_boundary": normalize_text(raw_data.get("trust_boundary", "internal"), max_length=128, lowercase=True),
                "authority_level": normalize_text(raw_data.get("authority_level", "nominal"), max_length=128, lowercase=True),
                "safe_states": self._normalize_optional_list(raw_data.get("safe_states", self._cfg("sos.default_safe_states", ["SAFE", "NORMAL"])), max_items_key="max_states", max_length=128),
                "unsafe_states": self._normalize_optional_list(raw_data.get("unsafe_states", []), max_items_key="max_states", max_length=128),
                "hazard_links": cast(Dict[str, Any], redact_value(dict(cast(Mapping[Any, Any], raw_data.get("hazard_links", {}))))) if isinstance(raw_data.get("hazard_links"), Mapping) else {},
                "metadata": cast(Dict[str, Any], redact_value(dict(cast(Mapping[Any, Any], raw_data.get("metadata", {}))))) if isinstance(raw_data.get("metadata"), Mapping) else {},
            }
        state.control_structure = normalized
        state.process_models = defaultdict(dict)
        for component, data in dict(process_models or {}).items(): state.process_models[normalize_identifier(component, max_length=128)] = self._normalize_process_model(data)
        for component, data in normalized.items():
            if component not in state.process_models:
                state.process_models[component] = {
                    "variables": list(data["process_vars"]), "constraints": [],
                    "states": list(data["safe_states"]) + list(data["unsafe_states"]) + ["INIT"],
                    "safe_states": list(data["safe_states"]), "unsafe_states": list(data["unsafe_states"]), "initial_state": str(self._cfg("sos.initial_state", "INIT")), "transitions": [],
                    "model_quality": "placeholder_missing_process_model",
                }
        state.component_states = defaultdict(dict)
        for component in normalized:
            state.component_states[component] = {"current": normalize_text(state.process_models[component].get("initial_state", "INIT"), max_length=128), "transitions": defaultdict(list), "last_updated": utc_iso()}
        state.analysis_stats["components"] = len(normalized); state.analysis_stats["last_updated"] = utc_iso()
        self._store_artifact({"control_structure": state.control_structure, "process_models": state.process_models}, tags=["stpa_model"], sensitivity=0.75)

    def _normalize_process_model(self, data: Any) -> Dict[str, Any]:
        if not isinstance(data, Mapping): return {"model_quality": "invalid"}
        return {
            "variables": self._normalize_optional_list(data.get("variables", data.get("process_variables", [])), max_items_key="max_process_variables", max_length=256),
            "constraints": self._normalize_optional_list(data.get("constraints", []), max_items_key="max_constraints", max_length=512),
            "states": self._normalize_optional_list(data.get("states", []), max_items_key="max_states", max_length=128),
            "safe_states": self._normalize_optional_list(data.get("safe_states", self._cfg("sos.default_safe_states", ["SAFE", "NORMAL"])), max_items_key="max_states", max_length=128),
            "unsafe_states": self._normalize_optional_list(data.get("unsafe_states", []), max_items_key="max_states", max_length=128),
            "initial_state": normalize_text(data.get("initial_state", self._cfg("sos.initial_state", "INIT")), max_length=128),
            "transitions": self._normalize_transitions(data.get("transitions", [])),
            "model_quality": normalize_identifier(data.get("model_quality", "analyst_supplied"), max_length=64),
            "metadata": redact_value(dict(data.get("metadata", {}))) if isinstance(data.get("metadata"), Mapping) else {},
        }

    def _normalize_transitions(self, transitions: Any) -> List[Dict[str, str]]:
        result: List[Dict[str, str]] = []
        if isinstance(transitions, Mapping):
            for source, targets in transitions.items():
                for target in self._normalize_optional_list(targets, max_items_key="max_states", max_length=128): result.append({"from": normalize_text(source, max_length=128), "to": target})
        elif isinstance(transitions, Iterable) and not isinstance(transitions, (str, bytes)):
            for item in transitions:
                if isinstance(item, Mapping):
                    source = normalize_text(item.get("from", item.get("source", "")), max_length=128); target = normalize_text(item.get("to", item.get("target", "")), max_length=128)
                    if source and target: result.append({"from": source, "to": target})
        return result[:self._max_items("max_transitions", 500)]

    # ------------------------------------------------------------------
    # UCA and context analysis
    # ------------------------------------------------------------------
    def _guidewords(self, custom_guidewords: Optional[List[str]]) -> List[str]:
        configured = list(self._cfg("guidewords", [])) + list(self._cfg("security_guidewords", []))
        return self._normalize_items(custom_guidewords or configured, field_name="guidewords", max_items_key="max_guidewords", max_length=256)

    def identify_unsafe_control_actions(self, custom_guidewords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        self._require_scope_and_model(); state = self._state(); state.uca_table = []
        for controller, data in state.control_structure.items():
            for action in data.get("outputs", []):
                for guideword in self._guidewords(custom_guidewords):
                    if not self._guideword_applicable(action, guideword): continue
                    hazard, rationale, indicators, method, review = self._determine_hazard_link(controller, action, guideword)
                    severity = self._estimate_uca_severity(hazard, guideword, controller, action)
                    exposure = self._estimate_uca_likelihood(controller, action, guideword)
                    risk = clamp_score(severity * exposure)
                    uca = UnsafeControlAction(f"uca_{len(state.uca_table)+1:04d}", controller, action, guideword, hazard, self._get_state_constraints(controller), severity, exposure, risk, categorize_risk(risk), self._risk_decision(risk), rationale, indicators, method, review)
                    item = uca.to_dict(); state.uca_table.append(item); self._store_artifact(item, tags=["unsafe_control_action", f"controller:{controller}"], sensitivity=0.8)
        state.analysis_stats["unsafe_control_actions"] = len(state.uca_table); state.analysis_stats["last_updated"] = utc_iso()
        return list(state.uca_table)

    def _guideword_applicable(self, action: str, guideword: str) -> bool:
        mapping = self._cfg("guideword_applicability", {}) or {}
        if not isinstance(mapping, Mapping) or not mapping: return True
        action_norm = normalize_identifier(action, max_length=128)
        configured = mapping.get(action_norm) or mapping.get("*")
        if not configured: return True
        return normalize_identifier(guideword, max_length=128) in {normalize_identifier(v, max_length=128) for v in configured}

    def _determine_hazard_link(self, controller: str, action: str, guideword: str) -> Tuple[str, str, List[str], str, bool]:
        state = self._state()
        if not state.hazards: raise SecurityError(SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION, "No hazards available for STPA hazard linking.", component="secure_stpa")
        explicit = get_nested(state.control_structure.get(controller, {}), f"hazard_links.{action}.{guideword}", None) or get_nested(state.control_structure.get(controller, {}), f"hazard_links.{action}", None)
        if explicit:
            candidate = normalize_text(explicit[0] if isinstance(explicit, list) else explicit, max_length=512)
            if candidate in state.hazards: return candidate, "Explicit analyst/model hazard link.", ["explicit_link"], "explicit", False
        query_tokens = self._token_set(f"{controller} {action} {guideword}"); current_state = normalize_text(state.component_states[controller].get("current", "INIT"), lowercase=True)
        ranked: List[Tuple[str, float, List[str]]] = []
        for hazard in state.hazards:
            hazard_tokens = self._token_set(hazard); overlap = query_tokens & hazard_tokens; union = query_tokens | hazard_tokens; score = len(overlap)/max(len(union), 1); indicators = sorted(overlap)
            lowered = hazard.lower()
            if controller.lower() in lowered: score += 0.15; indicators.append("controller_match")
            if any(token in lowered for token in self._token_set(action)): score += 0.15; indicators.append("action_match")
            if "emergency" in current_state: score += 0.05; indicators.append("state_context")
            ranked.append((hazard, score, dedupe_preserve_order(indicators)))
        ranked.sort(key=lambda item: item[1], reverse=True); hazard, score, indicators = ranked[0]
        return hazard, f"Heuristic lexical/state candidate score {score:.3f}; analyst review required.", indicators[:10], "heuristic_lexical", True

    def _token_set(self, value: Any) -> Set[str]:
        text = normalize_text(value, max_length=2048, lowercase=True); stop = set(self._cfg("analysis.stop_words", ["the", "and", "or", "of", "to", "a", "an", "in", "for", "with", "by", "on"]))
        return {token for token in re.findall(r"[a-z0-9_'-]+", text) if token not in stop and len(token) > 1}

    def _estimate_uca_severity(self, hazard_link: str, guideword: str, controller: str, action: str) -> float:
        text = f"{hazard_link} {guideword} {controller} {action}".lower(); score = coerce_float(self._cfg("severity.default", 0.55), 0.55, minimum=0.0, maximum=1.0)
        for band, terms in dict(self._cfg("severity.keyword_bands", {}) or {}).items():
            if any(str(term).lower() in text for term in terms or []): score = max(score, coerce_float(self._cfg(f"severity.band_scores.{band}", score), score, minimum=0.0, maximum=1.0))
        return clamp_score(score)

    def _estimate_uca_likelihood(self, controller: str, action: str, guideword: str) -> float:
        """Compatibility name; returns heuristic exposure/prioritization, not probability."""
        component = self._state().control_structure.get(controller, {}); base = coerce_float(self._cfg("probability.base_uca_likelihood", 0.35), 0.35, minimum=0.0, maximum=1.0)
        complexity = min(1.0, (len(component.get("inputs", []))+len(component.get("outputs", []))+len(component.get("process_vars", []))) / max(coerce_float(self._cfg("probability.complexity_normalizer", 12.0), 12.0, minimum=1.0), 1.0))
        timing = coerce_float(self._cfg("probability.timing_guideword_bonus", 0.12), 0.12) if any(term in guideword.lower() for term in ["early", "late", "order", "timing"]) else 0.0
        trust = coerce_float(self._cfg("probability.external_trust_boundary_bonus", 0.10), 0.10) if component.get("trust_boundary") in {"external", "third_party", "untrusted"} else 0.0
        return clamp_score(base + 0.35*complexity + timing + trust)

    def _get_state_constraints(self, controller: str) -> List[str]:
        state = self._state(); values = list(state.process_models.get(controller, {}).get("constraints", []))
        if not values: values = [constraint for constraint in state.safety_constraints if self._token_set(controller) & self._token_set(constraint)]
        return dedupe_preserve_order(values)

    def build_context_tables(self, formal_spec: Optional[Dict[str, Any]] = None, fta_config: Optional[Dict[str, Any]] = None) -> Dict[str, List[Dict[str, Any]]]:
        state = self._state()
        if not state.uca_table: self.identify_unsafe_control_actions()
        state.context_tables = defaultdict(list)
        for uca in state.uca_table:
            entry = self._generate_context_entry(uca, formal_spec or {}, fta_config or {})
            state.context_tables[entry["controller"]].append(entry); self._store_artifact(entry, tags=["stpa_context", f"controller:{entry['controller']}"], sensitivity=0.8)
        state.analysis_stats["contexts"] = sum(len(v) for v in state.context_tables.values()); state.analysis_stats["last_updated"] = utc_iso()
        return dict(state.context_tables)

    def _generate_context_entry(self, uca: Dict[str, Any], formal_spec: Dict[str, Any], fta_config: Dict[str, Any]) -> Dict[str, Any]:
        state = self._state(); controller = str(uca["controller"]); action = str(uca["control_action"])
        variables = list(state.process_models.get(controller, {}).get("variables", [])) + list(state.control_structure.get(controller, {}).get("process_vars", []))
        if isinstance(formal_spec.get(controller), Mapping): variables.extend(self._normalize_optional_list(formal_spec[controller].get(action, []), max_items_key="max_process_variables", max_length=256))
        conditions: List[Dict[str, Any]] = []
        if fta_config:
            for condition in fta_config.get(action, []) if isinstance(fta_config.get(action, []), list) else []:
                if isinstance(condition, Mapping): conditions.append(redact_value(dict(condition)))
        security_context = {"trust_boundary": state.control_structure.get(controller, {}).get("trust_boundary", "internal"), "authority_level": state.control_structure.get(controller, {}).get("authority_level", "nominal"), "link_method": uca.get("link_method"), "requires_review": uca.get("requires_review", True)}
        entry = ContextTableEntry(generate_identifier("stpa_ctx"), str(uca["id"]), controller, action, str(uca["guideword"]), dedupe_preserve_order(variables), conditions, list(uca.get("state_constraints", [])), security_context, clamp_score(uca.get("risk_score", 0.0)), categorize_risk(uca.get("risk_score", 0.0)), self._risk_decision(uca.get("risk_score", 0.0)))
        return entry.to_dict()

    # ------------------------------------------------------------------
    # Loss scenarios and SoS
    # ------------------------------------------------------------------
    def identify_loss_scenarios(self, probability_model: str = "heuristic") -> List[Dict[str, Any]]:
        state = self._state()
        if not state.context_tables: self.build_context_tables()
        supported = set(self._cfg("probability.supported_models", ["heuristic", "conservative"]))
        if probability_model not in supported: raise ConfigurationTamperingError("secure_stpa.probability.supported_models", f"Unsupported model: {probability_model}", component="secure_stpa")
        scenarios: List[Dict[str, Any]] = []
        for contexts in state.context_tables.values():
            for context in contexts:
                scenario = self._generate_loss_scenario(context, probability_model); scenarios.append(scenario); self._store_artifact(scenario, tags=["loss_scenario", f"risk:{scenario.get('risk_band', 'unknown')}"], sensitivity=0.85)
        scenarios.sort(key=lambda item: coerce_float(item.get("normalized_risk", 0.0), 0.0), reverse=True); state.loss_scenarios = scenarios
        state.analysis_stats["loss_scenarios"] = len(scenarios); state.analysis_stats["last_updated"] = utc_iso(); return scenarios

    def _generate_loss_scenario(self, context: Dict[str, Any], probability_model: str) -> Dict[str, Any]:
        state = self._state(); loss = self._select_loss(context); severity = clamp_score(context.get("risk_score", 0.0) or self._cfg("severity.default", 0.55))
        exposure = clamp_score(coerce_float(self._cfg(f"probability.{probability_model}_base", 0.4 if probability_model == "heuristic" else 0.55), 0.4))
        exposure = clamp_score(exposure + min(len(context.get("hazard_conditions", [])), 5) * coerce_float(self._cfg("probability.condition_increment", 0.05), 0.05))
        priority = clamp_score(severity * exposure)
        mitigation = self._scenario_mitigations(context, priority); causal = self._scenario_causal_factors(context)
        scenario = LossScenario(generate_identifier("stpa_scenario"), str(context["context_id"]), loss, severity, exposure, priority, priority, categorize_risk(priority), self._risk_decision(priority), mitigation, causal, {"uca_id": context.get("uca_id"), "link_review_required": get_nested(context, "security_context.requires_review", True), "model_quality": self._state().process_models.get(context.get("controller", ""), {}).get("model_quality", "unknown")})
        return scenario.to_dict()

    def _select_loss(self, context: Mapping[str, Any]) -> str:
        state = self._state(); tokens = self._token_set(stable_json(context)); ranked = sorted(state.losses, key=lambda value: len(tokens & self._token_set(value)), reverse=True)
        return ranked[0] if ranked else "Unspecified loss"

    def _scenario_mitigations(self, context: Mapping[str, Any], risk: float) -> List[str]:
        values = list(context.get("state_constraints", []) or [])
        values.extend(str(v) for v in self._cfg("mitigations.default", ["Add explicit control constraints and feedback monitoring.", "Validate the control action against runtime state before execution."]) or [])
        if get_nested(context, "security_context.trust_boundary", "internal") in {"external", "third_party", "untrusted"}: values.append("Enforce authentication, authorization, integrity and anti-replay controls at the trust boundary.")
        if risk >= coerce_float(self._cfg("thresholds.block", 0.72), 0.72): values.append("Require explicit human or policy approval before the associated high-risk control action.")
        return dedupe_preserve_order(values)[:self._max_items("max_mitigations_per_scenario", 20)]

    def _scenario_causal_factors(self, context: Mapping[str, Any]) -> List[str]:
        values = [f"Guideword: {context.get('guideword')}"]
        values.extend(f"Condition: {redact_text(stable_json(v), max_length=160)}" for v in context.get("hazard_conditions", []) or [])
        values.extend(f"Process variable: {v}" for v in context.get("process_variables", []) or [])
        return dedupe_preserve_order(values)[:self._max_items("max_causal_factors", 30)]

    def perform_sos_analysis(self, consistency_checks: bool = True, deadlock_detection: bool = True, safe_state_reachability: bool = True) -> Dict[str, Any]:
        self._require_control_structure(); result: Dict[str, Any] = {"schema_version": "secure_stpa.sos.v3", "timestamp": utc_iso(), "analysis_id": generate_identifier("sos")}
        if consistency_checks: result["state_inconsistencies"] = self._check_state_consistency()
        if deadlock_detection: result["deadlock_risks"] = self._detect_communication_deadlocks()
        if safe_state_reachability: result["safe_state_analysis"] = self._analyze_safe_state_reachability()
        result["risk_score"] = self._score_sos_results(result); result["risk_level"] = categorize_risk(result["risk_score"]); result["decision"] = self._risk_decision(result["risk_score"])
        self._store_artifact(result, tags=["sos_analysis"], sensitivity=0.75); return redact_value(result)

    def _check_state_consistency(self) -> List[Dict[str, Any]]:
        state = self._state(); output: List[Dict[str, Any]] = []
        for component, info in state.component_states.items():
            current = str(info.get("current", "INIT")); unsafe = set(state.process_models.get(component, {}).get("unsafe_states", []))
            if current in unsafe: output.append({"component": component, "state": current, "issue": "component_in_declared_unsafe_state"})
        return output

    def _detect_communication_deadlocks(self) -> List[Dict[str, Any]]:
        state = self._state(); output: List[Dict[str, Any]] = []
        for component, data in state.control_structure.items():
            if data.get("inputs") and not data.get("feedback") and coerce_bool(self._cfg("sos.flag_missing_feedback", True), True): output.append({"component": component, "issue": "inputs_without_declared_feedback_path"})
        return output

    def _analyze_safe_state_reachability(self) -> Dict[str, Any]:
        state = self._state(); result: Dict[str, Any] = {}
        for component, model in state.process_models.items():
            initial = str(model.get("initial_state", "INIT")); safe = set(model.get("safe_states", [])); transitions = model.get("transitions", []) or []
            graph: Dict[str, Set[str]] = defaultdict(set)
            for item in transitions:
                if isinstance(item, Mapping): graph[str(item.get("from"))].add(str(item.get("to")))
            visited = {initial}; queue = deque([initial])
            while queue:
                current = queue.popleft()
                for target in graph.get(current, set()):
                    if target not in visited: visited.add(target); queue.append(target)
            result[component] = {"reachable_safe_state": bool(safe & visited) if transitions else bool(initial in safe), "safe_states": sorted(safe), "visited_count": len(visited), "model_quality": model.get("model_quality", "unknown")}
        return result

    def _score_sos_results(self, result: Mapping[str, Any]) -> float:
        inconsistencies = len(result.get("state_inconsistencies", []) or []); deadlocks = len(result.get("deadlock_risks", []) or []); unreachable = sum(not coerce_bool(v.get("reachable_safe_state"), False) for v in (result.get("safe_state_analysis", {}) or {}).values() if isinstance(v, Mapping))
        return combine_risk_scores(clamp_score(inconsistencies/5.0), clamp_score(deadlocks/5.0), clamp_score(unreachable/5.0), method="noisy_or")

    # ------------------------------------------------------------------
    # Transitions, integrity, report
    # ------------------------------------------------------------------
    def record_component_transition(self, component: str, to_state: str, *, trigger: str = "unspecified", from_state: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        state = self._state(); normalized = normalize_identifier(component, max_length=128)
        if normalized not in state.component_states: raise SecurityError(SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION, "Transition references unknown component.", component="secure_stpa", context={"component": normalized})
        current = normalize_text(state.component_states[normalized].get("current", "INIT"), max_length=128); source = normalize_text(from_state or current, max_length=128); target = normalize_text(to_state, max_length=128)
        transition = {"transition_id": generate_identifier("stpa_transition"), "component": normalized, "from_state": source, "to_state": target, "trigger": normalize_text(trigger, max_length=256), "timestamp": utc_iso(), "metadata": redact_value(dict(metadata or {}))}
        state.component_states[normalized]["current"] = target; state.component_states[normalized]["last_updated"] = transition["timestamp"]; state.component_states[normalized]["transitions"][source].append(target); state.transition_history[normalized].append(transition)
        self._store_artifact(transition, tags=["component_transition", f"component:{normalized}"], sensitivity=0.65); return transition

    def get_analysis_summary(self) -> Dict[str, Any]:
        state = self._state(); scenario = [coerce_float(v.get("normalized_risk", 0.0), 0.0) for v in state.loss_scenarios]; uca = [coerce_float(v.get("risk_score", 0.0), 0.0) for v in state.uca_table]; risk = max(scenario or uca or [0.0])
        return {"scope_defined": bool(state.scope), "component_count": len(state.control_structure), "loss_count": len(state.losses), "hazard_count": len(state.hazards), "constraint_count": len(state.safety_constraints), "unsafe_control_action_count": len(state.uca_table), "context_count": sum(len(v) for v in state.context_tables.values()), "loss_scenario_count": len(state.loss_scenarios), "max_risk": clamp_score(risk), "risk_level": categorize_risk(risk), "decision": self._risk_decision(risk), "stats": dict(state.analysis_stats), "timestamp": utc_iso(), "methodological_note": "Likelihood/probability fields are heuristic prioritization values unless externally supplied empirical data is configured."}

    def generate_mitigation_backlog(self) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for scenario in self._state().loss_scenarios:
            risk = coerce_float(scenario.get("normalized_risk", 0.0), 0.0)
            for mitigation in scenario.get("mitigation", []) or []: output.append({"id": generate_identifier("mit"), "source_scenario": scenario.get("scenario_id"), "context_id": scenario.get("context_id"), "mitigation": normalize_text(mitigation, max_length=512), "priority": categorize_risk(risk), "risk_score": clamp_score(risk), "decision": self._risk_decision(risk)})
        output.sort(key=lambda item: item["risk_score"], reverse=True); return redact_value(output[:self._max_items("max_mitigations", 250)])

    def validate_analysis_integrity(self) -> Dict[str, Any]:
        state = self._state(); issues: List[Dict[str, Any]] = []
        if not state.scope: issues.append({"type": "missing_scope", "severity": "high"})
        if not state.control_structure: issues.append({"type": "missing_control_structure", "severity": "high"})
        uca_ids = {v.get("id") for v in state.uca_table}; context_ids = {v.get("context_id") for entries in state.context_tables.values() for v in entries}
        for controller, entries in state.context_tables.items():
            for entry in entries:
                if entry.get("uca_id") not in uca_ids: issues.append({"type": "orphan_context", "controller": controller, "uca_id": entry.get("uca_id"), "severity": "medium"})
        for scenario in state.loss_scenarios:
            if scenario.get("context_id") not in context_ids: issues.append({"type": "orphan_loss_scenario", "scenario_id": scenario.get("scenario_id"), "severity": "medium"})
        result = {"valid": not issues, "issue_count": len(issues), "issues": redact_value(issues), "fingerprint": fingerprint({"summary": self.get_analysis_summary(), "issues": issues}), "timestamp": utc_iso()}
        if issues and coerce_bool(self._cfg("integrity.fail_closed", False), False): raise SecurityError(SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION, "STPA analysis contains integrity issues.", component="secure_stpa", context={"issue_count": len(issues)})
        return result

    def export_analysis_report(self, format: str = "json", include_sos: bool = False) -> Dict[str, Any]:
        state = self._state(); normalized = normalize_identifier(format, max_length=16).lower()
        if normalized not in {"json", "yaml", "dict"}: raise ConfigurationTamperingError("secure_stpa.report.allowed_formats", f"Unsupported format: {format}", component="secure_stpa")
        if not state.loss_scenarios: self.identify_loss_scenarios()
        report = {"schema_version": REPORT_SCHEMA_VERSION, "module_version": MODULE_VERSION, "metadata": {"report_id": generate_identifier("stpa_report"), "created": utc_iso(), "system_boundary": state.system_boundary, "scope_id": state.scope_id, "format": normalized}, "summary": self.get_analysis_summary(), "losses": list(state.losses), "hazards": list(state.hazards), "safety_constraints": list(state.safety_constraints), "control_structure": redact_value(state.control_structure), "unsafe_control_actions": redact_value(state.uca_table), "context_tables": redact_value(dict(state.context_tables)), "loss_scenarios": redact_value(state.loss_scenarios), "mitigation_backlog": self.generate_mitigation_backlog(), "integrity_validation": self.validate_analysis_integrity()}
        if include_sos: report["sos_analysis"] = self.perform_sos_analysis()
        report["integrity"] = {"report_fingerprint": fingerprint(report), "artifact_count": len(state.uca_table)+sum(len(v) for v in state.context_tables.values())+len(state.loss_scenarios)}
        state.analysis_stats["reports_generated"] += 1; state.analysis_stats["last_updated"] = utc_iso(); self._store_artifact(report, tags=["stpa_report"], sensitivity=0.85); return redact_value(report)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _require_scope_and_model(self) -> None:
        if not self._state().scope: raise SecurityError(SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION, "STPA scope must be defined first.", component="secure_stpa")
        self._require_control_structure()

    def _require_control_structure(self) -> None:
        if not self._state().control_structure: raise SecurityError(SecurityErrorType.SYSTEM_INTEGRITY_VIOLATION, "STPA control structure must be modeled first.", component="secure_stpa")

    def _risk_decision(self, risk: float) -> str:
        return threshold_decision(risk, block_threshold=coerce_float(self._cfg("thresholds.block", 0.72), 0.72), review_threshold=coerce_float(self._cfg("thresholds.review", 0.40), 0.40))

    def _memory_tags(self, *tags: str) -> List[str]:
        return dedupe_preserve_order([normalize_identifier(v, max_length=96) for v in list(self._cfg("memory.base_tags", ["secure_stpa", "safety_analysis"]) or []) + list(tags) if str(v).strip()])

    def _store_artifact(self, artifact: Mapping[str, Any], *, tags: Sequence[str], sensitivity: Optional[float] = None, purpose: str = "secure_stpa_analysis", classification: Optional[str] = None) -> Optional[str]:
        if not coerce_bool(self._cfg("memory.store_artifacts", True), True): return None
        try:
            safe = redact_value(dict(artifact))
            return self.memory.add(safe, tags=self._memory_tags(*tags), sensitivity=coerce_float(sensitivity if sensitivity is not None else self._cfg("memory.default_sensitivity", 0.7), 0.7, minimum=0.0, maximum=1.0), ttl_seconds=coerce_int(self._cfg("memory.ttl_seconds", 604800), 604800, minimum=0), purpose=purpose, owner="secure_stpa", classification=classification or str(self._cfg("memory.default_classification", "confidential")), source="secure_stpa", metadata={"module_version": MODULE_VERSION, "artifact_fingerprint": fingerprint(safe), "eligible_for_compliance": True})
        except Exception as exc:
            if coerce_bool(self._cfg("memory.fail_closed_on_store_error", False), False): raise AuditLogFailureError("secure_stpa.memory", f"Failed to store STPA artifact: {type(exc).__name__}", component="secure_stpa", cause=exc) from exc
            return None


__all__ = ["MODULE_VERSION", "SCOPE_SCHEMA_VERSION", "UCA_SCHEMA_VERSION", "CONTEXT_SCHEMA_VERSION", "SCENARIO_SCHEMA_VERSION", "REPORT_SCHEMA_VERSION", "STPAScope", "UnsafeControlAction", "ContextTableEntry", "LossScenario", "SecureSTPA"]
